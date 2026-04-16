"""ARIA MoldVision — Model Bundle Publishing.

Upload trained model bundles to S3 and update the central catalog.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .appconfig import get_shared_root

_log = logging.getLogger(__name__)

CATALOG_SCHEMA_VERSION = "catalog-v1"
CATALOG_KEY = "catalog.json"
_ENV_PUBLISH_TARGET = "ARIA_MODEL_PUBLISH_TARGET"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_manifest(bundle_path: Path) -> Dict[str, Any]:
    """Read manifest.json from a bundle directory or .mpk/.zip file."""
    if bundle_path.is_dir():
        for name in ("manifest.json", "bundle_manifest.json"):
            p = bundle_path / name
            if p.exists():
                return json.loads(p.read_text(encoding="utf-8"))
        raise FileNotFoundError(f"No manifest found in {bundle_path}")
    # Archive
    import zipfile
    with zipfile.ZipFile(bundle_path, "r") as zf:
        for name in ("manifest.json", "bundle_manifest.json"):
            if name in zf.namelist():
                return json.loads(zf.read(name).decode("utf-8"))
        raise FileNotFoundError(f"No manifest found in {bundle_path}")


def _ensure_mpk(bundle_path: Path) -> Path:
    """Ensure we have an .mpk (zip) file. If input is a directory, pack it."""
    if bundle_path.is_file() and bundle_path.suffix in (".mpk", ".zip"):
        return bundle_path
    if not bundle_path.is_dir():
        raise ValueError(f"Expected a directory or .mpk/.zip file: {bundle_path}")
    import zipfile
    mpk_path = bundle_path.with_suffix(".mpk")
    with zipfile.ZipFile(mpk_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(bundle_path.rglob("*")):
            if f.is_file():
                zf.write(f, f.relative_to(bundle_path))
    _log.info("Packed bundle directory to %s", mpk_path)
    return mpk_path


def _load_publish_config() -> Dict[str, Any]:
    """Load publish config from data_lake_config.json or environment."""
    # Check environment variables first
    bucket = os.environ.get("ARIA_MODEL_CATALOG_BUCKET")
    region = os.environ.get("ARIA_MODEL_CATALOG_REGION", "eu-west-1")
    prefix = os.environ.get("ARIA_MODEL_CATALOG_PREFIX", "")
    if bucket:
        return {"bucket": bucket, "region": region, "prefix": prefix}

    # Try data_lake_config.json in common locations
    for candidate in [
        Path("data_lake_config.json"),
        Path.home() / ".aria" / "data_lake_config.json",
    ]:
        if candidate.exists():
            cfg = json.loads(candidate.read_text(encoding="utf-8"))
            if "model_catalog_bucket" in cfg:
                return {
                    "bucket": cfg["model_catalog_bucket"],
                    "region": cfg.get("model_catalog_region", "eu-west-1"),
                    "prefix": cfg.get("model_catalog_prefix", ""),
                }

    raise RuntimeError(
        "No publish configuration found. Set ARIA_MODEL_CATALOG_BUCKET env var "
        "or add 'model_catalog_bucket' to data_lake_config.json."
    )


def _s3_client(region: str):
    """Create a boto3 S3 client."""
    try:
        import boto3
    except ImportError:
        raise ImportError("boto3 is required for publishing. Install with: pip install boto3")
    return boto3.client("s3", region_name=region)


def _fetch_catalog(s3, bucket: str, prefix: str) -> Dict[str, Any]:
    """Fetch the current catalog.json from S3, or return an empty catalog."""
    key = f"{prefix}{CATALOG_KEY}" if prefix else CATALOG_KEY
    try:
        resp = s3.get_object(Bucket=bucket, Key=key)
        return json.loads(resp["Body"].read().decode("utf-8"))
    except Exception as e:
        if "NoSuchKey" in str(type(e).__name__) or "NoSuchKey" in str(e):
            _log.info("No existing catalog found, creating new one.")
            return {
                "schema_version": CATALOG_SCHEMA_VERSION,
                "updated_at": None,
                "models": [],
            }
        raise


def _upload_catalog(s3, bucket: str, prefix: str, catalog: Dict[str, Any]) -> None:
    """Write catalog.json back to S3."""
    key = f"{prefix}{CATALOG_KEY}" if prefix else CATALOG_KEY
    body = json.dumps(catalog, indent=2, ensure_ascii=False).encode("utf-8")
    s3.put_object(Bucket=bucket, Key=key, Body=body, ContentType="application/json")
    _log.info("Updated catalog at s3://%s/%s", bucket, key)


def _resolve_publish_target() -> str:
    target = os.environ.get(_ENV_PUBLISH_TARGET, "auto").strip().lower()
    if target not in {"auto", "s3", "shared"}:
        raise ValueError(f"Unknown publish target {target!r}. Valid: auto, s3, shared")
    if target == "auto":
        return "shared" if get_shared_root() is not None else "s3"
    return target


def _shared_publish_destination(shared_root: Path, *, role: str, manifest: Dict[str, Any]) -> Path:
    normalized_role = role.strip().lower()
    bundle_type = str(manifest.get("bundle_type", "")).strip().lower()
    published = shared_root / "published"
    if bundle_type == "startup_suggestion" or normalized_role in {
        "startup_suggestion",
        "startup_suggestions",
        "suggestions",
    }:
        return published / "moldpilot" / "suggestions"
    if normalized_role in {"defect_detector", "detection", "moldpilot_detection"}:
        return published / "moldpilot" / "detection"
    if normalized_role in {"monitor_segmenter", "components", "ocr_recognizer"}:
        return published / "moldtrace" / "roles" / normalized_role
    raise ValueError(f"Unsupported shared publish role: {role!r}")


def _copy_bundle_tree(bundle_path: Path, destination_dir: Path) -> None:
    if destination_dir.exists():
        shutil.rmtree(destination_dir)
    destination_dir.parent.mkdir(parents=True, exist_ok=True)
    if bundle_path.is_dir():
        shutil.copytree(bundle_path, destination_dir)
        return
    import zipfile

    if not zipfile.is_zipfile(bundle_path):
        raise ValueError(f"Expected a bundle directory or zip-compatible archive: {bundle_path}")
    destination_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(bundle_path, "r") as zf:
        zf.extractall(destination_dir)


def _publish_to_shared_root(
    *,
    shared_root: Path,
    bundle_path: Path,
    role: str,
    manifest: Dict[str, Any],
    catalog_entry: Dict[str, Any],
    channel: str,
    dry_run: bool,
) -> Dict[str, Any]:
    target_root = _shared_publish_destination(shared_root, role=role, manifest=manifest)
    bundle_id = catalog_entry["bundle_id"]
    bundle_dir = target_root / "bundles" / bundle_id
    index_path = target_root / "index.json"

    result = dict(catalog_entry)
    result["publish_target"] = "shared"
    result["destination"] = str(bundle_dir)
    result["artifact_key"] = f"bundles/{bundle_id}/"

    if dry_run:
        _log.info("DRY RUN — would publish to shared root:\n%s", json.dumps(result, indent=2))
        return result

    _copy_bundle_tree(bundle_path, bundle_dir)

    if index_path.exists():
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
        except Exception:
            index = {}
    else:
        index = {}
    bundles = [b for b in index.get("bundles", []) if b.get("bundle_id") != bundle_id]
    bundles.append(
        {
            **catalog_entry,
            "artifact_key": f"bundles/{bundle_id}/",
            "published_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    index = {
        "schema_version": "published-bundles-v1",
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "role": role,
        "active_by_channel": {
            **(index.get("active_by_channel") or {}),
            channel: bundle_id,
        },
        "bundles": bundles,
    }
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps(index, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result


def publish_bundle(
    bundle_path: Path,
    *,
    role: str,
    channel: str = "stable",
    compatible_layouts: Optional[List[str]] = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Publish a model bundle to the S3 model catalog.

    Args:
        bundle_path: Path to a bundle directory or .mpk/.zip archive.
        role: Model role (e.g., "defect_detector", "monitor_segmenter").
        channel: Release channel ("stable" or "beta").
        compatible_layouts: HMI layouts this model supports. None = ["*"].
        dry_run: If True, print what would be done without uploading.

    Returns:
        Dict with published bundle metadata.
    """
    bundle_path = Path(bundle_path)
    manifest = _read_manifest(bundle_path)

    bundle_id = manifest.get("bundle_id") or manifest.get("model_name", "unknown")
    model_version = manifest.get("model_version", "0.0.0")
    supersedes = manifest.get("supersedes")
    min_app_version = manifest.get("min_app_version", "0.0.0")

    mpk_path = _ensure_mpk(bundle_path)
    sha256 = _sha256_file(mpk_path)
    size_bytes = mpk_path.stat().st_size

    artifact_key_name = f"bundles/{bundle_id}.mpk"
    layouts = compatible_layouts or ["*"]

    catalog_entry = {
        "bundle_id": bundle_id,
        "model_name": manifest.get("model_name", bundle_id),
        "model_version": model_version,
        "channel": channel,
        "role": role,
        "min_app_version": min_app_version,
        "artifact_key": artifact_key_name,
        "sha256": sha256,
        "size_bytes": size_bytes,
        "compatible_layouts": layouts,
        "supersedes": supersedes,
        "published_at": datetime.now(timezone.utc).isoformat(),
    }

    publish_target = _resolve_publish_target()
    shared_root = get_shared_root()
    if publish_target == "shared":
        if shared_root is None:
            raise RuntimeError(
                "Shared publish target requires ARIA_SHARED_ROOT or moldvision config.shared_root."
            )
        return _publish_to_shared_root(
            shared_root=shared_root,
            bundle_path=bundle_path,
            role=role,
            manifest=manifest,
            catalog_entry=catalog_entry,
            channel=channel,
            dry_run=dry_run,
        )

    if dry_run:
        _log.info("DRY RUN — would publish:\n%s", json.dumps(catalog_entry, indent=2))
        return catalog_entry

    config = _load_publish_config()
    s3 = _s3_client(config["region"])
    bucket = config["bucket"]
    prefix = config.get("prefix", "")

    # Upload .mpk
    artifact_key = f"{prefix}{artifact_key_name}" if prefix else artifact_key_name
    _log.info("Uploading %s to s3://%s/%s ...", mpk_path.name, bucket, artifact_key)
    s3.upload_file(str(mpk_path), bucket, artifact_key)
    _log.info("Upload complete (%d bytes, sha256=%s)", size_bytes, sha256[:16])

    # Update catalog
    catalog = _fetch_catalog(s3, bucket, prefix)
    # Remove any existing entry with same bundle_id
    catalog["models"] = [
        m for m in catalog.get("models", []) if m.get("bundle_id") != bundle_id
    ]
    catalog["models"].append(catalog_entry)
    catalog["updated_at"] = datetime.now(timezone.utc).isoformat()
    _upload_catalog(s3, bucket, prefix, catalog)

    _log.info("Published %s (v%s) to channel '%s', role '%s'", bundle_id, model_version, channel, role)
    return catalog_entry
