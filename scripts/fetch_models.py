#!/usr/bin/env python3
"""
Pull model artifacts from Azure Blob Storage into the container's model
directory, before the API starts.

Why this exists
---------------
Model weights do not belong in a Docker image. Baking them in means every
retrain produces a new multi-gigabyte image, the registry fills up, and the
image and the weights version together instead of independently. Instead the
image is weights-free and immutable, and the weights are pulled at boot from
a versioned, immutable blob prefix.

That gives you the property that actually matters operationally: rolling a
model back is an environment-variable change (OCR_MODEL_VERSION=v3 -> v2 and
restart), not a rebuild.

Configuration (environment)
---------------------------
  OCR_MODEL_BASE_URL   https://<account>.blob.core.windows.net/<container>
  OCR_MODEL_SAS_TOKEN  read-only SAS, with or without the leading '?'
  OCR_MODEL_VERSION    version prefix inside the container (default: v1)
  OCR_MODEL_DIR        local destination (default: /models)
  OCR_SKIP_MODEL_FETCH set to 1 to bypass entirely (local dev with mounts)

The manifest (models.manifest.json) declares what to fetch. Artifacts marked
`"optional": true` log a warning and are skipped on failure; required ones
abort the boot, because a container that silently starts without its
segmentation model is worse than one that refuses to start.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import tarfile
import time
import zipfile
from pathlib import Path

import httpx

CHUNK = 1024 * 1024
MANIFEST = Path(__file__).resolve().parents[1] / "models.manifest.json"


def log(msg: str) -> None:
    print(f"[fetch-models] {msg}", flush=True)


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def build_url(base: str, version: str, blob: str, sas: str) -> str:
    prefix = f"{version.strip('/')}/" if version else ""
    url = f"{base.rstrip('/')}/{prefix}{blob.lstrip('/')}"
    if sas:
        url = f"{url}?{sas.lstrip('?')}"
    return url


def download(url: str, dest: Path, attempts: int = 3) -> None:
    """Stream to a .part file, then atomically rename.

    The atomic rename matters: if the container is killed mid-download (very
    plausible on a platform that scales to zero), the next boot must not find
    a truncated checkpoint sitting at the final path and treat it as cached.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")

    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            with httpx.stream("GET", url, timeout=120.0, follow_redirects=True) as r:
                r.raise_for_status()
                total = int(r.headers.get("content-length", 0))
                written = 0
                with tmp.open("wb") as fh:
                    for chunk in r.iter_bytes(CHUNK):
                        fh.write(chunk)
                        written += len(chunk)
                if total and written != total:
                    raise OSError(f"short read: {written} of {total} bytes")
            tmp.replace(dest)
            log(f"  downloaded {dest.name} ({dest.stat().st_size / 1e6:.1f} MB)")
            return
        except Exception as exc:  # noqa: BLE001 - retry on anything transient
            last_error = exc
            tmp.unlink(missing_ok=True)
            if attempt < attempts:
                backoff = 2 ** attempt
                log(f"  attempt {attempt} failed ({exc}); retrying in {backoff}s")
                time.sleep(backoff)

    raise RuntimeError(f"failed after {attempts} attempts: {last_error}")


def unpack(archive: Path, dest_dir: Path, kind: str) -> None:
    if dest_dir.exists():
        shutil.rmtree(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    if kind in ("tar.gz", "tgz"):
        with tarfile.open(archive, "r:gz") as tf:
            tf.extractall(dest_dir, filter="data")
    elif kind == "zip":
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(dest_dir)
    else:
        raise ValueError(f"unknown archive kind '{kind}'")
    archive.unlink(missing_ok=True)

    # Collapse a single wrapping directory, so the manifest's `dest` is always
    # the directory the model loader is pointed at regardless of how the
    # archive happened to be rolled.
    entries = list(dest_dir.iterdir())
    if len(entries) == 1 and entries[0].is_dir():
        inner = entries[0]
        for item in inner.iterdir():
            item.rename(dest_dir / item.name)
        inner.rmdir()
    log(f"  unpacked -> {dest_dir}")


def fetch_one(art: dict, base: str, version: str, sas: str, model_dir: Path) -> bool:
    name = art["name"]
    dest = model_dir / art["dest"]
    archive_kind = art.get("archive")
    marker = dest / ".fetched" if archive_kind else dest

    if marker.exists():
        log(f"{name}: already present, skipping")
        return True

    url = build_url(base, version, art["blob"], sas)
    log(f"{name}: fetching {art['blob']}")

    if archive_kind:
        tmp_archive = model_dir / f"{name}.{archive_kind}"
        download(url, tmp_archive)
        expected = art.get("sha256")
        if expected:
            actual = sha256_of(tmp_archive)
            if actual != expected:
                tmp_archive.unlink(missing_ok=True)
                raise RuntimeError(f"checksum mismatch: expected {expected}, got {actual}")
            log("  checksum ok")
        unpack(tmp_archive, dest, archive_kind)
        (dest / ".fetched").write_text(version or "v1")
    else:
        download(url, dest)
        expected = art.get("sha256")
        if expected:
            actual = sha256_of(dest)
            if actual != expected:
                dest.unlink(missing_ok=True)
                raise RuntimeError(f"checksum mismatch: expected {expected}, got {actual}")
            log("  checksum ok")

    return True


def main() -> int:
    if os.getenv("OCR_SKIP_MODEL_FETCH", "").lower() in ("1", "true", "yes"):
        log("OCR_SKIP_MODEL_FETCH set — using whatever is already on disk")
        return 0

    base = os.getenv("OCR_MODEL_BASE_URL", "").strip()
    if not base:
        log("OCR_MODEL_BASE_URL not set — skipping fetch (models must be mounted)")
        return 0

    sas = os.getenv("OCR_MODEL_SAS_TOKEN", "").strip()
    version = os.getenv("OCR_MODEL_VERSION", "v1").strip()
    model_dir = Path(os.getenv("OCR_MODEL_DIR", "/models"))
    model_dir.mkdir(parents=True, exist_ok=True)

    if not MANIFEST.exists():
        log(f"no manifest at {MANIFEST}; nothing to do")
        return 0

    manifest = json.loads(MANIFEST.read_text())
    artifacts = manifest.get("artifacts", [])
    log(f"manifest version={version}, {len(artifacts)} artifact(s), dest={model_dir}")

    failed_required = []
    for art in artifacts:
        try:
            fetch_one(art, base, version, sas, model_dir)
        except Exception as exc:  # noqa: BLE001
            if art.get("optional"):
                log(f"{art['name']}: OPTIONAL artifact failed ({exc}) — continuing")
            else:
                log(f"{art['name']}: REQUIRED artifact failed ({exc})")
                failed_required.append(art["name"])

    if failed_required:
        log(f"aborting: required artifacts missing: {', '.join(failed_required)}")
        return 1

    log("all artifacts ready")
    return 0


if __name__ == "__main__":
    sys.exit(main())
