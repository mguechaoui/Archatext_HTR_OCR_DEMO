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

TLS note — read this before touching verification again
---------------------------------------------------------
Azure has been migrating blob storage endpoints off the old Baltimore
CyberTrust root onto newer roots (DigiCert Global Root G2 / Microsoft TLS
RSA Root G2). Whether a given container can verify that chain depends on
whether *either* its OS ca-certificates snapshot or its pinned `certifi`
version already contains the new root — and either one can lag behind on
its own, which is exactly what produced the repeated
"[SSL: CERTIFICATE_VERIFY_FAILED] unable to get local issuer certificate"
failures here.

The fix below does not pick one trust source. It builds a single SSLContext
that loads *both* certifi's bundle and the OS trust store
(/etc/ssl/certs/ca-certificates.crt, present because the runtime image
installs ca-certificates and drops DigiCert Global Root G2 into it).
Verification therefore succeeds as long as either source has the needed
root. If the OS bundle isn't present (e.g. running this script on a host
that lacks it), it degrades to certifi-only rather than failing to build
the context at all.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import ssl
import sys
import tarfile
import time
import zipfile
from pathlib import Path

import certifi
import httpx

CHUNK = 1024 * 1024
MANIFEST = Path(__file__).resolve().parents[1] / "models.manifest.json"
_OS_CA_BUNDLE = Path("/etc/ssl/certs/ca-certificates.crt")


def log(msg: str) -> None:
    print(f"[fetch-models] {msg}", flush=True)


def _build_ssl_context() -> ssl.SSLContext:
    """One trust store, fed from every CA source we have available.

    Starts from certifi's bundle (guaranteed present — it's an httpx
    dependency), then additionally loads the OS trust store on top if it
    exists. load_verify_locations() is additive, not a replacement, so this
    is a union of both sources, not a fallback between them.
    """
    ctx = ssl.create_default_context(cafile=certifi.where())
    if _OS_CA_BUNDLE.exists():
        try:
            ctx.load_verify_locations(cafile=str(_OS_CA_BUNDLE))
            log(f"TLS trust store: certifi + {_OS_CA_BUNDLE}")
        except ssl.SSLError as exc:
            log(f"  could not merge OS CA bundle ({exc}); using certifi only")
    else:
        log("TLS trust store: certifi only (no OS bundle found)")
    return ctx


_SSL_CTX = _build_ssl_context()


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


# Split timeouts, not one flat number. A single timeout=120.0 applies to
# connect AND read, which means an endpoint that is simply unreachable
# (blocked egress, wrong hostname, DNS black hole) hangs for the full 120s
# before failing — and across 3 attempts x however many artifacts, that is
# what turns "broken" into "looks stuck for twenty minutes with zero signal".
# connect/write/pool stay short, because a real Azure Storage endpoint
# either responds to the TLS handshake in a couple of seconds or it never
# will. read stays generous, because a multi-hundred-MB checkpoint over a
# slow link legitimately needs time once bytes are actually flowing.
_TIMEOUT = httpx.Timeout(connect=10.0, read=60.0, write=10.0, pool=10.0)


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
            with httpx.stream(
                "GET",
                url,
                timeout=_TIMEOUT,
                follow_redirects=True,
                verify=_SSL_CTX,
            ) as r:
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

    # Collapse a single wrapping directory, so the manifest's `dest` is
    # always the directory the model loader is pointed at regardless of how
    # the archive happened to be rolled.
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
                raise RuntimeError(
                    f"checksum mismatch: expected {expected}, got {actual}"
                )
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
                raise RuntimeError(
                    f"checksum mismatch: expected {expected}, got {actual}"
                )
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