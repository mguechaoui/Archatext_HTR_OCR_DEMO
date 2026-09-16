# Deployment

## The short version — total cost $0

| Piece | Where | Cost |
|---|---|---|
| API container (FastAPI + kraken + torch) | Azure Container Apps, 1 vCPU / 2GiB, scale-to-zero | $0 — free grant |
| Segmentation model + kraken checkpoint | Azure Blob Storage, versioned prefixes | ~$0.01/month |
| Container image | GitHub Container Registry (public) | $0 |
| Frontend | Served by the same container | $0 |
| CI/CD | GitHub Actions | $0 |

Nothing here needs Render, and nothing charges a card. The $100 student
credit is the safety net, not the funding source — under normal demo traffic
you should not draw it down at all.

### Where the "$0" actually comes from

Azure Container Apps grants **180,000 vCPU-seconds and 360,000 GiB-seconds
per subscription per month, permanently** — not a trial, not part of the
student credit. At 1 vCPU / 2GiB that works out to 50 hours of *active*
container time per month on both meters:

- 180,000 vCPU-s ÷ 1 vCPU = 50 hours
- 360,000 GiB-s ÷ 2 GiB = 50 hours

With `min-replicas 0`, idle time consumes neither meter — the container does
not exist when nobody is using it. A demo link on a CV will not accumulate
50 hours of actual request processing in a month. If you somehow exceed it,
the overflow is a few cents an hour against the credit, not a bill.

Blob Storage for ~500MB of checkpoints on Standard_LRS hot tier is about one
cent a month. Egress is free for the first 100GB.

The one thing that *would* have cost money is Azure Container Registry
(~$5/month for Basic, roughly $60/year out of your credit). The setup uses
GitHub Container Registry instead, which is free for public packages. The
image contains no weights and no secrets — weights are pulled at runtime
from your private blob container — so there is nothing to protect by paying
for a private registry.

---

## Why not Render, and why not Vercel

**Render free will not run this.** Free and Starter instances are both 512MB
RAM. CPU-only torch sits at roughly 300–400MB resident before kraken loads a
single weight; the segmentation model plus a recognition checkpoint pushes
past 512MB during startup. The container OOMs, and it does it during model
loading, which presents as a mysterious restart loop rather than a clear
error. The plan that actually works is Standard at ~$25/month — which is
exactly what you said you do not want to pay, and you do not have to, because
Azure Container Apps gives you the same 2GB for free.

**Vercel cannot host the backend at all.** Serverless function bundles cap at
250MB unzipped; CPU torch alone is ~200MB before kraken, scipy, shapely and
transformers. Beyond size, functions are stateless and cold-start per
invocation — the entire `lifespan` design in this codebase exists to load
models once and reuse them, and on Vercel you would pay that cost on every
cold start. Page-level segmentation plus OCR on CPU also exceeds the function
timeout.

Vercel is genuinely good at what it does; a stateful ML inference server is
not that. The **frontend** would be a fine fit if you want Vercel on your CV
— see the last section.

---

## Architecture: why weights are not in the image

The image is built **without** model weights. `scripts/fetch_models.py` pulls
them from Azure Blob Storage at container boot, into `/models`, before
uvicorn starts.

This is the single most useful MLOps decision in the setup, and it is worth
being able to explain:

- **The image and the model version independently.** Retraining does not
  trigger a rebuild. A code fix does not force re-uploading a gigabyte of
  weights.
- **Rollback is a config change.** `OCR_MODEL_VERSION=v3` → `v2`, restart.
  Seconds, not a rebuild-and-redeploy cycle.
- **Image size stays sane.** Cold start time is a direct function of image
  pull time, and scale-to-zero means you pay it regularly.
- **Weights stay out of git.** They are already gitignored; this keeps them
  out of the registry too.

Blob layout:

```
models/                          <- the blob container
  v1/
    seg/muharaf_seg_best.mlmodel
    ocr/arman_run1_91.mlmodel
    ocr/hatformer.tar.gz
  v2/
    ...
```

`models.manifest.json` declares what to fetch. Fill in the `sha256` fields
after uploading — with them, a corrupt or accidentally-swapped blob fails
loudly at boot instead of quietly producing wrong predictions.

---

## Step 1 — Run it locally in Docker first

```bash
mkdir -p model_weights/seg model_weights/ocr
cp ../fuzzysearch_aproach/seg_model/muharaf_seg_best.mlmodel model_weights/seg/
cp ../fuzzysearch_aproach/arman_training_last/checkpoints/arman_run1_91.mlmodel \
   model_weights/ocr/arman_best.mlmodel

docker compose up --build
```

Then open http://localhost:8000 and check http://localhost:8000/health/ready.

`docker-compose.yml` caps the container at 2GB deliberately, matching the
production instance, so an out-of-memory problem surfaces on your laptop
rather than in a live demo. If it OOMs here, set
`OCR_ENABLED_OCR_ENGINES=my_ocr_model` to drop HATFormer and try again.

Expect the first build to take 10–20 minutes. Subsequent builds are cached.

## Step 2 — Provision Azure

Push to `main` first so GitHub Actions builds and publishes the image to
GHCR, then:

```bash
az login
GITHUB_USER=yourname ./infra/azure/provision.sh
```

It creates the resource group and storage account, uploads your segmentation
model and kraken checkpoint under `v1/`, mints a read-only SAS token, creates
the Container Apps environment, and deploys the app pointing at your GHCR
image. It prints your live URL at the end.

Edit the `SEG_MODEL` / `OCR_MODEL` / `HATFORMER_DIR` paths at the top of the
script first, or pass them as environment variables. Make the GHCR package
public once (GitHub → your profile → Packages → ocr-platform → Package
settings → Change visibility) so Azure can pull it without credentials.

Teardown, which stops all spend:

```bash
az group delete --name rg-ocr-platform --yes --no-wait
```

### The scale-to-zero trade-off

`--min-replicas 0` means the app costs nothing when idle, which is what keeps
it inside the free grant. The cost is that the first request after an idle
period waits for the container to start *and* load models — plausibly 60–90
seconds on CPU.

For a link on a CV that a recruiter clicks once, that is a bad first
impression. Before an interview or a demo:

```bash
az containerapp update -n ocr-platform-api -g rg-ocr-platform --min-replicas 1
```

and set it back to 0 afterwards. Idle replicas bill at a reduced rate, not at
zero, so leaving it at 1 permanently will draw down the student credit.

A middle option: put a scheduled GitHub Action on a cron that pings
`/health` every 10 minutes during working hours. It keeps the container warm
for the cost of a few minutes of compute a day.

## Step 3 — Wire up CI/CD

Create a service principal and store it as a GitHub secret:

```bash
az ad sp create-for-rbac \
  --name "gh-ocr-platform" \
  --role contributor \
  --scopes /subscriptions/$(az account show --query id -o tsv)/resourceGroups/rg-ocr-platform \
  --sdk-auth
```

Repository secrets to add: `AZURE_CREDENTIALS` (the JSON above), `AZURE_RG`,
`AZURE_CONTAINERAPP`. Nothing else — pushing to GHCR uses the automatic
`GITHUB_TOKEN`, so there is no registry credential to manage.

The pipeline in `.github/workflows/ci-cd.yml` runs lint and manifest
validation, then builds the image and **boots it with no weights present**
to assert that liveness passes while readiness correctly reports 503. That
degraded-start behaviour is a deliberate property of the registry design, so
it is worth having a test that fails if someone changes it. Deploy runs on
`main` only and polls `/health/ready` until the new revision actually serves,
failing the run if it never does.

---

## Health endpoints, and why there are two

- **`/health`** — liveness. The process is up. Says nothing about models.
  A liveness probe that fails on a missing checkpoint makes the orchestrator
  restart-loop a container that restarting will never fix.
- **`/health/ready`** — readiness. Returns 503 until the segmentation model
  is loaded, since segmentation is the entry point of every pipeline here.
  Reports which OCR engines loaded and which are degraded.
- **`/version`** — git SHA, image tag, and model version together. When a
  prediction looks wrong, the first question is which code and which weights
  produced it.

One fix was needed to make any of this work: the static file mount at `/`
was registered before the route definitions, and a Starlette `Mount("/")`
matches every path. `/health` was returning 404, which a platform health
probe reads as "this container is dead". The mounts now come last in
`app/main.py`.

## Configuration worth knowing

| Variable | Why it matters |
|---|---|
| `OCR_ENABLED_OCR_ENGINES` | The memory lever. `my_ocr_model` alone fits comfortably in 2GB; adding `hatformer` costs well over a gigabyte more. Same image either way. |
| `OCR_MODEL_VERSION` | Which blob prefix to pull. Your rollback switch. |
| `WEB_CONCURRENCY` | Keep at 1. Each uvicorn worker loads its own full copy of every model — a second worker on 2GB is an OOM, not throughput. Scale with replicas. |
| `OMP_NUM_THREADS` | Cap torch's thread pool to the vCPU allocation. Left unset, torch spawns threads for cores the container cannot use and thrashes. |
| `OCR_CORS_ALLOW_ORIGINS` | Only needed if the frontend is hosted separately. Empty by default. |

## What is ephemeral

Uploads and outputs are written to container-local disk, which disappears on
restart and is not shared between replicas. For a demo this is fine — a user
uploads a page, works with it, exports. It breaks if you scale past one
replica or expect results to survive a redeploy.

If you want to fix it (and it is a good second iteration to be able to talk
about): `app/services/page_store.py` and `upload_service.py` are the only two
places that touch the filesystem, so swapping in Azure Blob Storage is a
contained change rather than a refactor. The architecture already isolated
this — worth pointing out in an interview.

## Splitting the frontend onto Vercel

Optional, and only worth doing if you want Vercel on your CV. The frontend
calls a relative `/api/v1`, so same-origin works with zero configuration
today. To split:

1. In `static/js/app.js`, change `const API = "/api/v1"` to read from a
   build-time constant or a `window.OCR_API_BASE` global.
2. Deploy `static/` to Vercel as a static site.
3. Set `OCR_CORS_ALLOW_ORIGINS=https://your-app.vercel.app` on the container.

The CORS middleware in `app/main.py` activates only when that variable is
non-empty, so this costs nothing until you use it.

## For the CV

What this actually demonstrates, phrased usefully:

- Multi-stage Docker build with a CPU-only torch index — an image around 1GB
  instead of roughly 4GB, and a concrete number you can quote.
- Model artifacts versioned in object storage and pulled at boot, with
  checksum verification and atomic writes; rollback by environment variable.
- Liveness and readiness separated, with readiness reporting per-engine
  degradation rather than a single boolean.
- Graceful degradation: a missing or broken checkpoint disables one engine
  instead of taking down the service, and CI asserts this.
- Infrastructure as code, with a documented cost analysis behind the hosting
  choice (free grant math, why 512MB platforms OOM, why the registry is GHCR
  and not ACR).
- CI that boots the built image and verifies it serves before deploying, and
  polls the deployed revision before declaring success.

The honest framing of the cold start is a feature of the answer, not a hole
in it: "scale-to-zero, so it costs nothing idle, at the price of a slow first
request — here is the trade-off and here is how I'd fix it with a warm
replica" is a stronger response than pretending the problem is not there.
