# Manuscript OCR Platform

A production-structured FastAPI application for segmenting and recognizing
text on historical Arabic manuscript pages, built around the research code
in `seg_muhref.py` (segmentation) and `train_arman_working.py` (the ArMan
fine-tuning pipeline).

## Running it

Local, in Docker (mirrors the deployed container):

```bash
docker compose up --build     # -> http://localhost:8000
```

Local, without Docker:

```bash
pip install -r requirements.txt
cp .env.example .env           # then point OCR_MODEL_DIR at your weights
uvicorn app.main:app --reload
```

Deployment — container image, Azure Blob Storage for weights, Azure Container
Apps for the API, GitHub Actions for CI/CD — is documented in
[`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md). Total hosting cost is $0: the
Container Apps free grant covers the API, Blob Storage costs about a cent a
month for the checkpoints, and GHCR hosts the image for free.

Ops endpoints: `/health` (liveness), `/health/ready` (readiness, per-engine
load state), `/version` (git SHA + image tag + model version).

## Architecture

```
app/
  api/v1/         FastAPI route modules (one per resource: upload, segmentation, ocr, status, export)
  core/            Cross-cutting concerns: DI providers, lifespan/startup, exceptions, error handlers
  config/          Centralized settings (env-driven), nothing hardcoded elsewhere
  models/          Framework-agnostic domain dataclasses (Segment, PageSegmentation, OCR results)
  schemas/         Pydantic request/response models for the API layer
  segmentation/    The segmentation engine (refactored from seg_muhref.py) + its singleton registry
  ocr/             The OCR plugin system: BaseOCR interface, concrete engines, registry
  services/        Application-layer orchestration (upload, segmentation run, OCR run, export, page store)
  utils/           Small shared helpers (device resolution, logging)
static/            The frontend (vanilla HTML/CSS/JS annotation-tool UI)
uploads/           Uploaded page images (gitignored)
outputs/           Generated crops, overlays, and exports (gitignored)
```

### Why it's organized this way

Research code and API code are different things with different lifecycles.
`app/segmentation/service.py` and `app/ocr/engines/*.py` contain the actual
model-loading and inference logic — they know nothing about FastAPI,
Pydantic, or HTTP. `app/services/*_orchestrator.py` is the layer that wires
those into the request lifecycle (loading a page's image, persisting
results, handling partial failure). `app/api/v1/*.py` is thin: it validates
input via Pydantic, calls one orchestrator method, and serializes the
result. This means the segmentation/OCR code could be lifted into a CLI
tool, a notebook, or a batch job with zero changes.

## The OCR plugin architecture

```
BaseOCR (abstract)
  +-- MyOCRModel   - fine-tuned kraken checkpoint, via kraken.lib.models.load_any + kraken.rpred
  +-- HATFormerOCR - transformers Vision2Seq checkpoint, via Auto* loading + .generate()
```

Adding a third engine:
1. Create `app/ocr/engines/your_engine.py` implementing `BaseOCR` (`load()`, `is_loaded`, `predict()`).
2. Add it to `OCREngine` in `app/models/ocr_result.py` and to `_ENGINE_CLASSES` in `app/ocr/registry.py`.

Nothing in the API routes, the OCR orchestrator, or the frontend needs to change —
the frontend already lists whatever engines the registry reports as available.

### A note on `MyOCRModel`'s inference path

`train_arman_working.py` only ever calls `ketos test`, which reports
aggregate CER/WER over a whole test set — there's no existing function in
either of your scripts that takes one image and returns recognized text.
`MyOCRModel.predict()` uses kraken's own standard recognition API
(`kraken.lib.models.load_any` + `kraken.rpred.rpred`) on your `.mlmodel`
checkpoint, which is the natural per-image counterpart to what `ketos test`
already does in aggregate. If your actual inference behavior differs
(custom post-processing, a different bounds format, etc.), `app/ocr/engines/my_ocr_model.py`
is the one file to change.

### A note on `HATFormerOCR`

No HATFormer code was provided. `HATFormerOCR` is a real (not a stub)
adapter using the standard HuggingFace `transformers` loading pattern
(`AutoProcessor` + `AutoModelForVision2Seq` + `.generate()`), so pointing
`OCR_HATFORMER_CHECKPOINT_PATH` at a compatible local checkpoint or HF Hub
id should work out of the box. If your checkpoint uses a non-standard
architecture, `app/ocr/engines/hatformer.py` is the one file to adjust.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# edit .env: point OCR_SEGMENTATION_MODEL_PATH, OCR_MYOCR_CHECKPOINT_PATH,
# and OCR_HATFORMER_CHECKPOINT_PATH at your actual model files.

uvicorn app.main:app --reload
```

Then open `http://localhost:8000`.

If a model checkpoint is missing, the app still starts — that engine is
simply reported as unavailable (the frontend disables it in the dropdown,
the API returns 503 if called directly). This keeps a partial deployment
useful instead of crash-looping.

## API overview

| Method | Path                                       | Purpose                            |
|--------|---------------------------------------------|-------------------------------------|
| POST   | `/api/v1/pages/upload`                      | Upload a manuscript page            |
| POST   | `/api/v1/pages/{id}/segmentation/run`       | Run line segmentation               |
| GET    | `/api/v1/pages/{id}/segmentation`           | Fetch segmentation results          |
| GET    | `/api/v1/pages/{id}/segmentation/overlay`   | Fetch the rendered overlay PNG      |
| GET    | `/api/v1/pages/{id}/segments/{sid}/crop`    | Fetch one segment's cropped image   |
| GET    | `/api/v1/pages/ocr/engines`                 | List OCR engines + availability     |
| POST   | `/api/v1/pages/{id}/ocr/run?engine=...`     | Run OCR with the chosen engine      |
| GET    | `/api/v1/pages/{id}/ocr`                    | Fetch OCR results                   |
| GET    | `/api/v1/pages/{id}/text`                   | Fetch reconstructed page text       |
| GET    | `/api/v1/pages/{id}/status`                 | Poll pipeline status                |
| GET    | `/api/v1/pages/{id}/export/{fmt}`           | Download txt/json/csv/alto/page-xml |

Interactive docs at `/docs` (FastAPI's built-in Swagger UI).

## Extending later

The structure was built so these don't require restructuring:
- **New OCR/segmentation models** — see "OCR plugin architecture" above.
- **Background/async processing** — wrap `OCROrchestrator.run` / `SegmentationOrchestrator.run`
  in a task queue (Celery/RQ/arq); the route just enqueues instead of calling directly.
- **Persistence beyond memory** — `PageStore` is the only place that knows pages
  live in a dict; swap it for a DB-backed implementation behind the same interface.
- **Auth** — add a dependency in `app/core/dependencies.py` and apply it via `Depends()` on routers.
- **Docker/CI** — the app has no implicit state outside `uploads/`, `outputs/`, and the
  model checkpoint paths, all of which are env-configurable, so containerizing is just
  mounting those three locations as volumes.
