#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Provisions the entire platform on Azure for $0.
#
#   Blob Storage   -> versioned model artifacts (~$0.01/month for ~500MB)
#   Container Apps -> the API, 1 vCPU / 2GiB, scale-to-zero
#                     (covered by the permanent monthly free grant)
#
# The container image lives on GitHub Container Registry, which is free for
# public packages. Azure Container Registry is deliberately NOT used here:
# ACR Basic is about $5/month, which is ~$60/year out of a $100 student
# credit for something GHCR does at no cost. The image contains no weights
# and no secrets, so publishing it is safe.
#
# Prerequisites: az CLI (`az login`), and the image already pushed to GHCR by
# the GitHub Actions workflow (push to main once before running this).
#
# Usage:
#   GITHUB_USER=yourname ./infra/azure/provision.sh
# ---------------------------------------------------------------------------
set -euo pipefail

# --- Configuration ---------------------------------------------------------
LOCATION="${LOCATION:-westeurope}"
RG="${RG:-rg-ocr-platform}"
STORAGE="${STORAGE:-stocrplatform$RANDOM}"   # 3-24 chars, globally unique
CONTAINER_NAME="${CONTAINER_NAME:-models}"
ACA_ENV="${ACA_ENV:-env-ocr-platform}"
APP="${APP:-ocr-platform-api}"
MODEL_VERSION="${MODEL_VERSION:-v1}"

GITHUB_USER="${GITHUB_USER:-}"
IMAGE="${IMAGE:-ghcr.io/${GITHUB_USER}/ocr-platform:latest}"

# Local paths to your checkpoints — edit these or pass as env vars.
SEG_MODEL="${SEG_MODEL:-../fuzzysearch_aproach/seg_model/muharaf_seg_best.mlmodel}"
OCR_MODEL="${OCR_MODEL:-../fuzzysearch_aproach/arman_training_last/checkpoints/arman_run1_91.mlmodel}"
HATFORMER_DIR="${HATFORMER_DIR:-./checkpoints/ocr/muharaf_ours/best}"

say() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }

if [[ -z "$GITHUB_USER" ]]; then
    echo "Set GITHUB_USER (your GitHub username, lowercase) so the image can"
    echo "be pulled from ghcr.io. Example:"
    echo "    GITHUB_USER=yourname ./infra/azure/provision.sh"
    exit 1
fi

say "Resource group: $RG ($LOCATION)"
az group create --name "$RG" --location "$LOCATION" --output none

# --- 1. Storage for model artifacts ----------------------------------------
# Standard_LRS, hot tier. ~500MB of checkpoints is around one cent a month;
# the student credit will not notice it.
say "Storage account: $STORAGE"
az storage account create \
    --name "$STORAGE" \
    --resource-group "$RG" \
    --location "$LOCATION" \
    --sku Standard_LRS \
    --kind StorageV2 \
    --min-tls-version TLS1_2 \
    --allow-blob-public-access false \
    --output none

STORAGE_KEY=$(az storage account keys list \
    --account-name "$STORAGE" --resource-group "$RG" \
    --query '[0].value' -o tsv)

az storage container create \
    --name "$CONTAINER_NAME" \
    --account-name "$STORAGE" \
    --account-key "$STORAGE_KEY" \
    --output none

# --- 2. Upload the weights under a version prefix --------------------------
# The version prefix is the point: v1/ is never overwritten. Retraining
# produces v2/, and rolling back is a single env-var change and a restart —
# no rebuild, no code redeploy.
upload() {
    local src="$1" blob="$2"
    if [[ ! -e "$src" ]]; then
        echo "  SKIP (not found): $src"
        return
    fi
    say "Uploading $(basename "$src") -> $MODEL_VERSION/$blob"
    az storage blob upload \
        --account-name "$STORAGE" --account-key "$STORAGE_KEY" \
        --container-name "$CONTAINER_NAME" \
        --name "$MODEL_VERSION/$blob" \
        --file "$src" \
        --overwrite false \
        --output none
    echo "  sha256: $(sha256sum "$src" | cut -d' ' -f1)"
}

upload "$SEG_MODEL" "seg/muharaf_seg_best.mlmodel"
upload "$OCR_MODEL" "ocr/arman_run1_91.mlmodel"

if [[ -d "$HATFORMER_DIR" ]]; then
    say "Packing HATFormer checkpoint directory"
    tar -czf /tmp/hatformer.tar.gz -C "$HATFORMER_DIR" .
    upload /tmp/hatformer.tar.gz "ocr/hatformer.tar.gz"
    rm -f /tmp/hatformer.tar.gz
fi

echo
echo "Copy the sha256 values above into models.manifest.json. With them, a"
echo "corrupt or swapped blob fails loudly at boot instead of quietly"
echo "producing wrong predictions."

# --- 3. Read-only SAS for the container ------------------------------------
# Container-scoped, read+list only, one year. The app never holds a storage
# account key — only this narrow, expiring token.
SAS_EXPIRY=$(date -u -d '+1 year' '+%Y-%m-%dT%H:%MZ' 2>/dev/null \
          || date -u -v+1y '+%Y-%m-%dT%H:%MZ')
SAS=$(az storage container generate-sas \
    --name "$CONTAINER_NAME" \
    --account-name "$STORAGE" --account-key "$STORAGE_KEY" \
    --permissions rl \
    --expiry "$SAS_EXPIRY" \
    --https-only \
    -o tsv)

MODEL_BASE_URL="https://${STORAGE}.blob.core.windows.net/${CONTAINER_NAME}"

# --- 4. Container Apps environment + app -----------------------------------
say "Container Apps environment: $ACA_ENV"
az extension add --name containerapp --upgrade --only-show-errors --yes >/dev/null 2>&1 || true
az provider register --namespace Microsoft.App --wait
az provider register --namespace Microsoft.OperationalInsights --wait

az containerapp env create \
    --name "$ACA_ENV" --resource-group "$RG" --location "$LOCATION" \
    --output none

# 1 vCPU / 2GiB with min-replicas 0.
#
# The free grant is 180,000 vCPU-seconds and 360,000 GiB-seconds per
# subscription per month. At this size that is 50 hours of *active* time a
# month, on both meters. Scaled to zero, idle time consumes neither. Demo
# traffic will not come close to 50 hours of actual request processing, so
# the running cost is genuinely zero — the student credit is the safety net,
# not the funding source.
say "Container app: $APP  (image: $IMAGE)"
az containerapp create \
    --name "$APP" \
    --resource-group "$RG" \
    --environment "$ACA_ENV" \
    --image "$IMAGE" \
    --target-port 8000 \
    --ingress external \
    --cpu 1.0 \
    --memory 2.0Gi \
    --min-replicas 0 \
    --max-replicas 2 \
    --secrets "model-sas=${SAS}" \
    --env-vars \
        "OCR_DEVICE=cpu" \
        "OCR_MODEL_DIR=/models" \
        "OCR_MODEL_BASE_URL=${MODEL_BASE_URL}" \
        "OCR_MODEL_VERSION=${MODEL_VERSION}" \
        "OCR_MODEL_SAS_TOKEN=secretref:model-sas" \
        "OCR_ENABLED_OCR_ENGINES=*" \
        "OCR_EAGER_LOAD_MODELS=true" \
        "OMP_NUM_THREADS=2" \
        "WEB_CONCURRENCY=1" \
    --output none

FQDN=$(az containerapp show --name "$APP" --resource-group "$RG" \
    --query 'properties.configuration.ingress.fqdn' -o tsv)

say "Done — nothing here bills under normal demo use"
cat <<SUMMARY

  Live URL      https://${FQDN}
  Readiness     https://${FQDN}/health/ready
  Build info    https://${FQDN}/version

  GitHub Actions secrets for automated deploys:
    AZURE_RG=$RG
    AZURE_CONTAINERAPP=$APP

  Before a demo or interview, warm it up (removes the ~90s cold start):
    az containerapp update -n $APP -g $RG --min-replicas 1
  And put it back afterwards, since idle replicas do bill:
    az containerapp update -n $APP -g $RG --min-replicas 0

  Teardown (stops absolutely all spend):
    az group delete --name $RG --yes --no-wait

SUMMARY
