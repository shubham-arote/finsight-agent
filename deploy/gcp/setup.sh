#!/usr/bin/env bash
# One-time GCP project setup for finsight (week-3 reference pattern):
# APIs, Artifact Registry, GCS upload bucket -> Pub/Sub topic, secrets, service account.
# Usage:  PROJECT_ID=my-proj ./setup.sh          (run in Cloud Shell or with gcloud auth)
set -euo pipefail

: "${PROJECT_ID:?set PROJECT_ID}"
export REGION="${REGION:-us-central1}"
export BUCKET_NAME="${BUCKET_NAME:-${PROJECT_ID}-finsight-uploads}"
export TOPIC_NAME="${TOPIC_NAME:-finsight-ingest-topic}"
export SA_NAME="${SA_NAME:-finsight-runner}"
export SA_EMAIL="${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"
export REPO="${REPO:-finsight}"

gcloud config set project "$PROJECT_ID"

echo "── 1. APIs"
gcloud services enable run.googleapis.com cloudbuild.googleapis.com \
  artifactregistry.googleapis.com pubsub.googleapis.com storage.googleapis.com \
  secretmanager.googleapis.com aiplatform.googleapis.com

echo "── 2. Artifact Registry"
gcloud artifacts repositories create "$REPO" --repository-format=docker \
  --location="$REGION" 2>/dev/null || echo "repo exists"

echo "── 3. Upload bucket + OBJECT_FINALIZE -> Pub/Sub (the async ingestion trigger)"
gcloud storage buckets create "gs://$BUCKET_NAME" --location="$REGION" 2>/dev/null || echo "bucket exists"
gcloud pubsub topics create "$TOPIC_NAME" 2>/dev/null || echo "topic exists"
# let the GCS service agent publish to the topic (week-3 step)
PN=$(gcloud projects describe "$PROJECT_ID" --format="value(projectNumber)")
gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member="serviceAccount:service-${PN}@gs-project-accounts.iam.gserviceaccount.com" \
  --role="roles/pubsub.publisher" --quiet
gcloud storage buckets notifications create "gs://$BUCKET_NAME" \
  --topic="$TOPIC_NAME" --event-types=OBJECT_FINALIZE 2>/dev/null || echo "notification exists"

echo "── 4. Secrets (create empty if unset; fill via console or rerun with values)"
for S in QDRANT_URL QDRANT_API_KEY GROQ_API_KEY GEMINI_API_KEY COHERE_API_KEY; do
  VAL="${!S:-}"
  if gcloud secrets describe "$S" >/dev/null 2>&1; then echo "secret $S exists"; else
    echo -n "${VAL:-CHANGE_ME}" | gcloud secrets create "$S" --data-file=- --replication-policy=automatic
  fi
done

echo "── 5. Service account (least privilege, week-3 pattern)"
gcloud iam service-accounts create "$SA_NAME" --display-name="finsight runner" 2>/dev/null || echo "sa exists"
for ROLE in roles/storage.objectViewer roles/secretmanager.secretAccessor roles/aiplatform.user; do
  gcloud projects add-iam-policy-binding "$PROJECT_ID" \
    --member="serviceAccount:$SA_EMAIL" --role="$ROLE" --quiet
done
# Pub/Sub OIDC push needs token creation on the SA
gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member="serviceAccount:service-${PN}@gcp-sa-pubsub.iam.gserviceaccount.com" \
  --role="roles/iam.serviceAccountTokenCreator" --quiet

echo "done — next: ./deploy.sh"
