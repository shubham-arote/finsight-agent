#!/usr/bin/env bash
# Tear down everything setup.sh + deploy.sh created, so an idle project bills nothing.
# Safe to re-run: every step tolerates already-deleted resources.
#
#   PROJECT_ID=my-proj ./destroy.sh
#
# NOT deleted (deliberately): the project itself, and any Qdrant Cloud cluster — that
# lives outside GCP. Redeploy afterwards with ./setup.sh && ./deploy.sh.
set -uo pipefail

: "${PROJECT_ID:?set PROJECT_ID}"
REGION="${REGION:-us-central1}"
BUCKET_NAME="${BUCKET_NAME:-${PROJECT_ID}-finsight-uploads}"
TOPIC_NAME="${TOPIC_NAME:-finsight-ingest-topic}"
SUB_NAME="${SUB_NAME:-finsight-ingest-sub}"
SA_NAME="${SA_NAME:-finsight-runner}"
SA_EMAIL="${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"
REPO="${REPO:-finsight}"

echo "Tearing down finsight in ${PROJECT_ID}…"

echo "── Cloud Run services (stops all serving cost)"
for SVC in finsight finsight-ingest; do
  gcloud run services delete "$SVC" --region="$REGION" --project="$PROJECT_ID" --quiet \
    2>/dev/null && echo "   deleted $SVC" || echo "   $SVC absent"
done

echo "── Pub/Sub subscription + topic"
gcloud pubsub subscriptions delete "$SUB_NAME" --project="$PROJECT_ID" --quiet 2>/dev/null \
  && echo "   deleted $SUB_NAME" || echo "   $SUB_NAME absent"
gcloud pubsub topics delete "$TOPIC_NAME" --project="$PROJECT_ID" --quiet 2>/dev/null \
  && echo "   deleted $TOPIC_NAME" || echo "   $TOPIC_NAME absent"

echo "── GCS bucket (notification goes with it)"
gcloud storage rm -r "gs://${BUCKET_NAME}" --project="$PROJECT_ID" 2>/dev/null \
  && echo "   deleted gs://${BUCKET_NAME}" || echo "   bucket absent"

echo "── Secret Manager (billed per secret per month)"
for S in QDRANT_URL QDRANT_API_KEY GROQ_API_KEY GEMINI_API_KEY COHERE_API_KEY \
         LANGFUSE_PUBLIC_KEY LANGFUSE_SECRET_KEY; do
  gcloud secrets delete "$S" --project="$PROJECT_ID" --quiet 2>/dev/null \
    && echo "   deleted $S" || true
done

echo "── Artifact Registry repo (image storage)"
gcloud artifacts repositories delete "$REPO" --location="$REGION" --project="$PROJECT_ID" \
  --quiet 2>/dev/null && echo "   deleted $REPO" || echo "   $REPO absent"

echo "── Service account + its role bindings"
for ROLE in roles/storage.objectViewer roles/secretmanager.secretAccessor roles/aiplatform.user; do
  gcloud projects remove-iam-policy-binding "$PROJECT_ID" \
    --member="serviceAccount:${SA_EMAIL}" --role="$ROLE" --quiet >/dev/null 2>&1 || true
done
gcloud iam service-accounts delete "$SA_EMAIL" --project="$PROJECT_ID" --quiet 2>/dev/null \
  && echo "   deleted $SA_EMAIL" || echo "   service account absent"

echo ""
echo "done — nothing left running."
echo "Remaining (free unless used): enabled APIs, Cloud Build history, this project."
echo "Redeploy with:  PROJECT_ID=${PROJECT_ID} ./setup.sh && PROJECT_ID=${PROJECT_ID} ./deploy.sh"
