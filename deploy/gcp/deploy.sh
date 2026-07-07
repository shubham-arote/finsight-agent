#!/usr/bin/env bash
# Build once, deploy twice (week-3 + week-4 reference patterns):
#   1. Cloud Build -> one lean image in Artifact Registry
#   2. `finsight`        Cloud Run multi-container service (agent + MCP sidecar)
#   3. `finsight-ingest` Cloud Run worker + Pub/Sub push subscription (async ingestion)
# Usage:  PROJECT_ID=my-proj ./deploy.sh          (after ./setup.sh)
set -euo pipefail

: "${PROJECT_ID:?set PROJECT_ID}"
export REGION="${REGION:-us-central1}"
export TOPIC_NAME="${TOPIC_NAME:-finsight-ingest-topic}"
export SA_EMAIL="${SA_EMAIL:-finsight-runner@${PROJECT_ID}.iam.gserviceaccount.com}"
export REPO="${REPO:-finsight}"
export IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/finsight:latest"

gcloud config set project "$PROJECT_ID"
cd "$(dirname "$0")/../.."               # repo root

echo "── 1. Build the image (Cloud Build)"
gcloud builds submit --tag "$IMAGE" .

echo "── 2. Deploy the app (agent + MCP sidecar, scale-to-zero)"
envsubst '$PROJECT_ID $IMAGE $SA_EMAIL' < deploy/gcp/service-app.yaml \
  | gcloud run services replace - --region="$REGION"
gcloud run services add-iam-policy-binding finsight --region="$REGION" \
  --member="allUsers" --role="roles/run.invoker" --quiet   # public demo URL

echo "── 3. Deploy the ingest worker (private — only Pub/Sub may push)"
gcloud run deploy finsight-ingest --image="$IMAGE" --region="$REGION" \
  --service-account="$SA_EMAIL" --no-allow-unauthenticated \
  --memory=1Gi --max-instances=2 \
  --command=uvicorn --args=finsight.ingest_worker:app,--host,0.0.0.0,--port,8080 \
  --port=8080 \
  --set-secrets="QDRANT_URL=QDRANT_URL:latest,QDRANT_API_KEY=QDRANT_API_KEY:latest,GROQ_API_KEY=GROQ_API_KEY:latest,GEMINI_API_KEY=GEMINI_API_KEY:latest,COHERE_API_KEY=COHERE_API_KEY:latest" \
  --set-env-vars="GOOGLE_CLOUD_PROJECT=${PROJECT_ID},QDRANT_COLLECTION=finsight_chunks"

echo "── 4. Pub/Sub push subscription -> worker (OIDC-authenticated)"
WORKER_URL=$(gcloud run services describe finsight-ingest --region="$REGION" --format='value(status.url)')
gcloud run services add-iam-policy-binding finsight-ingest --region="$REGION" \
  --member="serviceAccount:$SA_EMAIL" --role="roles/run.invoker" --quiet
gcloud pubsub subscriptions create finsight-ingest-sub \
  --topic="$TOPIC_NAME" --push-endpoint="$WORKER_URL" \
  --push-auth-service-account="$SA_EMAIL" \
  --message-retention-duration=1h 2>/dev/null || echo "subscription exists"

APP_URL=$(gcloud run services describe finsight --region="$REGION" --format='value(status.url)')
echo ""
echo "done."
echo "  app:     $APP_URL          (verify: uv run python scripts/verify_stack.py $APP_URL)"
echo "  ingest:  gsutil cp report.pdf gs://\$BUCKET_NAME/   -> auto-indexed via Pub/Sub"
