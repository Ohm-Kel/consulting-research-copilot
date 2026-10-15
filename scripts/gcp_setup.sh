#!/usr/bin/env bash
# One-time Google Cloud setup so GitHub Actions can deploy the copilot to Cloud Run.
# Run it in Cloud Shell (https://shell.cloud.google.com) after creating a project with billing:
#
#   PROJECT_ID=your-project-id bash scripts/gcp_setup.sh
#
# It is safe to run again: every step skips what already exists. Nothing here stores a
# long-lived key: GitHub signs in through Workload Identity Federation, and the OpenAI key
# goes straight into Secret Manager. Full walkthrough: docs/deployment.md
set -euo pipefail

: "${PROJECT_ID:?Set PROJECT_ID, for example: PROJECT_ID=my-project bash scripts/gcp_setup.sh}"
REGION="${REGION:-us-central1}"
GITHUB_REPO="${GITHUB_REPO:-Ohm-Kel/consulting-research-copilot}"
AR_REPO="copilot"
RUNTIME_SA="copilot-runtime@${PROJECT_ID}.iam.gserviceaccount.com"
DEPLOYER_SA="github-deployer@${PROJECT_ID}.iam.gserviceaccount.com"
POOL="github"
PROVIDER="consulting-copilot"

gcloud config set project "$PROJECT_ID" --quiet
PROJECT_NUMBER="$(gcloud projects describe "$PROJECT_ID" --format='value(projectNumber)')"

echo "1/7 Enabling the APIs (Cloud Run, Artifact Registry, Secret Manager, IAM credentials, STS)"
gcloud services enable run.googleapis.com artifactregistry.googleapis.com secretmanager.googleapis.com \
  iamcredentials.googleapis.com sts.googleapis.com

echo "2/7 Docker repository '${AR_REPO}' in ${REGION}, keeping the 3 newest images"
if ! gcloud artifacts repositories describe "$AR_REPO" --location="$REGION" >/dev/null 2>&1; then
  gcloud artifacts repositories create "$AR_REPO" --repository-format=docker --location="$REGION" \
    --description="Consulting Research Copilot images"
fi
POLICY_FILE="$(mktemp)"
cat >"$POLICY_FILE" <<'JSON'
[
  {"name": "keep-3-newest", "action": {"type": "Keep"}, "mostRecentVersions": {"keepCount": 3}},
  {"name": "delete-older", "action": {"type": "Delete"}, "condition": {"tagState": "any", "olderThan": "1d"}}
]
JSON
gcloud artifacts repositories set-cleanup-policies "$AR_REPO" --location="$REGION" --policy="$POLICY_FILE" \
  --no-dry-run --quiet >/dev/null
rm -f "$POLICY_FILE"

echo "3/7 Service accounts: copilot-runtime (runs the app) and github-deployer (used by GitHub Actions)"
created=0
for name in copilot-runtime github-deployer; do
  if ! gcloud iam service-accounts describe "${name}@${PROJECT_ID}.iam.gserviceaccount.com" >/dev/null 2>&1; then
    gcloud iam service-accounts create "$name" --display-name="$name"
    created=1
  fi
done
if [ "$created" -eq 1 ]; then
  sleep 15 # new service accounts take a few seconds to reach IAM; granting roles too early fails
fi

echo "4/7 OpenAI API key in Secret Manager (readable by copilot-runtime only)"
if ! gcloud secrets describe openai-api-key >/dev/null 2>&1; then
  read -rsp "Paste your OpenAI API key (it is not shown): " OPENAI_KEY
  echo
  printf '%s' "$OPENAI_KEY" | gcloud secrets create openai-api-key --replication-policy=automatic --data-file=-
  unset OPENAI_KEY
fi
gcloud secrets add-iam-policy-binding openai-api-key --member="serviceAccount:${RUNTIME_SA}" \
  --role=roles/secretmanager.secretAccessor --quiet >/dev/null

echo "5/7 Deployer permissions: deploy Cloud Run, push images, run the service as copilot-runtime"
gcloud projects add-iam-policy-binding "$PROJECT_ID" --member="serviceAccount:${DEPLOYER_SA}" \
  --role=roles/run.admin --condition=None --quiet >/dev/null
gcloud artifacts repositories add-iam-policy-binding "$AR_REPO" --location="$REGION" \
  --member="serviceAccount:${DEPLOYER_SA}" --role=roles/artifactregistry.writer --quiet >/dev/null
gcloud iam service-accounts add-iam-policy-binding "$RUNTIME_SA" --member="serviceAccount:${DEPLOYER_SA}" \
  --role=roles/iam.serviceAccountUser --quiet >/dev/null

echo "6/7 Workload Identity Federation: only ${GITHUB_REPO} may act as github-deployer"
if ! gcloud iam workload-identity-pools describe "$POOL" --location=global >/dev/null 2>&1; then
  gcloud iam workload-identity-pools create "$POOL" --location=global --display-name="GitHub Actions"
fi
if ! gcloud iam workload-identity-pools providers describe "$PROVIDER" --location=global \
  --workload-identity-pool="$POOL" >/dev/null 2>&1; then
  gcloud iam workload-identity-pools providers create-oidc "$PROVIDER" --location=global \
    --workload-identity-pool="$POOL" --display-name="consulting-research-copilot" \
    --issuer-uri="https://token.actions.githubusercontent.com" \
    --attribute-mapping="google.subject=assertion.sub,attribute.repository=assertion.repository" \
    --attribute-condition="assertion.repository == '${GITHUB_REPO}'"
fi
gcloud iam service-accounts add-iam-policy-binding "$DEPLOYER_SA" --role=roles/iam.workloadIdentityUser \
  --member="principalSet://iam.googleapis.com/projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/${POOL}/attribute.repository/${GITHUB_REPO}" \
  --quiet >/dev/null

echo "7/7 Done. Add these as GitHub repository variables (Settings > Secrets and variables > Actions > Variables):"
echo
echo "  GCP_PROJECT_ID   = ${PROJECT_ID}"
echo "  GCP_REGION       = ${REGION}"
echo "  GCP_WIF_PROVIDER = projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/${POOL}/providers/${PROVIDER}"
echo "  GCP_DEPLOYER_SA  = ${DEPLOYER_SA}"
echo
echo "None of these are secrets. Then run the Deploy workflow from the Actions tab."
