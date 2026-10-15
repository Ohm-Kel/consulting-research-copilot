# Deploying to Google Cloud Run

The copilot runs as one container on [Cloud Run](https://cloud.google.com/run): the same
Dockerfile that CI builds and smoke-tests, deployed by a GitHub Actions workflow. It scales to
zero when nobody is using it, and a demo's traffic fits in Cloud Run's free tier.

```
git tag v1.6.0 ──► Deploy workflow ──► sign in to Google with no stored key (Workload Identity Federation)
                                   ──► docker build (the Dockerfile CI tests) ──► Artifact Registry
                                   ──► Cloud Run: 2 vCPU, 2 GiB, 0 to 1 instance ──► smoke test /health and /search
                       OpenAI key: Secret Manager ──► OPENAI_API_KEY inside the container
```

## One-time setup (about 15 minutes)

You need a Google account and a payment card. The card is only for the billing account; a demo
stays within the free tier (see [What it costs](#what-it-costs)).

1. **Create a project with billing.** In the [Google Cloud console](https://console.cloud.google.com),
   create a project (for example `consulting-copilot`) and link a billing account to it. Note the
   project ID.
2. **Set a budget alert.** Billing → Budgets & alerts → Create budget: $5 a month, with email
   alerts at 50%, 90% and 100%. An alert warns you; it does not stop spending, so it is a safety
   net on top of the daily question limit.
3. **Run the setup script in Cloud Shell.** Open [Cloud Shell](https://shell.cloud.google.com)
   (a terminal in the browser with `gcloud` installed), then:

   ```bash
   git clone https://github.com/Ohm-Kel/consulting-research-copilot.git
   cd consulting-research-copilot
   PROJECT_ID=your-project-id bash scripts/gcp_setup.sh
   ```

   It enables the APIs, creates the image repository and two service accounts, stores your
   OpenAI key in Secret Manager (it asks for the key and does not display it), and lets only this
   GitHub repository deploy. At the end it prints four values.
4. **Give GitHub the four values.** In the repository on GitHub: Settings → Secrets and
   variables → Actions → **Variables** tab → New repository variable, once for each of
   `GCP_PROJECT_ID`, `GCP_REGION`, `GCP_WIF_PROVIDER` and `GCP_DEPLOYER_SA`. They are
   identifiers, not secrets. Optional: `COPILOT_QUERY_DAILY_LIMIT` (default 50).
5. **Deploy.** Actions → Deploy → Run workflow on `main`. The first run takes about 15 minutes,
   most of it building the image. The run summary shows the public URL.

After that, every release tag (`git push origin v1.6.0`) deploys automatically.

## What it costs

Free-tier allowances, as listed by Google on 2026-10-15:

| Service | Free each month | This app |
|---|---|---|
| Cloud Run | 2 million requests, 180,000 vCPU-seconds, 360,000 GB-seconds | An agent question uses about 20 s × 2 vCPU = 40 vCPU-seconds, so roughly 4,000 questions a month fit |
| Artifact Registry | 0.5 GB of storage | The image is larger, so storage costs a few cents a month; old images are deleted automatically (the 3 newest are kept) |
| Secret Manager | 6 active secret versions, 10,000 accesses | One secret, read when an instance starts |

The OpenAI bill is separate. With `COPILOT_QUERY_DAILY_LIMIT=50` the agent spends at most about
$0.07 a day on `gpt-5.6-luna`; `/search` never calls OpenAI.

**Cold starts.** When idle, the service scales to zero. The next visitor waits about a minute
while the models load, then answers come at normal speed.

## The settings, and why

| Setting | Value | Why |
|---|---|---|
| CPU, memory | 2 vCPU, 2 GiB | The API peaks under 1 GB; two CPUs keep reranking near laptop speed |
| Instances | 0 to 1 | Scales to zero when idle; one instance keeps the in-memory rate limits and the daily limit exact |
| Concurrency | 4 | Reranking is CPU-bound, so a few requests at a time is the useful maximum |
| Startup CPU boost | on | Shortens cold starts, when the models load |
| `COPILOT_QUERY_DAILY_LIMIT` | 50 | Caps OpenAI spend; `/search` keeps working after the limit |
| `FORWARDED_ALLOW_IPS` | `*` | Cloud Run sits behind Google's proxy; this lets the per-client rate limits see each visitor's address |
| Access | public (`--allow-unauthenticated`) | It is a demo; set `COPILOT_API_KEYS` as well to require keys |

## Running it

- **Logs:** Cloud Run → `consulting-research-copilot` → Logs. Each question writes one JSON line
  with the model, tool calls, tokens, time and cost.
- **Change the daily limit:** set the `COPILOT_QUERY_DAILY_LIMIT` repository variable, then
  re-run the Deploy workflow.
- **Replace the OpenAI key** in Cloud Shell, without the key landing in your shell history:

  ```bash
  read -rsp "New OpenAI key: " KEY && printf '%s' "$KEY" | gcloud secrets versions add openai-api-key --data-file=- && unset KEY
  ```

  New instances use the newest version; re-run Deploy to switch immediately.
- **Roll back:** Cloud Run → Revisions → send 100% of traffic to an earlier revision, or re-run
  the Deploy workflow on an earlier tag.
- **Take it down:** `gcloud run services delete consulting-research-copilot --region us-central1`.

## Security

- GitHub holds no Google key. The workflow gets a short-lived token through Workload Identity
  Federation, and Google accepts it only from `Ohm-Kel/consulting-research-copilot`.
- `github-deployer` can deploy Cloud Run services, push images to the one repository and run the
  service as `copilot-runtime`. `copilot-runtime` can read the OpenAI secret and nothing else.
- Per-client rate limits read the client address from the `X-Forwarded-For` header, which a
  determined client can fake. The daily limit counts every `/query` request regardless, so it is
  the real ceiling on spending.

Why Cloud Run rather than other hosts: [ADR 0013](adr/0013-host-on-cloud-run.md).
