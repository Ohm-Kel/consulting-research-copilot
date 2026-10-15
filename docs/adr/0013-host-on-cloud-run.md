# 0013. Host the demo on Google Cloud Run

- Status: Accepted
- Date: 2026-10-15

## Context

The project needs a public demo, built from the same Dockerfile that CI tests. The container
holds torch, two models and the index; the API peaks under 1 GB of memory, so a host needs at
least 2 GB to be safe. Neither maintainer's laptop can run Docker, so images must be built in CI.
The OpenAI credit is small and prepaid, so a public endpoint must not be able to spend it freely.

## Options (checked 2026-10-13 to 2026-10-15)

- **Hugging Face Spaces:** builds a Dockerfile directly, but Docker Spaces now need the PRO plan
  ($9 a month) before the free CPU hardware (2 vCPU, 16 GB) can be used.
- **Google Cloud Run:** runs any container, scales to zero, and has a monthly free tier (2 million
  requests, 180,000 vCPU-seconds, 360,000 GB-seconds). It needs a billing account with a card.

## Decision

Deploy to Cloud Run from GitHub Actions on release tags or on demand:

- GitHub signs in through Workload Identity Federation, restricted to this repository, by
  impersonating a deployer service account. No long-lived key is stored anywhere.
- The image is built on the GitHub runner, pushed to Artifact Registry (only the 3 newest images
  are kept), and run as a service account that can read only the OpenAI secret.
- 2 vCPU, 2 GiB, 0 to 1 instance, startup CPU boost, and a daily limit of 50 agent questions.

## Consequences

- A demo's traffic costs nothing on Cloud Run. Artifact Registry storage costs a few cents a
  month, and OpenAI spending is capped at about $0.07 a day.
- Cold starts take about a minute after the service has been idle.
- With at most one instance, the in-memory rate limits and the daily limit are exact; more
  instances would need shared state (for example Redis) first.
- The one-time setup (`scripts/gcp_setup.sh`) needs the project owner to run it in Cloud Shell;
  after that, deploying is a tag push.
