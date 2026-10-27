# 0013. A recorded demo instead of a hosted one

- Status: Accepted
- Date: 2026-10-27

## Context

A public demo helps a reader see the copilot work without installing it. The engineering
standards ask that any hosting builds from the same Dockerfile CI tests. The container needs
about 1 GB of memory (measured: 758 MB in use, 979 MB peak), and this is a portfolio project,
not a production service, so its owner does not want to pay for hosting or attach a payment card.

## Options (checked 2026-10-13 to 2026-10-27)

- **Hugging Face Docker Spaces:** build the Dockerfile directly, but need the PRO plan
  ($9 a month).
- **Google Cloud Run:** a free tier that covers a demo, but it needs a billing account with a
  card. A complete setup (deploy workflow, one-time setup script, guide) was built and checked
  against Google's documentation, but never run against a real project. It is kept on the
  branch `feat/cloud-run-deploy` (PR #13, closed).
- **Azure for Students:** no card, a $100 credit for 12 months, for full-time students who
  sign up with a school email. It would need a new deploy setup and a yearly renewal.
- **A recording:** a GIF of the demo page answering a real question, plus local setup in five
  commands.

## Decision

Show the demo as a recording in the README (`docs/images/demo.gif`): the real demo page,
served locally, answering a question about Nike's gross margin with page citations. Keep the
Docker image and its CI smoke test, so the service stays ready to host.

## Consequences

- No running costs, no account to maintain, and no public endpoint spending the OpenAI credit.
- Readers cannot click a live link. They see the recording, and they can run the same page
  locally (`uv run uvicorn copilot.api:app`, then http://localhost:8000/).
- The recording shows one question. `docs/examples.md` holds more real answers, and the
  evaluation results cover 65 questions.
- If hosting is wanted later, reopening PR #13 needs only the billing account; Azure for
  Students is the no-card alternative.
