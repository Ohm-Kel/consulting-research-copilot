# 0009. 10-K print editions for two reports

- Status: Accepted
- Date: 2026-09-24

## Context

The designed annual reports for Lululemon and Under Armour embed fonts without a text
mapping, so `pypdf` extracted gibberish from them.

## Decision

Use the plain 10-K print editions for Lululemon (FY2024) and Under Armour (FY2025). Download
URLs and SHA-256 checksums for all five reports live in `scripts/download_data.py`, and
`fonttools` is a dependency so that `pypdf` can decode the embedded fonts in the other reports.

## Consequences

- Text extraction is reliable for all five reports, and a changed or corrupted download fails
  its checksum.
- The 10-K editions leave out some of the charts and narrative pages of the designed reports.
