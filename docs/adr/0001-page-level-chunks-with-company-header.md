# 0001. Page-level chunks with a company header

- Status: Accepted
- Date: 2026-09-24

## Context

Every answer must cite the report page that supports it. Annual reports mix narrative text
with financial tables, and a bare table row ("Gross margin 42.7% 44.6%") does not say which
company it belongs to. Five companies report similar metrics, so a retriever that loses the
company name can return the wrong report's numbers.

## Decision

- Split each PDF page on its own into chunks of 300 words with a 50-word overlap. A chunk
  never crosses a page boundary, so it has exactly one page number to cite.
- Embed each chunk with a short header, for example "Nike annual report, page 38.", so a table
  still carries its company and page.
- Cite PDF page numbers, not the numbers printed on the page.

## Consequences

- Citations are unambiguous and easy to check against the PDF.
- A passage that runs over a page break is split in two, and either half may miss the full
  answer.
- PDF page numbers can differ from printed ones; the README says so.
- Tables are flattened to text, which still hurts some numeric questions (see the roadmap).
