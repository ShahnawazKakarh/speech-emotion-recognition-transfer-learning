# Writeups

Long-form artifacts for this project, all sharing the same numbers from
[`results/results.md`](../../results/results.md):

| File | Audience | Length | Format |
|---|---|---|---|
| [`blog-post.md`](./blog-post.md) | General + practitioner | ~1 700 words | SEO-optimized Markdown (front-matter + JSON-LD-ready FAQ schema) |
| [`paper.tex`](./paper.tex) | Research community | 4-page short paper | LaTeX (venue-agnostic; drop into IEEEtran or Interspeech template) |
| [`linkedin-post.md`](./linkedin-post.md) | Professional network | 150 / 200 / 280 words (3 variants) | LinkedIn-ready text |

## Headline finding

**Multimodal fusion isn't universally better.** Same architecture, same pipeline, two datasets:

- **RAVDESS** (clean lab speech, 2 fixed sentences) → multimodal wins by **+6.9 pp WF1**
- **MELD** (noisy *Friends* dialogue) → text-only wins by **+1.9 pp WF1**

The determining factor is modality complementarity: fusion helps when both modalities carry independent, non-noisy signal.

## How to use

- **Blog post** — render with any Markdown engine that supports YAML front-matter. The `faq`, `schema`, and `keywords` sections are ready for JSON-LD generation and OpenGraph cards.
- **Paper** — copy `paper.tex` into a venue-specific template (`IEEEtran.cls`, `interspeech2025.sty`, etc.). Bibliography is already in `\thebibliography` form for easy conversion to BibTeX if needed.
- **LinkedIn** — three pre-written variants (long / short / comment-bait). Pick one, paste directly. Posting strategy notes included at the bottom of the file.

## Updating

If you re-run experiments and the numbers change, update [`results/results.md`](../../results/results.md) first — that's the canonical source. Then sweep the three writeups for the same numbers (search for `0.728`, `0.609`, `0.590`, `0.357`, `+6.9`, `-1.9`, etc.).
