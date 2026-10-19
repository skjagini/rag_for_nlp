# Data for every lab

The labs share one frozen corpus and one question set, so every change you make is measured on the same
questions. Do not edit these files; each lab checks their SHA-256 hashes before it uses them.

| File | What it holds |
|---|---|
| `squad-passages.json` | 120 Wikipedia paragraphs: the first three paragraphs of the first 40 articles in the SQuAD 1.1 development set, text unchanged. Each has a `source_id` such as `S03-P00` (article 3, paragraph 0) |
| `squad-chunks.json` | The same 120 paragraphs split into 272 chunks. Each chunk keeps its `source_id`, its own `chunk_id` (`S03-P00-C00`) and the `start` and `end` character offsets of its text inside the paragraph (start included, end excluded) |
| `development-questions.json` | 30 questions from articles 0 to 4, six per article, each with its SQuAD answers and `known_support`: the paragraph the question was written from |
| `support-case.json` | Five short product policies, H01 to H05, for two made-up products, Atlas and Boreal, written for these labs. Each has a version and an access level |
| `dataset-manifest.json` | Where the SQuAD data came from, how the sample was drawn, its licence and the hashes |
| `SHA256SUMS.txt` | The SHA-256 hash of each file above, which Lab 00 checks |

Two limits to keep in mind when you read a score:

- `known_support` names the paragraph each question was written from. Another paragraph may also answer it, so
  these labels are not a complete list of relevant chunks.
- The 30 questions come from only five articles, so one article's quirks repeat six times.

A second set of 30 questions from other articles is held back by the instructor and never scored in the labs: it
is the set you would score once, after every choice is frozen.

**Licence.** The SQuAD passages and questions are from Rajpurkar et al., SQuAD 1.1, and Wikipedia contributors,
under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). Source: the
[official SQuAD dataset](https://rajpurkar.github.io/SQuAD-explorer/). The text is unchanged; the IDs and chunk
records are additions. The Atlas and Boreal policies are original material written for this course.
