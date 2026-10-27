# Data for every lab

The labs share one frozen corpus and one question set, so every change you make is measured on the same
questions. Do not edit these files; each lab checks their SHA-256 hashes before it uses them.

Most of it is a sample of SQuAD 1.1, the Stanford Question Answering Dataset ([Rajpurkar et al., 2016](https://arxiv.org/abs/1606.05250)): a well-known public benchmark of more than 100,000 questions that people wrote about Wikipedia paragraphs, each with the answer's span in the paragraph. The labs use 120 of its paragraphs and 30 of its development-set questions, unchanged, so that every run is cheap and every result can be checked by hand. A score on this sample is not a SQuAD benchmark score, and cannot be compared with published results.

| File | What it holds |
|---|---|
| `squad-passages.json` | 120 Wikipedia paragraphs: the first three paragraphs of the first 40 articles in the SQuAD 1.1 development set, text unchanged. Each has a `source_id` such as `S03-P00` (article 3, paragraph 0) |
| `squad-chunks.json` | The same 120 paragraphs split into 272 chunks. Each chunk keeps its `source_id`, its own `chunk_id` (`S03-P00-C00`) and the `start` and `end` character offsets of its text inside the paragraph (start included, end excluded) |
| `development-questions.json` | 30 questions from articles 0 to 4, six per article, each with its SQuAD answers and `known_support`: the paragraph the question was written from |
| `support-case.json` | Five short product policies, H01 to H05, for two made-up products, Atlas and Boreal, written for these labs. Each has a version and an access level |
| `dataset-manifest.json` | Where the SQuAD data came from, how the sample was drawn, its licence and the hashes |
| `SHA256SUMS.txt` | The SHA-256 hash of each file above, which Lab 00 checks |

## How these files relate to SQuAD's own file

SQuAD's own file nests everything by article. Each article holds its paragraphs, and each paragraph (`context`)
holds the questions written about it (`qas`), with the answer's position in the paragraph:

```json
{"version": "v1.1",
 "data": [{"title": "Super_Bowl_50",
           "paragraphs": [{"context": "Super Bowl 50 was an American football game ...",
                           "qas": [{"id": "56be4db0acb8001400a502ec",
                                    "question": "Which NFL team represented the AFC at Super Bowl 50?",
                                    "answers": [{"answer_start": 177, "text": "Denver Broncos"}]}]}]}]}
```

That layout suits reading comprehension, where a model is handed the paragraph with the question. A RAG system
has to find the paragraph itself, so these labs split the file in two and add one file of their own:

| SQuAD's own file | Here | Why |
|---|---|---|
| each `context` | `squad-passages.json`, one record per paragraph | What the search looks through: text only, with a short `source_id` (`S00-P00` is article 0, paragraph 0) that a citation can name, and `article_index` and `paragraph_index`, its place in SQuAD's file |
| each `qas` entry | `development-questions.json`, one record per question | Kept apart from the paragraphs, so the search cannot find a question's paragraph by matching the question itself. Each keeps SQuAD's `id` (as `question_id`) and `answers`; `known_support` writes down the paragraph it was nested under, which the split would otherwise lose |
| — | `squad-chunks.json` | Made for these labs: each paragraph cut into windows of 96 tokens, a new window every 80 tokens, counted with another model's tokenizer. Every chunk keeps its paragraph's `source_id` and the character offsets of its text, so it slices back to the paragraph exactly. Lab 03 shows how and compares other ways to cut |

The text of every paragraph, question and answer is SQuAD's, unchanged; an `answer_start` still points into the
paragraph's text.

Two limits to keep in mind when you read a score:

- `known_support` names the paragraph each question was written from. Another paragraph may also answer it, so
  these labels are not a complete list of relevant chunks.
- The 30 questions come from only five articles, so one article's quirks repeat six times.

A second set of 30 questions from other articles is held back and never scored in the labs: it
is the set you would score once, after every choice is frozen.

**Licence.** The SQuAD passages and questions are from Rajpurkar et al., SQuAD 1.1, and Wikipedia contributors,
under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). Source: the
[official SQuAD dataset](https://rajpurkar.github.io/SQuAD-explorer/). The text is unchanged; the IDs and chunk
records are additions. The Atlas and Boreal policies are original material written for these labs.
