# Final challenge: How to rescue a failing RAG assistant

**CPU | API key: `OPENAI_API_KEY` (chat and embedding calls) | Self-study, after Day 2**

## Goal

Take one question the assistant gets wrong, find the stage where it fails, change that stage, and show with a
before-and-after on the same questions what the change fixed and what it broke. There is no notebook and no new code
in `shared/`: you use the pieces Labs 04, 06 and 08 built, and write a short report.

You can already trace a question (Lab 04), score a golden set (Lab 06) and compare two versions question by question
(Lab 08). Each lab did one of those on its own. This guide puts them in the order you would follow when a real
assistant gets a question wrong.

## What you will learn

- How to choose a failure worth fixing from saved evaluation results.
- How to name the failing stage (corpus, retrieval, packing or generation) from a trace, not from the answer alone.
- How to change one stage and keep everything else the same.
- How to compare the two versions on all 38 golden records, with the regressions listed.
- How to write a short report a colleague could check.

## Before you start

You need Labs 04, 06 and 08's `shared/` modules (`shared/trace.py`, `shared/evals.py`, `shared/hybrid.py`) and the
environment you used for them. You do not need the capstone, Gradio or the `capstone` dependency group.

Work in a new notebook inside this folder. Copy in the **Setup** and **Config** cells from any lab (Lab 06's are the
same as every other lab's): they find the repository folder and your key, and give you `settings` and `ROOT`.

⚠️ Each scoring of the 38 golden records makes 38 chat calls, and every step below scores at least twice. On the
shipped model that is a few US cents; check the prices in `models.env` if you use another model.

## Step 1: Score the golden set and pick one failure

Rebuild Lab 06's baseline: the 277 records in a vector store, and one row per golden record.

```python
from shared import evals, lc_pipeline
from shared.trace import corpus

golden = evals.golden_set()
store = lc_pipeline.build_store(lc_pipeline.to_documents(corpus()), settings)
before = [evals.score(record, store, settings) for record in golden]
print(f"passed {sum(row['passed'] for row in before)} of {len(before)}")
for row in before:
    if not row["passed"]:
        print(row["question_id"], row["type"], row["retrieved"], "|", row["answer"])
```

In Lab 06's saved run the pipeline passed 34 of 38, and three policy cases failed in all three runs (0 of 3 each).
Pick one of these, or another failure your run shows:

| Record | Question | What Lab 06's run shows |
|---|---|---|
| G08 | *Who can export all reports in Boreal?* | No Boreal export policy exists. The pipeline retrieved Atlas's H03 first and answered *Only workspace owners can export all reports in Boreal.*, the Atlas rule with the product changed |
| G09 | *Do I need the mobile app to reset my Atlas password?* | Boreal's H04 ranked first, Atlas's H01 second, and the answer gave the Boreal rule: *Yes, password reset uses a mobile app.* |
| G05 | *Can temporary Atlas accounts use the usual email route?* | H01 ranked first and holds the exception, but the answer said *No* without the route that works (*a support agent*) |

A third kind of failure lives in Lab 09: after a question about Boreal, the follow-up *How long is the link valid?*
searched as typed finds the Atlas policy. It is not a golden record, so if you choose it, compare the two versions on
Lab 09's seven conversations (`09_Follow_Up_Questions/resources/conversations.json`) instead of the 38 records.

Your run may differ from Lab 06's: answers move between runs. Score the failing record two more times before you
choose it; a failure that comes and goes is a poor target for a before-and-after.

## Step 2: Trace the failing record

Follow the record through the pipeline Lab 06 scores (`lc_pipeline`, k=3, no token budget), one stage per print:

```python
record = next(r for r in golden if r["question_id"] == "G08")    # your record
hits = lc_pipeline.retrieve(store, record["question"])
for document, score in hits:
    print(f"{score:.3f}  {document.id}  {document.page_content[:70]}")
lines = lc_pipeline.evidence(hits)
reply = lc_pipeline.answer(record["question"], lines, settings)
print(reply.answer)
for claim in reply.claims:
    print(" ", claim.source_id, "|", claim.quote)
```

Lab 04's `trace()` records the same stages for Day 1's plain pipeline (k=5 and a 400-token pack), with the prompt hash
and the tokens billed. Run it too when you suspect packing: Lab 05 showed that the two pipelines give the answer step
different evidence.

```python
from shared.search import chunk_vectors
from shared.trace import trace

chunks = corpus()
record_trace = trace(record["question"], chunks, chunk_vectors(chunks, settings), settings)
print(record_trace["hits"], record_trace["packed"], record_trace["reply"], sep="\n")
```

## Step 3: Name the stage

Use Lab 04's table. Follow the support (the record's `known_support`) through the pipeline; the first step it does
not reach names the stage.

| The support is ... | Stage | What you could change |
|---|---|---|
| not in the corpus | corpus | the documents, or the rule that should make the model refuse |
| in the corpus, not in the hits | retrieval | `k`, hybrid search (Lab 08), a rewritten query (Lab 09), a product filter |
| in the hits, not in the pack | packing | the budget, the order |
| in the pack, and the answer is wrong | generation | a rule line, what else is packed beside the support |

Some records have no support on purpose. G08 should be refused, because no Boreal export policy exists: ask what the
model was given that made it answer, and which stage could have stopped it.

If you want names for failures beyond these four stages, the
[RAG Failure Diagnostics Clinic](https://github.com/Shubhamsaboo/awesome-llm-apps/tree/main/rag_tutorials/rag_failure_diagnostics_clinic)
lists twelve patterns (grounding drift, chunk boundaries, query routing, evaluation blind spots and others). Use it as
vocabulary; the trace is your evidence.

## Step 4: Change one thing

Write a copy of `evals.score()` with your one change, and leave everything else as it is. Here the change is `k`, as
an example of the shape; yours is the change your diagnosis points to:

```python
def score_changed(record, store, settings, k=3):
    if "pack" in record:
        hits = [(document, None) for document in store.get_by_ids(record["pack"])]
    else:
        hits = lc_pipeline.retrieve(store, record["question"], k=k)      # the one change
    reply = lc_pipeline.answer(record["question"], lc_pipeline.evidence(hits), settings)
    refused = reply.answer.strip() == evals.REFUSAL
    passed = refused if record["refuse"] else evals.found(reply.answer, record["references"])
    return {"question_id": record["question_id"], "retrieved": [d.id for d, _ in hits],
            "answer": reply.answer, "passed": passed}

after = [score_changed(record, store, settings, k=5) for record in golden]
print(f"passed {sum(row['passed'] for row in after)} of {len(after)}")
```

Other changes fit the same shape: `retrieve(..., filter=...)` with a function of a `Document` (the policies'
`metadata["title"]` starts with the product), Lab 08's `bm25_search()` and `rrf()` in place of `retrieve()`, a
rewritten question from Lab 09, or `answer(..., rules=...)` with one line added to `shared.answer.RULES`. Change one
thing; two changes at once leave you unable to say which one helped.

## Step 5: Compare the two versions on every record

A fix for one record can break others. Compare per record with Lab 08's helpers:

```python
from shared.hybrid import paired, sign_test

table = paired([row["passed"] for row in before], [row["passed"] for row in after])
print(f"both {table['both']} | gained {len(table['gained'])} | lost {len(table['lost'])} | neither {table['neither']}")
for i in table["gained"] + table["lost"]:
    moved = "gained" if i in table["gained"] else "lost"
    print(moved, golden[i]["question_id"], "|", before[i]["answer"], "->", after[i]["answer"])
print(f"sign test: p = {sign_test(len(table['gained']), len(table['lost'])):.3f}")
```

Read every lost record's two answers before you count it. Lab 04 showed a right answer counted as a miss because the
answer key did not list its words, and Lab 05 that answers move between two runs of the same pipeline. Score both
versions a second time; a record that moves in only one of the two comparisons is noise, not your change.

### Checkpoint

| You should see | Why it matters |
|---|---|
| Your record fails before and passes after, in both comparisons | the change does what the diagnosis said |
| A list of lost records, each read | regressions are named, not hidden in a total |
| A p-value from the sign test | with a handful of changed records, a large p is expected: say so |

## Step 6: Write the report

One page, in this order:

1. **The failure.** The record, its question, the wrong answer, and how many runs it failed.
2. **The diagnosis.** The trace (hits, evidence, answer) and the stage you named, with the line of evidence that
   shows it.
3. **The change.** The one thing you changed and why that stage.
4. **Before and after.** Passed before and after on 38 records, the paired table, every gained and lost record with
   both answers, the sign test, and the second comparison.
5. **Limits.** What the 38 records cannot tell you (Lab 06: they compare versions of this pipeline, they do not
   estimate reliability), what your change costs (calls, tokens), and what you would test next.

| Part | Points | A full score |
|---|---|---|
| Diagnosis | 30 | the stage is named from the trace, and the evidence is shown |
| The change | 25 | one change, aimed at that stage, explained |
| Before-and-after evidence | 30 | all 38 records, paired, two comparisons, every lost record read |
| Limits and regressions | 15 | regressions named, the set's limits and the change's cost stated |

## What you can do now

- Choose a steady failure from saved evaluation results.
- Name the failing stage from a trace.
- Change one stage and keep the rest of the pipeline fixed.
- Show a change's gains and losses on the same records, and read each loss.
- Write a report that a colleague can rerun and check.

## Resources

- Lab 04, Part D: the stage table this guide uses.
- Lab 06, Part F: the 38 golden records and the three steady failures.
- Lab 08, Part D: `paired()` and `sign_test()`.
- [RAG Failure Diagnostics Clinic (Awesome LLM Apps)](https://github.com/Shubhamsaboo/awesome-llm-apps/tree/main/rag_tutorials/rag_failure_diagnostics_clinic):
  twelve named failure patterns and a script that asks a model to classify a bug report.
