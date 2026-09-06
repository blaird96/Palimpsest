# Palimpsest `src` code review

Review date: 2026-09-06  
Scope: `src/ingest.py`, `src/search.py`, `src/check_environment.py`, `src/processing/config.py`, `src/__init__.py`, `src/processing/__init__.py`  
Out of scope for fixes: this document reports issues only; it does not change runtime code.

Severity:

- **P0** — incorrect results, data corruption, or likely runtime failure on the current corpus
- **P1** — real bugs, fragile contracts, or missing safeguards that will bite during normal use
- **P2** — design / maintainability / test gaps that will slow the project or hide P0/P1 bugs
- **P3** — style, naming, and small polish

---

## Executive summary

The ingest/search path is a thin Chroma + Ollama pipeline: discover `metadata.json` + `source.txt` pairs, character-chunk the text, embed with `embeddinggemma`, upsert, then query by embedding. The happy path is readable and small. The main risks are **stale index rows on re-ingest**, **character chunking that splits words and keeps Gutenberg boilerplate**, **duplicated settings that can drift between ingest and search**, **no embedding-count or metadata-type checks**, and **tests that currently fail or do not exercise the pipeline**.

---

## P0 — correctness and data integrity

### 1. Re-ingest does not replace a source; it can orphan old chunks

**Where:** `src/ingest.py` — `create_chunk_id()`, `ingest_source()`

Chunk IDs are `{source_id}:{chunk_number}:{sha256(text)[:12]}`. `collection.upsert` only updates rows with the **same** ID.

If a source is re-ingested after any of these change:

- `source.txt` content
- `normalize_text()` behavior
- `ChunkConfig.size` / `overlap`

then new IDs are written and **old passages remain in the collection**. Search can return duplicate or obsolete text for the same work.

**Recommendation:** delete existing IDs for `source_id` (or the whole collection on full rebuild) before upsert, or use a stable ID such as `{source_id}:{chunk_number}` and always overwrite.

### 2. Chunk IDs are not unique under hash collision (truncated SHA-256)

**Where:** `src/ingest.py` — `create_chunk_id()`

Only 12 hex characters (48 bits) of the digest are kept. A collision between two different chunks of the same source at the same `chunk_number` is unlikely, but if it happens Chroma will silently overwrite. Combined with item 1, the hash suffix is doing the opposite of what a research index usually wants: **identity should be stable; content should be overwritten**.

### 3. Character windows split words, sentences, and citations

**Where:** `src/ingest.py` — `chunk_text()`; `src/processing/config.py` — `ChunkConfig` (`size=1200`, `overlap=200`)

Chunking is raw Unicode-code-point slicing (`text[start:end]`), then `.strip()`. There is no split on paragraph, sentence, or token boundaries. Corpus files use `citation_scheme` values such as `book-chapter-verse` and `book-and-section`, but those schemes are never applied.

Effects:

- Embeddings mix half-sentences from adjacent passages.
- Search hits cannot be cited as “Book 2.5” or “John 1:1”; only `chunk_number` is stored.
- `.strip()` can drop leading/trailing whitespace that still counted toward the window, so overlap is not a true 200-character textual overlap after stripping.

### 4. Project Gutenberg (and similar) front/back matter is indexed as scripture/philosophy

**Where:** `src/ingest.py` — `load_source()`, `normalize_text()`; corpus `source.txt` files (e.g. KJV)

`normalize_text()` only collapses whitespace. It does not strip license headers, `*** START OF THE PROJECT GUTENBERG EBOOK ... ***`, or end-of-file license blocks. Those chunks are embedded with the work’s `title` / `author` / `tradition` metadata, so a query can rank Gutenberg legalese as if it were the work.

### 5. Embedding vector count is not checked against batch size

**Where:** `src/ingest.py` — `embed_batch()`, `ingest_source()`

`embed_batch` returns whatever `response.embeddings` contains. If Ollama returns fewer (or more) vectors than `len(batch)`, `collection.upsert` will fail or, worse depending on client version, associate vectors with the wrong documents.

`tests/test_ingest.py` currently mocks **one** embedding for **two** texts and asserts that single vector is returned, so the test encodes the bug instead of catching it.

### 6. Collection distance space is unspecified

**Where:** `src/ingest.py` — `get_collection()`

`get_or_create_collection` does not set `hnsw:space` (or equivalent). Chroma’s default is typically **L2**. EmbeddingGemma is generally retrieved with **cosine** (often after normalization). Ingest and search both pass precomputed vectors, so ranking uses whatever space the collection was first created with. An existing `data/chroma` directory created under an old default will **not** pick up a later metadata change (`get_or_create` does not migrate space).

Wrong metric → systematically worse neighbors, not a crash.

---

## P1 — runtime failures, fragile contracts, operational gaps

### 7. Duplicated `PATH` / `OLLAMA` constants will drift

**Where:** `src/ingest.py`, `src/search.py` (and model name again in `src/check_environment.py`)

`PROJECT_ROOT`, `DATABASE_PATH`, `COLLECTION_NAME`, and `EMBEDDING_MODEL` are copied. If ingest starts writing a new collection or model and search is not updated, search either misses the index or queries with a **different embedding model** (vectors are not comparable).

`OLLAMA.COLLECTION_NAME` is also misnamed: the collection is a Chroma name, not an Ollama setting.

### 8. `CONFIG` is unused; `main()` ignores it

**Where:** `src/ingest.py` — class `CONFIG` vs `main()`

`main()` constructs fresh `ChunkConfig()` and `EmbeddingConfig()` instead of `CONFIG.chunk_config` / `CONFIG.embedding_config`. Tests use `CONFIG`. Operators cannot change one place and have ingest follow.

`EmbeddingConfig.batch_size` is never validated. `batch_size <= 0` makes `range(0, n, batch_size)` raise `ValueError`.

### 9. Metadata is unsanitized before Chroma `upsert`

**Where:** `src/ingest.py` — `load_source()`, `ingest_source()`

`load_source` requires keys `id`, `title`, `author`, `tradition`, `period`, `genre`, `source_type` but:

- Does not require `metadata` to be a `dict` (`json.loads` can return a list).
- Does not coerce types. Chroma metadata values must be `str`, `int`, `float`, or `bool`. Nested objects, lists, or `null` will fail at upsert. Current corpus JSON is flat and safe; the code will not stay safe when a field is added.
- Spreads **all** JSON keys into each chunk (`**source_metadata`). Extra keys are fine today; a future `notes: []` is not.
- Does not validate that `id` is a non-empty string.

Latin Confessions metadata omits `translator` / `translation_year`; that is valid given the required set, but search UI should not assume those keys exist.

### 10. `discover_sources` fails the entire run on the first incomplete directory

**Where:** `src/ingest.py` — `discover_sources()`

A `metadata.json` without `source.txt` raises immediately. One broken edition blocks ingest of every other work. There is no per-source try/except in `main()` either: an Ollama timeout halfway through leaves a **partial** index (see P0.1).

### 11. Ingest is not resumable and has no CLI

**Where:** `src/ingest.py` — `main()`

There is no `--source`, `--limit`, or skip-if-unchanged check. Full-corpus ingest (KJV alone is ~100k lines) means many embedding round-trips. A crash requires starting over, and because of P0.1 a retry can duplicate data.

No logging module: only `print`. Failures from `ollama.embed` / Chroma are uncaught and unretried.

### 12. Search creates a new client every call and assumes a populated collection

**Where:** `src/search.py` — `search_passages()`, `main()`

- `PersistentClient` + `get_collection` on every query. Fine for a CLI; wasteful if this becomes a library.
- `get_collection` (not `get_or_create`) raises if ingest has not run — good — but `main()` does not catch it.
- Empty query after `input().strip()` raises `ValueError` with no user-facing handling.
- `zip(documents, metadatas, distances)` silently truncates if Chroma returns mismatched list lengths (`strict=True` would surface this).
- `n_results=5` is not clamped to collection size; empty or tiny collections may error or return fewer hits without explanation.
- `main()` indexes `metadata['title']`, `metadata['author']`, `metadata['chunk_number']` with no defaults.
- Rank line is missing a space: `f"[{passage['rank']}]{metadata['title']}"`.

### 13. Relative imports vs script execution

**Where:** `src/ingest.py`, `src/search.py` vs `src/check_environment.py`

Ingest and search use package-relative imports (`from .processing.config import ...`). They must be run as modules, e.g. `python -m src.ingest`. `python src/ingest.py` fails with `ImportError`.

`check_environment.py` has no relative imports, so `python src/check_environment.py` works. Entry points are inconsistent, and `README.md` does not document either.

### 14. Environment check does not actually check

**Where:** `src/check_environment.py`

- Lists installed models but does **not** assert that `gemma3:4b` and `embeddinggemma` are present before calling `chat` / `embed`. Ollama model tags often appear as `embeddinggemma:latest`; a naive `in` check must account for that.
- `generation_response.message.content` is printed without checking it equals the requested string, so a wrong or unloaded model can still print “operational.”
- Does not check Chroma persistence path, collection existence, or that embedding dimensionality matches an existing index.

### 15. Windows `source_path` strings

**Where:** `src/ingest.py` — `ingest_source()` metadata `"source_path": str(source_directory.relative_to(PATH.CORPUS_ROOT))`

On Windows this stores backslashes. Filters, docs, or later POSIX machines will not match paths unless normalized (`as_posix()`).

### 16. Corpus root is hard-coded to `corpus/public`

**Where:** `src/ingest.py` — `PATH.CORPUS_ROOT`

`.gitignore` excludes `corpus/private`, but ingest never looks there. Private material cannot be indexed without a code change. That may be intentional; it is still an implicit product decision with no flag.

---

## P2 — architecture, tests, and project hygiene

### 17. `src/processing` is a stub

**Where:** `src/processing/__init__.py` (empty), `src/processing/config.py`

Only frozen dataclasses live here. Normalization, chunking, and embedding still live in `ingest.py`. The package name promises a pipeline that does not exist yet, which makes it easy to keep growing `ingest.py` instead of a real processing layer.

### 18. Tests are incomplete and currently incorrect

**Where:** `tests/test_ingest.py`

| Test | Problem |
|------|---------|
| `test_normalize_text` | Input has no extra inner whitespace or blank-line noise; does not prove collapsing or dropping empty lines. |
| `test_chunk_text` | Only the short-text path (one chunk). No overlap, no `ValueError` cases, no mid-string split, no empty result. |
| `test_embed_batch` | Asserts a **single** vector for two inputs. `assert_called_once_with(model=...)` omits `input=texts`, so this should **fail** against the real `ollama.embed(..., input=texts)` call. |

There are **no** tests for: `load_source`, required metadata, `discover_sources`, `create_chunk_id`, `ingest_source` (upsert/delete behavior), or `search_passages`.

### 19. No shared application config or dependency pins

**Where:** `requirements.txt`, duplicated classes in ingest/search

`requirements.txt` is:

```
chromadb
ollama
```

No versions. Chroma collection APIs and default metrics have changed across releases; unpinned installs can change ranking and `query` `include=` behavior.

### 20. `normalize_text` is lossy for verse/line-sensitive works

**Where:** `src/ingest.py` — `normalize_text()`

Non-empty lines are joined with a single space per original line, paragraphs joined with `\n\n`. Poetry, dialogue, and some biblical lineation become prose. Combined with character chunking, structure that users would search by is discarded before index time.

### 21. No generation path despite a generation model

**Where:** `src/check_environment.py` uses `gemma3:4b`; ingest/search never call chat

The environment check implies a RAG or commentary loop that is not implemented. Not a bug in isolation; it is a dangling dependency and an unused model pull.

### 22. Package markers are empty

**Where:** `src/__init__.py`, `src/processing/__init__.py`

Empty files are valid namespace packages in spirit, but they export nothing. Callers must import `src.ingest` internals. Fine for a prototype; weak if `src` is meant to be a library.

---

## P3 — style and small issues

### 23. Class-as-namespace for config

`class CONFIG`, `class PATH`, `class OLLAMA` with class attributes work, but they are not enumerations and invite accidental instantiation. Module-level constants or a single frozen dataclass would be clearer.

### 24. Formatting and naming nits

- `ingest_source(...)` parameter list is packed onto one line; `main()` has a trailing space in `def main() -> None: `.
- `# GLOBAL VARS` comment in `check_environment.py` is noise.
- `relative_path` in `ingest_source` is computed for logging, then `relative_to` is computed again for metadata.
- `get_collection` collection `metadata["description"]` uses extra parentheses around a string (harmless; looks like a one-tuple).

### 25. Search result type is `list[dict[str, Any]]`

No TypedDict / dataclass for a passage. Easy to typo keys (`document` vs `text`) as the UI grows.

---

## File-by-file checklist

### `src/ingest.py`

- [ ] Replace hash-suffixed IDs or delete-by-`source_id` before upsert
- [ ] Validate embedding count == batch length
- [ ] Validate `batch_size > 0`
- [ ] Sanitize / flatten metadata for Chroma
- [ ] Normalize `source_path` with `as_posix()`
- [ ] Catch per-source failures; continue or report
- [ ] Strip or skip Gutenberg wrappers before chunking
- [ ] Chunk on semantic boundaries (or at least paragraph)
- [ ] Set and persist vector space (`cosine` vs `l2`)
- [ ] Use a single shared config module
- [ ] Run as `python -m src.ingest` (document it)

### `src/search.py`

- [ ] Import shared path/model/collection constants
- [ ] Handle missing collection / empty index / empty query
- [ ] `zip(..., strict=True)` or length assertions
- [ ] Safe metadata access; fix rank print spacing
- [ ] Optional: reuse client; allow filters (`tradition`, `work_id`)
- [ ] Over-fetch candidates and diversify adjacent / duplicate chunks before display
- [ ] Return a typed passage result with citation / edition fields
- [ ] Add retrieval-evaluation hooks instead of coupling retrieval directly to CLI output

### `src/check_environment.py`

- [ ] Require models (with `:latest` suffix handling)
- [ ] Verify chat text and embedding rank/dim
- [ ] Align entry-point style with ingest/search
- [ ] Optionally ping Chroma path

### `src/processing/config.py`

- [ ] Single source of truth for chunk, embed, collection, paths
- [ ] Validate invariants in `__post_init__`
- [ ] Allow source-aware chunking strategy selection without duplicating configuration

### `src/processing/__init__.py` / `src/__init__.py`

- [ ] Export the public API when the pipeline is split out

---

## Suggested fix order

1. Make chunk IDs stable and delete prior rows for a source (P0.1–P0.2).
2. Assert `len(embeddings) == len(batch)` and fix the embed unit test (P0.5, P2.18).
3. Centralize Chroma path, collection name, model, and `hnsw:space` (P0.6, P1.7).
4. Stop indexing Gutenberg wrappers; then improve chunk boundaries (P0.3–P0.4).
5. Harden metadata and `discover_sources` so one bad edition cannot wipe a run (P1.9–P1.10).
6. Expand tests around `load_source`, chunk overlap, and search empty/missing index (P2.18).

---


## Retrieval refinement recommendations

The first live corpus test (`"Overcoming anger"`) is encouraging: the top five results were all substantively relevant and crossed traditions and translations. The Dhammapada, Ecclesiasticus, Proverbs, and Ephesians all surfaced coherent prescriptions around anger, restraint, forgiveness, and wrath. That suggests the basic embedding → Chroma retrieval path is already useful.

The next phase should therefore emphasize **quality refinement rather than adding generation features**. Fix the correctness problems above first, then improve how retrieval results are represented, diversified, measured, and cited.

### R1. Add result diversification / adjacent-chunk deduplication

**Observed behavior:** adjacent Douay-Rheims chunks from Ecclesiasticus 27–28 both ranked in the top five for `"Overcoming anger"`.

This is expected with overlapping chunks, but it wastes result slots. A comparative-research tool benefits more from five distinct textual loci than from two overlapping windows around the same passage.

**Recommendation:**

1. Retrieve more candidates than are displayed (for example `n_results=15`).
2. Collapse or penalize adjacent chunks from the same `source_id`.
3. Return the best `k` diversified passages (for example 5).

A first implementation does not need a full reranker. A simple rule such as “do not return chunks within ±1 chunk of an already-selected chunk from the same source” would eliminate much of the redundancy.

Later, this can become a Maximal Marginal Relevance (MMR) or reranking stage.

### R2. Make citation metadata first-class

Current search output identifies results by work title and `chunk_number`. That is enough for debugging but weak for research use.

For structured works, ingestion should preserve meaningful textual coordinates whenever possible:

- Bible: book, chapter, verse / verse range
- Dhammapada: chapter and verse range
- Marcus Aurelius: book and section
- Plato / classical works: work-specific section identifiers when available
- Generic prose: chapter / section heading plus chunk index as fallback

**Recommendation:** extend each chunk metadata record with fields such as:

```text
work_id
edition_id
translation
book
chapter
section
verse_start
verse_end
citation
chunk_number
```

Not every source needs every field. `citation` should provide a normalized human-readable fallback such as:

```text
Ecclesiasticus 28:2–12
Dhammapada XVII.221–227
Meditations 4.3
```

The goal is for `chunk_number` to remain an internal retrieval coordinate, not the primary citation shown to the user.

### R3. Introduce source-aware chunking

A single character-window strategy is convenient but treats radically different source structures as identical.

A Bible, a philosophical dialogue, poetry, and continuous prose should not necessarily be chunked by the same boundary rules.

**Recommendation:** keep `ChunkConfig` as the common interface, but allow source metadata to select a chunking strategy:

```text
book-chapter-verse  -> verse-aware grouping
book-and-section    -> section-aware grouping
chapter             -> paragraph / chapter-aware grouping
generic             -> paragraph / sentence-aware fallback
```

The first refinement does not need sophisticated NLP. Preserving explicit corpus structure before falling back to character limits would already be a large improvement.

### R4. Treat overlapping translations as related, not merely independent documents

The live test already returned both Douay-Rheims and World English Bible material. As more translations are added, a top-five query can easily become five versions of the same underlying passage.

That behavior is useful in one mode and harmful in another.

**Recommendation:** distinguish at least conceptually between:

- **Passage search:** translations may compete independently.
- **Comparative search:** parallel translations of the same textual locus are grouped, allowing more distinct ideas / traditions into the result set.
- **Translation comparison:** deliberately retrieve all available editions of one locus.

This requires separating the identity of the **work/locus** from the identity of the **edition/translation** in metadata.

For example:

```text
work_id: bible
locus: matthew-5-44
edition_id: douay-rheims-challoner
```

versus:

```text
work_id: bible
locus: matthew-5-44
edition_id: world-english-bible
```

### R5. Build a small retrieval evaluation suite before adding generation

Manual inspection is currently the most valuable test because the corpus is still small enough to understand directly. Turn that into a repeatable evaluation set.

Create a file such as:

```text
tests/retrieval_cases.json
```

with representative queries in several categories:

| Query type | Example | What it tests |
|---|---|---|
| Exact | `Let a man overcome anger by love` | Known-passage retrieval |
| Concrete semantic | `How should I control my anger?` | Paraphrase retrieval |
| Abstract semantic | `What is the proper response to resentment?` | Conceptual retrieval |
| Comparative | `What should a person do when someone wrongs them?` | Cross-tradition breadth |
| Indirect | `Is revenge justified?` | Retrieval without obvious lexical overlap |
| Negative / alien | `How should Kubernetes handle persistent volume claims?` | False-positive behavior |

Each case should record one or more expected works / loci rather than requiring an exact chunk ID.

Initially, output a human-readable report. Once enough cases exist, track metrics such as:

- hit rate / Recall@K
- Mean Reciprocal Rank (MRR)
- number of distinct works in top K
- number of duplicate / adjacent chunks in top K

This gives chunk-size, overlap, distance-space, and embedding-model changes something objective to compare against.

### R6. Calibrate distance empirically; do not invent a threshold

The observed `"Overcoming anger"` results had L2 distances roughly from `1.22` to `1.33`, while remaining highly relevant. Those numbers are meaningful only in the context of:

- the chosen embedding model,
- whether vectors are normalized,
- the collection distance metric,
- and the corpus itself.

**Recommendation:** after explicitly fixing the collection metric, log distances for the retrieval evaluation suite and label results as relevant / irrelevant. Then determine whether a useful rejection threshold exists.

Do not hard-code a “good distance” value based on a handful of examples.

### R7. Add query-time filters without making them the default

As the corpus grows, researchers will want questions such as:

```text
search only Stoic works
search only biblical texts
compare Buddhist and Christian sources
search a particular translation
exclude modern commentary
```

The metadata already contains fields such as `tradition`, `genre`, and `source_type`.

**Recommendation:** extend `search_passages()` with an optional Chroma `where` filter, while preserving unrestricted semantic search as the default.

This will also make retrieval tests more precise.

### R8. Separate retrieval from presentation

`search_passages()` currently returns a loose dictionary and `main()` immediately prints it.

Before adding an LLM response layer, create a stable passage result type, for example a dataclass / TypedDict containing:

```text
rank
text
distance
source_id
title
author
citation
translation
chunk_number
metadata
```

Then keep these concerns separate:

```text
query -> retrieve -> diversify -> format / display
```

That makes it much easier to reuse the same retrieval engine later for:

- CLI search
- retrieval evaluation
- comparative views
- RAG context assembly
- a future API / UI

### R9. Add corpus-level diagnostics

Once multiple large works are present, retrieval quality can be distorted by corpus composition.

Useful diagnostics include:

- chunk count by source / edition
- average chunk length by source
- duplicate-text count
- percentage of corpus occupied by each work
- missing citation metadata
- number of adjacent chunks returned together across evaluation queries

A very large source should not silently dominate because it contributes far more overlapping chunks than everything else.

### R10. Delay answer generation until retrieval is measurable

`gemma3:4b` is currently only exercised by the environment check. That is fine.

Do **not** make generation the next priority. An answer model can make weak retrieval look convincing, which hides the exact defects the project is currently well-positioned to discover.

The preferred progression is:

```text
corpus validation
    -> structural chunking
    -> embedding
    -> retrieval
    -> diversification
    -> citation-quality output
    -> retrieval evaluation
    -> generation / synthesis
```

Generation becomes useful once Palimpsest can reliably answer the more important question:

> Which passages did it choose, and why should those passages be trusted?

---

## Revised refinement order

After the P0/P1 correctness work in the existing suggested fix order, the retrieval-quality sequence should be:

1. **Establish a retrieval baseline** using a fixed set of exact, semantic, comparative, indirect, and negative queries.
2. **Improve structural metadata and citations** so retrieved passages can be identified by meaningful textual loci.
3. **Replace raw character chunking with source-aware boundaries**, preserving existing corpus structure where available.
4. **Add candidate over-fetch + adjacent-chunk diversification** so overlap does not consume result slots.
5. **Model work / edition / translation identity separately** to support translation grouping and comparison.
6. **Add optional query filters** for tradition, work, translation, genre, and source type.
7. **Measure retrieval metrics and distance distributions** before introducing thresholds or changing embedding models.
8. **Add corpus diagnostics** to catch source imbalance, duplicates, malformed metadata, and wrapper text.
9. **Only then add generative synthesis**, with retrieved passages and citations exposed rather than hidden behind the model response.

A useful milestone for this phase is:

> Given a curated set of questions, Palimpsest consistently retrieves relevant, diverse, correctly cited passages from the expected works before any LLM is allowed to synthesize an answer.


## Notes on what looks sound

- UTF-8 reads and required metadata key check are a good minimum contract.
- Passing explicit embeddings into Chroma (instead of a hidden embedding function) keeps ingest and query on the same model **as long as the model name stays shared**.
- `overlap < chunk_size` and `chunk_size > 0` guards prevent an infinite loop in `chunk_text`.
- `PATH.DATABASE_PATH.mkdir(parents=True, exist_ok=True)` on ingest is appropriate; search correctly does not create an empty DB just to query.
- Current public `metadata.json` files are flat primitives, so Chroma upsert should accept them today.
- `.gitignore` correctly keeps `data/chroma` and `corpus/private` out of git.
