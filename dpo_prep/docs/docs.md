# Jokes DPO Dataset Pipeline

Final dataset columns:
- `setup`
- `chosen_punchline`, `rejected_punchline`
- `chosen_score`, `rejected_score`

Used for Direct Preference Optimization (DPO).

![DPO pipeline](dpo_pipeline.png)
---

## 1. Cleaning

**Column mapping to common schema**  
- Dad jokes: `question → setup`, `response → punchline`  
- ReditJokes set: `body → setup`, `punchline → punchline`  
- One million jokes: `setup` built from title + text, `punchline` extracted from body.

**Text cleanup**
- Replace missing text with empty string.
- Strip whitespace.
- Drop rows with empty `setup` or `punchline`.

**Meta-only setups**  
Drop rows where `setup` is only:
- “TL;DR”-style text,  
- “NSFW” labels,  
- bare URLs,  
- extremely short / obviously meta content.  

**Reddit tails (where applicable)**  
Strip from joke body:
- trailing `EDIT:` sections,  
- trailing URLs, signatures, or credits.

---

## 2. Deduplication

### 2.1 Inside each dataset

Define `normalize_for_dedup(text)`:
- lowercase,
- remove escape sequences (`\n`, `\r`, `\t`),
- remove all non-alphanumeric characters.

Build a `cluster_key`, for example:
- Dad jokes: `norm(question) + " || " + norm(response)`
- ReditJokes set: `norm(body) + " || " + norm(punchline)`

Group by `cluster_key`, then:
- look at `score` values,
- keep the row whose score is closest to the **cluster median**,
- drop all other rows.

### 2.2 Building preference pairs

Define `normalize_setup(text)`:
- lowercase,
- collapse multiple spaces,
- strip edges.

Add `setup_norm = normalize_setup(setup)` and group by it.  
For each group:

- `hi` = row with max `score`  
- `lo` = row with min `score`  
- if `hi.score == lo.score`: skip  
- else create one pair:

  - `setup = hi.setup`  
  - `chosen_punchline = hi.punchline`  
  - `rejected_punchline = lo.punchline`  
  - `chosen_score = hi.score`  
  - `rejected_score = lo.score`

---

## 3. Length and quality filters

Use `apply_setup_filters` and extra checks to:

- drop meta-only setups again,
- enforce `min_chars` for setups (very short setups removed),
- enforce `max_chars` for setups,
- in merge stage: drop pairs where **either** punchline exceeds a `max_punchline_chars` limit (for example 128 characters).

---

## 4. Merge and final dedup

We start from three per-source DPO datasets (after cleaning and pair building):

- Dad jokes pairs  
- ReditJokes set pairs  
- One million jokes pairs  

### 4.1 Priority merge

Priority order: **Dad > ReditJokes > One million**.

`merge_with_priority(better, worse)`:

1. Compute `setup_norm = normalize_setup(setup)` in both.
2. Take all rows from `better`.
3. From `worse`, drop rows whose `setup_norm` already appears in `better`.
4. Concatenate and drop `setup_norm`.

Apply twice:
- `merged_1 = merge_with_priority(dad, ReditJokes)`
- `merged_all = merge_with_priority(merged_1, million)`

### 4.2 Final dedup

On `merged_all`:

- compute `setup_norm_dedup = normalize_for_dedup(setup)`,
- drop duplicates on `setup_norm_dedup`, keep first,
- drop helper column.

Save result as:

```text
data/generated_data/dpo_final_set.csv
