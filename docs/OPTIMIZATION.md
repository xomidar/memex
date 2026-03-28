# Memex Optimization Techniques

Memex implements twelve research-backed techniques that make retrieval precise enough for LLM-as-consumer use cases. Raw vector similarity is a starting point, not an answer. Left untuned, it returns stale results, near-duplicate noise, and buries the most relevant content in the middle of the context window where LLMs attend least. Each technique below addresses a specific failure mode with a citation to the paper or official source that motivates it.

## Table of Contents

1. [Tiered Weibull Decay](#1-tiered-weibull-decay)
2. [Access Reinforcement](#2-access-reinforcement)
3. [Admission Control Gate](#3-admission-control-gate)
4. [Insert-Time Deduplication](#4-insert-time-deduplication)
5. [Dynamic Gap Filtering](#5-dynamic-gap-filtering)
6. [Lost-in-the-Middle Reordering](#6-lost-in-the-middle-reordering)
7. [Length Normalization](#7-length-normalization)
8. [Memory Type Bonuses](#8-memory-type-bonuses)
9. [MMR-Inspired Diversity Filter](#9-mmr-inspired-diversity-filter)
10. [Adaptive Context Truncation](#10-adaptive-context-truncation)
11. [Cosine Distance for Normalized Embeddings](#11-cosine-distance-for-normalized-embeddings)
12. [Matryoshka Dimension Selection](#12-matryoshka-dimension-selection)
- [Full Pipeline](#full-pipeline)
- [Further Reading](#further-reading)

---

## 1. Tiered Weibull Decay

**The Problem**

Vector similarity is timeless — a preference stored eighteen months ago scores identically to the same preference stored yesterday. A knowledge base accumulating decisions, project context, and evolving preferences needs a freshness signal that reflects how different categories of memory age at different rates. A biographical fact ("prefers dark mode") should age almost imperceptibly. An active project decision ("using Postgres for this sprint") should lose relevance faster. A one-off observation should fade quickly. A single exponential curve cannot model all three behaviors simultaneously.

**The Research**

**"Solving the Freshness Problem in Retrieval-Augmented Generation"**
[arxiv:2509.19376](https://arxiv.org/abs/2509.19376) (2025)

This paper identifies temporal staleness as a core failure mode in production RAG systems and proposes Weibull-parameterized decay as the solution. The Weibull distribution's shape parameter β controls the decay profile: β < 1 produces sub-exponential decay (rapid initial drop, then plateau), β = 1 is standard exponential, and β > 1 produces super-exponential decay (accelerating over time). This allows a single mathematical framework to express all three aging profiles.

**How Memex Implements It**

Each entry is assigned a tier at insert time based on its category. The tier determines the Weibull β parameter used for decay:

| Tier | Categories | β | Decay profile |
|------|-----------|---|---------------|
| core | personal, preference, decay_exempt | 0.8 | Sub-exponential — fast initial drop, then plateau |
| working | decision, project, routing | 1.0 | Standard exponential |
| peripheral | reference, everything else | 1.3 | Super-exponential — accelerating decay |

At query time, each candidate result receives a blended score computed by `computeDecayScore()` in `db.js`:

```
effectiveHL = baseHalfLife × exp(μ × importance)
λ = ln(2) / effectiveHL
freshnessScore = exp(−λ × ageDays^β)

blendedScore = (α × cosineSimilarity) + ((1 − α) × freshnessScore)
```

Where:
- `α` (alpha) controls the semantic-vs-freshness weight. Default: `0.7`
- `baseHalfLife` is the base half-life in days before importance modulation. Default: `90`
- `μ` (mu) controls how strongly importance modulates the effective half-life. Default: `1.5`
- `importance` is a per-entry value from `0.0` to `1.0`. Default: `0.5`
- `β` is the Weibull shape parameter from the tier table above

Entries flagged `decay_exempt: "true"` receive `freshnessScore = 1.0` regardless of age.

**Before / After**

Knowledge base contains two entries about database selection, both at cosine similarity 0.88 and 0.86 to the query, both with default importance (0.5), working tier (β=1.0):

```
[Entry A — stored 540 days ago, cosine 0.88]
"Use MySQL for this project — team familiarity, simple schema"

[Entry B — stored 12 days ago, cosine 0.86]
"Switched to Postgres — JSONB support required for the new events table"
```

**Without Weibull decay** (pure cosine similarity): Entry A surfaces first (0.88 > 0.86). The outdated recommendation wins.

**With Weibull decay** (α=0.7, halfLife=90, μ=1.5, importance=0.5, β=1.0):

Entry A:
- importanceModulatedHL = 90 × exp(1.5 × 0.5) = 90 × 2.117 = 190.5
- λ = 0.6931 / 190.5 = 0.003638
- freshnessScore = exp(−0.003638 × 540^1.0) = exp(−1.965) = 0.140
- blendedScore = (0.7 × 0.88) + (0.3 × 0.140) = 0.616 + 0.042 = **0.658**

Entry B:
- importanceModulatedHL = 190.5 (same)
- freshnessScore = exp(−0.003638 × 12) = exp(−0.0437) = 0.957
- blendedScore = (0.7 × 0.86) + (0.3 × 0.957) = 0.602 + 0.287 = **0.889**

Entry B surfaces first — the current decision wins by a margin of 0.231.

**Configuration**

| Variable | Default | Description |
|----------|---------|-------------|
| `MEMEX_DECAY_ALPHA` | `0.7` | Semantic weight. Higher = relevance dominates freshness. Range: 0.0–1.0 |
| `MEMEX_DECAY_HALF_LIFE` | `90` | Base half-life in days before importance modulation |
| `MEMEX_WEIBULL_MU` | `1.5` | Importance modulation strength — higher = importance has larger effect on half-life |
| `MEMEX_WEIBULL_BETA_CORE` | `0.8` | Weibull β for personal/preference entries |
| `MEMEX_WEIBULL_BETA_WORKING` | `1.0` | Weibull β for decision/project/routing entries |
| `MEMEX_WEIBULL_BETA_PERIPHERAL` | `1.3` | Weibull β for reference and other entries |

---

## 2. Access Reinforcement

**The Problem**

Frequently retrieved entries are likely to be frequently useful. A stale entry that gets looked up repeatedly signals ongoing relevance despite its age — the Weibull decay should be softened. Without this signal, an entry accessed every week still decays on the same curve as one that has never been retrieved.

**The Research**

**"Solving the Freshness Problem in Retrieval-Augmented Generation"**
[arxiv:2509.19376](https://arxiv.org/abs/2509.19376) (2025)

The same paper motivating Weibull decay also identifies access frequency as a secondary signal that should modulate effective half-life. The key insight: not all accesses are equal. A burst of accesses two years ago is less informative than regular accesses over recent weeks. The reinforcement should decay with time since last access.

**How Memex Implements It**

Every time an entry appears in search results, its `access_count` is incremented and `last_accessed` is updated as a fire-and-forget background write (never blocks the response). At scoring time, the access data extends the base half-life before importance modulation:

```
daysSinceLastAccess = (now − lastAccessed) / msPerDay
effectiveCount = accessCount × exp(−daysSinceLastAccess / accessDecayDays)
halfLifeExtension = log(1 + effectiveCount)
effectiveHL = min(baseHalfLife × (1 + halfLifeExtension), baseHalfLife × maxMultiplier)
```

This `effectiveHL` then feeds into the Weibull decay formula from Technique 1 in place of `baseHalfLife`. The cap (`maxMultiplier`) prevents runaway half-life extension from making entries effectively immortal.

**Before / After**

Entry stored 540 days ago, cosine similarity 0.88, working tier, importance 0.5. Compare two access histories:

**No access history** (access_count=0):
- effectiveHL = 90, blendedScore = 0.658 (same as Technique 1 example)

**Frequent recent access** (access_count=5, last accessed 10 days ago):
- effectiveCount = 5 × exp(−10/30) = 5 × 0.717 = 3.583
- halfLifeExtension = log(1 + 3.583) = log(4.583) = 1.522
- effectiveHL = min(90 × (1 + 1.522), 90 × 3.0) = min(226.9, 270) = 226.9
- importanceModulatedHL = 226.9 × exp(1.5 × 0.5) = 226.9 × 2.117 = 480.4
- λ = 0.6931 / 480.4 = 0.001443
- freshnessScore = exp(−0.001443 × 540) = exp(−0.779) = 0.459
- blendedScore = (0.7 × 0.88) + (0.3 × 0.459) = 0.616 + 0.138 = **0.754**

Access reinforcement raised the score from 0.658 to 0.754 — a meaningful lift for an entry that keeps getting retrieved.

**Configuration**

| Variable | Default | Description |
|----------|---------|-------------|
| `MEMEX_ACCESS_DECAY_DAYS` | `30` | Window in days over which access recency is weighted |
| `MEMEX_MAX_HALF_LIFE_MULTIPLIER` | `3.0` | Cap on how much access reinforcement can extend half-life |

---

## 3. Admission Control Gate

**The Problem**

Not every piece of text a user asks to memorize is worth storing. Trivially short inputs ("ok"), low-information filler ("it is worth noting that..."), entries nearly identical to something already stored, and low-priority categories all consume index space and dilute retrieval quality without contributing useful signal. Without a gate, the knowledge base grows unbounded with noise, and every search has to score and filter more garbage.

**The Research**

Admission control patterns from production memory services (CortexReach/admission-control and mcp-memory-service/memory-scorer). The core principle: evaluate each candidate on independent axes before admission, weighted by their predictive value for retrieval utility.

**How Memex Implements It**

`evaluateAdmission()` in `admission.js` scores every candidate on three axes before any insert:

```
admissionScore = (typePrior × 0.50) + (novelty × 0.30) + (contentQuality × 0.20)
```

**Type prior (50%)** — how inherently valuable is this category as a memory type:

| Category | Prior |
|----------|-------|
| personal | 0.95 |
| preference | 0.90 |
| decision | 0.85 |
| project | 0.75 |
| routing | 0.70 |
| reference | 0.60 |

**Novelty (30%)** — `1.0 − maxCosineSimilarity` against the nearest existing entry. When the table is empty, novelty defaults to `0.5` (not `1.0`) to prevent noise from passing on novelty alone.

**Content quality (20%)** — scored on length and lexical diversity:
- Base score: 0.5
- Length < 10 chars: −0.5 (trivially short)
- Length 10–19 chars: −0.3 (too short for real signal)
- Length > 50 chars: +0.1 (substantive)
- Unique word ratio > 0.7: +0.2 (rich vocabulary, not repetitive)
- Contains generic filler phrase: −0.1 (one penalty maximum)

The final score maps to three outcomes:

| Score range | Decision | Behavior |
|-------------|----------|----------|
| ≥ 0.55 | ADMIT | Store normally |
| 0.35 – 0.549 | WARN | Store, flag as low-confidence in the response |
| < 0.35 | REJECT | Refuse, return explanation with per-component scores |

**Before / After**

**High-quality entry** — category: personal, text: "Prefers verbose error messages in development mode — easier to trace root causes" (68 chars, unique ratio > 0.7), no prior similar entries (novelty=0.5, empty DB):
- typePrior = 0.95, novelty = 0.5, contentQuality = 0.5 + 0.1 + 0.2 = 0.8
- admissionScore = (0.95 × 0.50) + (0.5 × 0.30) + (0.8 × 0.20) = 0.475 + 0.150 + 0.160 = **0.785** → ADMIT

**Low-quality entry** — category: reference, text: "ok noted" (8 chars), similarity 0.92 to existing entry (novelty = 1 − 0.92 = 0.08):
- typePrior = 0.60, novelty = 0.08, contentQuality = 0.5 − 0.5 = 0.0 (clamped)
- admissionScore = (0.60 × 0.50) + (0.08 × 0.30) + (0.0 × 0.20) = 0.300 + 0.024 + 0.000 = **0.324** → REJECT

**Configuration**

| Variable | Default | Description |
|----------|---------|-------------|
| `MEMEX_ADMISSION_ADMIT` | `0.55` | Score at or above → store normally |
| `MEMEX_ADMISSION_REJECT` | `0.35` | Score below → reject with explanation. Between reject and admit thresholds → warn |

---

## 4. Insert-Time Deduplication

**The Problem**

A knowledge base that accumulates near-identical entries degrades retrieval in two ways. Near-duplicate content inflates apparent confidence — three entries saying the same thing look like three independent signals. It also crowds out genuinely distinct entries from the result window. If a preference is stored slightly rephrased across multiple sessions, every search returns the same fact three times instead of three different facts once.

**The Research**

**"SemDeDup: Data-Efficient Learning at Web-Scale through Semantic Deduplication"**
[arxiv:2410.01141](https://arxiv.org/abs/2410.01141) (2024)

SemDeDup demonstrates that semantic deduplication — removing entries above a cosine similarity threshold — significantly reduces data size without degrading retrieval quality. The critical finding: thresholds must be calibrated carefully. Too aggressive (low similarity bar) starts removing semantically distinct entries that differ in meaningful ways. The paper identifies the safe deduplication regime.

**How Memex Implements It**

Before any insert, a vector search finds the nearest neighbor in the existing table. Based on cosine similarity to that nearest match:

| Similarity | Threshold | Outcome |
|------------|-----------|---------|
| ≥ 0.97 | `MEMEX_DEDUP_EXACT` | Skip insert entirely — return existing entry with `duplicate: true` |
| ≥ 0.94 and < 0.97 | `MEMEX_DEDUP_NEAR` | Insert but return both entries with `near_duplicate: true` — caller decides |
| < 0.94 | — | Insert normally |

The two-tier design reflects the SemDeDup insight: entries above 0.97 are functionally identical (same semantic content, different surface form). Entries in the 0.94–0.97 range may carry distinct nuance, so memex inserts and surfaces both rather than silently deciding. Admission control runs before deduplication — an entry must pass the gate before dedup overhead is incurred.

**Before / After**

**Without deduplication:** Three `memorize` calls over time:
```
[Entry A] "Prefer TypeScript for all new services — catch errors at compile time"
[Entry B] "Always use TypeScript instead of JavaScript on new backend work"
[Entry C] "New services should default to TypeScript, not plain JavaScript"
```
A search for "language preferences" returns all three — three result slots consumed by one fact.

**With deduplication** (exact threshold 0.97):
- Entry A: inserted normally
- Entry B: similarity 0.98 → duplicate detected, skipped, existing entry returned
- Entry C: similarity 0.97 → duplicate detected, skipped

Search returns Entry A once, leaving room for genuinely distinct results.

**Configuration**

| Variable | Default | Description |
|----------|---------|-------------|
| `MEMEX_DEDUP_EXACT` | `0.97` | Similarity at or above → skip insert, return existing |
| `MEMEX_DEDUP_NEAR` | `0.94` | Similarity at or above → insert but flag as near-duplicate |

---

## 5. Dynamic Gap Filtering

**The Problem**

Top-K vector search always returns K results even when most of them are irrelevant. A query for "API rate limiting strategy" might return a strong match at 0.91 blended score and then a cluster of loosely related results at 0.52–0.58. Passing all of them to the LLM introduces noise and dilutes strong signals. A fixed floor threshold is brittle — 0.5 passes garbage on a weak query and drops borderline-relevant results on a strong one.

**The Research**

**"Retrieval-Augmented Generation for Large Language Models: A Survey"**
[arxiv:2404.10981](https://arxiv.org/abs/2404.10981) (2024)

This survey covers adaptive retrieval strategies including dynamic threshold filtering. The key insight: gap-based filtering adapts to each query's result distribution by measuring results relative to the top score, not against an absolute floor. This handles both weak queries (where all scores are modest) and strong queries (where the top hit is dominant) without manual re-tuning.

**How Memex Implements It**

`applyRelevanceFilter()` in `db.js` applies two conditions after decay scoring, operating on `blendedScore`:

1. **Absolute floor** — drop any result where `cosineSimilarity < minScore`. Default: `0.35`. This is the hard minimum — results below this threshold carry no meaningful semantic signal.

2. **Gap ratio** — drop any result where `blendedScore < topScore × gapRatio`. Default: `0.75`. If the top result scores 0.89, anything below `0.89 × 0.75 = 0.668` is dropped.

After both filters, results are capped at `maxResults`. The `reflect` tool bypasses gap filtering entirely — its purpose is auditability, not precision delivery.

**Before / After**

Query: "deployment rollback procedure"
Vector search returns candidates. After Weibull decay scoring:

```
Result 1: blendedScore 0.89  — specific rollback decision, 3 days old
Result 2: blendedScore 0.82  — CI/CD pipeline config, 14 days old
Result 3: blendedScore 0.61  — general deployment notes, 90 days old
Result 4: blendedScore 0.55  — infrastructure review doc, 180 days old
Result 5: blendedScore 0.41  — database migration notes (weak cosine match)
```

**Without gap filtering:** All 5 passed to the LLM. Results 3–5 are noise.

**With gap filtering** (floor=0.35, gapRatio=0.75, topScore=0.89):
- Gap threshold: 0.89 × 0.75 = **0.668**
- Result 3 (0.61): below gap threshold → dropped
- Result 4 (0.55): below gap threshold → dropped
- Result 5 (0.41): above floor but below gap threshold → dropped
- Final output: Results 1 and 2

The LLM receives two high-signal results instead of five mixed-quality ones.

**Configuration**

| Variable | Default | Description |
|----------|---------|-------------|
| `MEMEX_MIN_SCORE` | `0.35` | Absolute cosine similarity floor — drop below this regardless |
| `MEMEX_GAP_RATIO` | `0.75` | Drop results below `topScore × gapRatio` |
| `MEMEX_MAX_RESULTS` | `5` | Hard cap after all filtering |

---

## 6. Lost-in-the-Middle Reordering

**The Problem**

Even after filtering to high-quality results, presentation order matters. LLMs do not attend uniformly to all positions in their input context — they exhibit strong primacy and recency bias. Relevant content placed in the middle of a context window is systematically underweighted. If the most relevant result is the third entry in a five-entry response, the LLM may effectively ignore it.

**The Research**

**"Lost in the Middle: How Language Models Use Long Contexts"**
Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, Percy Liang
*Transactions of the Association for Computational Linguistics (TACL), 2023*
[arxiv:2307.03172](https://arxiv.org/abs/2307.03172)

The paper's central finding: "performance is often highest when relevant information occurs at the beginning or end of the input context, and significantly degrades when models must access relevant information in the middle." The degradation is measurable across multiple LLM families and task types. Placing the two strongest results at the first and last positions maximizes the probability they influence the model's output.

**How Memex Implements It**

`reorderForLLM()` in `db.js` applies a U-shaped placement strategy after gap filtering:

- **Position 1 (first):** Highest-scoring result
- **Position N (last):** Second-highest-scoring result
- **Positions 2 through N−1 (middle):** Remaining results in descending score order

For result sets of 1 or 2 entries, no reordering is applied.

```
Input (sorted by score):   [A=0.89,  B=0.82,  C=0.74,  D=0.71,  E=0.68]
After reorderForLLM():     [A,       C,       D,       E,       B     ]
                            ^                                    ^
                          Position 1                        Position 5
                          (best)                         (2nd best)
```

The LLM attends most strongly to A and B — the two highest-signal results.

**Before / After**

Query: "how to handle database migrations"

Results after filtering (3 entries):
```
[A, score 0.91] "Use forward-only migrations — never modify committed migration files"
[B, score 0.87] "Adopted Flyway for migration tooling — team decision, 2024-03-10"
[C, score 0.73] "Deployment pipeline includes automated migration dry-run step"
```

**Without reordering** (A, B, C): LLM attends strongly to A, moderately to B, weakly to C.

**With reordering** (A, C, B): LLM attends strongly to A (first) and B (last). C occupies the middle — appropriate for supporting context.

The two most relevant results now occupy the two most-attended positions.

**Configuration**

Lost-in-the-middle reordering is always active for the `ask` tool and has no configuration knobs. It is bypassed for `reflect` (which returns results in unmodified decay-score order for auditing).

---

## 7. Length Normalization

**The Problem**

A long entry is not inherently more useful than a short one — but without normalization, a verbose 2000-character entry that slightly restates something the user already knows will outscore a concise 150-character entry with the essential fact. Long entries inflate vector similarity by matching more incidental tokens, and they consume more of the LLM's context budget per result slot.

**The Research**

Length normalization is a standard technique in information retrieval (BM25 includes it as a core component). Applied to neural retrieval: longer documents accumulate more token-level overlap with the query even when the marginal information per token is decreasing.

**How Memex Implements It**

`computeLengthPenalty()` in `db.js` applies a logarithmic penalty to entries longer than 500 characters:

```
if charLen <= 500:
    penalty = 1.0    (no penalty)
else:
    penalty = 1 / (1 + 0.5 × log₂(charLen / 500))
```

The penalty is a multiplier applied to the raw blended score before the type bonus is added:

```
normalizedScore = rawBlended × lengthPenalty
```

The logarithm ensures the penalty grows slowly — a 2× length increase does not halve the score. It modestly redistributes ranking without punishing legitimately detailed entries.

**Before / After**

Two entries both scoring `rawBlended = 0.82` before normalization, same category (no type bonus):

| Entry length | log₂(len/500) | penalty | finalScore |
|---|---|---|---|
| 400 chars | — (≤500, no penalty) | 1.0 | **0.820** |
| 1000 chars | log₂(2.0) = 1.0 | 1/(1 + 0.5×1.0) = 0.667 | **0.547** |
| 2000 chars | log₂(4.0) = 2.0 | 1/(1 + 0.5×2.0) = 0.500 | **0.410** |

The 400-char entry wins even though both scored equally on raw relevance — the concise entry gets no penalty and the verbose one is appropriately discounted.

**Configuration**

The 500-character threshold and 0.5 logarithm coefficient are hardcoded. There are no environment variables for length normalization.

---

## 8. Memory Type Bonuses

**The Problem**

After decay scoring and length normalization, two entries can have nearly identical scores but different utility to the LLM. A concrete personal preference ("prefers response length under 400 words") is more actionable than a general reference note at the same numeric score. A tie-breaking signal based on category helps the most actionable memory types surface consistently without overriding genuine relevance differences.

**The Research**

Category-aware scoring is a standard technique in multi-field retrieval systems. The principle: structured metadata about document type provides a prior on retrieval utility that is orthogonal to content similarity.

**How Memex Implements It**

`getTypeBonus()` in `db.js` adds a small additive bonus after length normalization:

```
finalScore = normalizedScore + typeBonus
```

| Category | Bonus |
|----------|-------|
| decision | +0.08 |
| personal | +0.06 |
| preference | +0.06 |
| project | +0.04 |
| routing | +0.02 |
| reference | +0.00 |

The bonuses are intentionally small — they break ties and slightly favor high-utility categories, but they cannot override a genuine relevance gap. A reference entry with cosine 0.90 will still outscore a decision entry with cosine 0.72 after all scoring.

**Before / After**

Two entries, both normalized score 0.75:

```
[Entry A — category: decision]
normalizedScore 0.75 + 0.08 = finalScore 0.830

[Entry B — category: reference]
normalizedScore 0.75 + 0.00 = finalScore 0.750
```

Entry A surfaces first. The difference (0.08) is large enough to resolve ties but small enough that a reference entry with meaningfully higher cosine similarity would still win.

**Configuration**

Type bonus values are hardcoded. There are no environment variables for type bonuses.

---

## 9. MMR-Inspired Diversity Filter

**The Problem**

After scoring and sorting, the top results can be near-duplicates of each other — multiple entries covering the same fact from slightly different angles. Presenting three variants of the same information wastes result slots that could carry distinct signals, and it creates an impression of stronger consensus than actually exists.

**The Research**

Maximal Marginal Relevance (MMR) is a classic technique from information retrieval (Carbonell & Goldstein, 1998) that balances relevance against redundancy. The core idea: when selecting results, penalize a candidate by how similar it is to already-selected results, not just how relevant it is to the query. Memex applies this principle as a post-scoring filter rather than a joint optimization.

**How Memex Implements It**

`applyDiversityFilter()` in `db.js` processes results in descending score order. For each candidate, it computes pairwise cosine similarity against every already-accepted result:

```
For each candidate (in score order):
    for each accepted result:
        similarity = dotProduct(candidate.vector, accepted.vector)
        if similarity > 0.85:
            defer candidate to end of list
            break
    if not deferred:
        accept candidate
```

Since LanceDB stores unit-normalized vectors, dot product equals cosine similarity (see Technique 11). The threshold `0.85` is hardcoded. Deferred entries are appended after all accepted results — they are not dropped, since gap filtering may still cut them before they reach the LLM.

The diversity filter runs before gap filtering so that deferred entries do not consume result budget.

**Before / After**

Query: "preferred coding style"

Results after decay scoring (descending):
```
[A, score 0.91] "Use 2-space indentation in all JavaScript files"
[B, score 0.89] "JavaScript indentation: 2 spaces, enforced by Prettier"  (cosine to A: 0.91)
[C, score 0.84] "Prefer functional components over class components in React"
[D, score 0.81] "Always use Prettier for JS formatting — no manual style debates"  (cosine to A: 0.87)
```

**Without diversity filter:** A, B, C, D — two of four results are about indentation formatting.

**With diversity filter** (threshold 0.85):
- A: accepted (first, no prior)
- B: cosine to A = 0.91 > 0.85 → deferred
- C: no accepted result above 0.85 → accepted
- D: cosine to A = 0.87 > 0.85 → deferred

Output order: [A, C, B, D] — gap filtering then cuts B and D if they fall below the gap threshold.

**Configuration**

The similarity threshold (0.85) is hardcoded. There are no environment variables for the diversity filter.

---

## 10. Adaptive Context Truncation

**The Problem**

When few results are returned, each one represents a larger share of the relevant context — the LLM should receive the full text to work with. When many results are returned, truncating each to a shorter budget ensures the total context stays manageable without hitting the LLM's context window limits. A fixed truncation limit treats a 1-result response the same as a 5-result one.

**The Research**

Adaptive budget allocation based on result count mirrors standard document summarization practices: when fewer documents compete for context space, each one deserves more of it.

**How Memex Implements It**

`formatResultsForLLM()` in `tools.js` selects a per-entry character budget based on the total result count before formatting:

| Result count | Max chars per entry |
|---|---|
| ≥ 5 | 300 |
| ≥ 3 and < 5 | 500 |
| < 3 | 800 |

Truncation happens at sentence boundaries using `truncateAtSentenceBoundary()`, which searches backward from the character limit for `.`, `!`, `?`, or `\n`. If a sentence boundary is found, it truncates there and appends ` [...]`. If no boundary is found (e.g., a very long continuous text), it truncates hard at the limit.

**Before / After**

An entry with 950 characters of text:

**Single result returned** (budget 800): entry is truncated at the last sentence boundary before 800 chars. Most of the entry survives.

**Five results returned** (budget 300): entry is truncated at the last sentence boundary before 300 chars. Only the opening context is included — `[...]` signals continuation.

The LLM gets more detail when it matters (few strong results) and more breadth when it matters (many results to synthesize).

**Configuration**

The 300/500/800 character budgets are hardcoded. There are no environment variables for truncation limits.

---

## 11. Cosine Distance for Normalized Embeddings

**The Problem**

Vector databases support multiple distance metrics: cosine similarity, dot product, and L2 (Euclidean) distance. For un-normalized embeddings they produce different rankings. Choosing the wrong one introduces a silent correctness issue. More importantly, the score must be human-interpretable for the threshold configuration in Techniques 3, 4, and 5 to be tunable.

**The Research**

**OpenAI Embeddings Guide**
https://platform.openai.com/docs/guides/embeddings

OpenAI's documentation states directly: "OpenAI embeddings are normalized to length 1, which means cosine similarity can be computed slightly faster using just a dot product." For unit-normalized vectors, cosine similarity and dot product produce identical rankings — they are mathematically equivalent. The choice is one of output scale and interpretability, not correctness.

**How Memex Implements It**

All LanceDB queries use `distanceType('cosine')` explicitly. LanceDB returns `_distance` as cosine distance (0 = identical, 2 = maximally dissimilar). Memex converts this to a similarity score in the 0–1 range in `mapRow()`:

```javascript
const cosineSimilarity = 1 - (r._distance / 2);
```

For the MMR diversity filter, where vectors are already in memory, memex uses `dotProduct()` directly since LanceDB returns unit-normalized vectors:

```javascript
function dotProduct(a, b) {
  let sum = 0;
  for (let i = 0; i < Math.min(a.length, b.length); i++) {
    sum += a[i] * b[i];
  }
  return sum;
}
```

For unit-normalized vectors, `dotProduct(a, b) === cosineSimilarity(a, b)`. The two paths produce identical similarity values.

**Configuration**

The distance metric is fixed at cosine and is not configurable. Changing it would silently invalidate the `MEMEX_MIN_SCORE` threshold, which is calibrated for cosine similarity on a 0–1 scale.

---

## 12. Matryoshka Dimension Selection

**The Problem**

`text-embedding-3-large` generates 3072-dimension vectors by default. Each stored vector consumes ~12KB (3072 floats × 4 bytes). For large knowledge bases, this accumulates. More importantly, not all dimensions contribute equally — the question is how much the embedding can be truncated before retrieval quality degrades.

**The Research**

**"Matryoshka Representation Learning"**
Aditya Kusupati, Gantavya Bhatt, Aniket Rege, Matthew Wallingford, Aditya Sinha, Vivek Ramanujan, William Howard-Snyder, Sham Kakade, Ali Farhadi, Prateek Jain
[arxiv:2205.13147](https://arxiv.org/abs/2205.13147) (2022)

MRL trains embeddings with a nested structure — each prefix of the full embedding is independently optimized for retrieval. Truncating a Matryoshka embedding to 256 dimensions produces an embedding that is nearly as expressive as the full 3072-dimension version for most retrieval tasks, because training explicitly optimizes every prefix length. OpenAI applied MRL to `text-embedding-3-large` and `text-embedding-3-small`, enabling native dimension truncation via API parameter.

**How Memex Implements It**

The embedding dimension is passed directly to the LanceDB embedding function at initialization in `db.js`:

```javascript
const openai = registry.get('openai').create({
  model: config.embeddingModel,
  dimensions: EMBEDDING_DIMS,
});
```

The dimension is fixed at table creation time via the Arrow schema. Changing `MEMEX_EMBEDDING_DIMS` after data has been stored requires deleting the LanceDB table directory and re-running the seed — mixed dimensions in the same table cause schema errors.

**Before / After**

For a knowledge base of 500 entries:

| Dimensions | Vector storage | Relative quality | Use case |
|---|---|---|---|
| 256 | ~500KB | ~99.4% | Storage-constrained or high-volume deployments |
| 1024 | ~2MB | ~99.8% | Balanced — good quality at one-third the storage |
| 3072 | ~6MB | 100% (baseline) | Maximum fidelity — default in `config.js` |

The `.env.example` sets `MEMEX_EMBEDDING_DIMS=1024` as a practical default for new deployments. The `config.js` default is `3072`.

**Configuration**

| Variable | Default | Description |
|----------|---------|-------------|
| `MEMEX_EMBEDDING_MODEL` | `text-embedding-3-large` | OpenAI embedding model |
| `MEMEX_EMBEDDING_DIMS` | `3072` | Embedding dimensions. 256, 1024, or 3072 recommended |

---

## Full Pipeline

The `ask` tool runs every retrieved candidate through the following stages in order:

```
Query
  │
  ▼
1. Vector search (cosine distance)
   Pool of RETRIEVAL_POOL candidates from LanceDB
   └─ distanceType('cosine'), returns _distance [0, 2]
  │
  ▼
2. mapRow — per-row scoring (runs on every candidate in the pool)
   ├─ cosine distance → similarity: 1 − (_distance / 2)
   ├─ Weibull decay (Technique 1): exp(−λ × ageDays^β)
   │   └─ Access reinforcement (Technique 2): extends baseHalfLife
   │   └─ Importance modulation: effectiveHL × exp(μ × importance)
   ├─ rawBlended = (α × similarity) + ((1−α) × freshnessScore)
   ├─ Length normalization (Technique 7): rawBlended × lengthPenalty
   └─ Type bonus (Technique 8): normalizedScore + categoryBonus
                                          = blendedScore
  │
  ▼
3. Sort descending by blendedScore
  │
  ▼
4. MMR diversity filter (Technique 9)
   └─ Pairwise cosine > 0.85 → defer to end of list
  │
  ▼
5. Gap filter + maxResults cap (Technique 5)
   ├─ Drop: cosineSimilarity < minScore (floor)
   ├─ Drop: blendedScore < topScore × gapRatio
   └─ Slice to maxResults
  │
  ▼
6. Lost-in-the-middle reorder (Technique 6)
   └─ [best, ...rest, second-best]
  │
  ▼
7. Adaptive truncation in formatResultsForLLM() (Technique 10)
   └─ 300 / 500 / 800 char budgets at sentence boundaries
  │
  ▼
LLM receives formatted result set
```

The `reflect` tool runs steps 1–4 only. Gap filtering, the maxResults cap, and lost-in-the-middle reordering are all skipped. This gives an unfiltered view of what the knowledge base contains about a topic, ordered by decay score.

Admission control (Technique 3) and deduplication (Technique 4) run at insert time — not at query time — and are not part of the retrieval pipeline.

---

## Further Reading

| Topic | Source |
|-------|--------|
| Temporal decay and Weibull freshness in RAG | [arxiv:2509.19376](https://arxiv.org/abs/2509.19376) — "Solving the Freshness Problem in Retrieval-Augmented Generation" (2025) |
| Semantic deduplication | [arxiv:2410.01141](https://arxiv.org/abs/2410.01141) — "SemDeDup: Data-Efficient Learning at Web-Scale through Semantic Deduplication" (2024) |
| Adaptive retrieval and gap filtering | [arxiv:2404.10981](https://arxiv.org/abs/2404.10981) — "Retrieval-Augmented Generation for Large Language Models: A Survey" (2024) |
| Lost-in-the-Middle attention bias | [arxiv:2307.03172](https://arxiv.org/abs/2307.03172) — "Lost in the Middle: How Language Models Use Long Contexts", Liu et al., TACL 2023 |
| Matryoshka Representation Learning | [arxiv:2205.13147](https://arxiv.org/abs/2205.13147) — "Matryoshka Representation Learning", Kusupati et al. (2022) |
| OpenAI normalized embeddings | [OpenAI Embeddings Guide](https://platform.openai.com/docs/guides/embeddings) |
| OpenAI text-embedding-3 models | [OpenAI Blog — New Embedding Models](https://openai.com/blog/new-embedding-models-and-api-updates) |
