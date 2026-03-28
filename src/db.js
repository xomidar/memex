import * as lancedb from '@lancedb/lancedb';
import { LanceSchema, getRegistry } from '@lancedb/lancedb/embedding';
import '@lancedb/lancedb/embedding/openai';  // registers OpenAI in the embedding registry
import { Utf8 } from 'apache-arrow';
import { randomUUID } from 'crypto';
import { config } from './config.js';
import { evaluateAdmission } from './admission.js';
import { recordUsage } from './cost-tracker.js';

const DB_PATH = config.dataDir;
const TABLE_NAME = config.tableName;
const EMBEDDING_DIMS = config.embeddingDims;

// UUID validation — prevents injection via string-interpolated where clauses
const UUID_RE = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;
function assertUUID(id) {
  if (!UUID_RE.test(id)) throw new Error(`Invalid UUID: ${id}`);
}

// Retrieval pool pulled from config so it can be tuned without a code change.
// Must be > maxResults so gap/decay filtering has candidates to work with.
const RETRIEVAL_POOL = config.retrievalPool;

if (config.retrievalPool <= config.maxResults) {
  process.stderr.write(`[memex] WARNING: retrievalPool (${config.retrievalPool}) should be > maxResults (${config.maxResults})\n`);
}

let db = null;
let table = null;

// OpenAI embedding function using LanceDB's registry
function getEmbeddingFunction() {
  const registry = getRegistry();
  const openai = registry.get('openai').create({
    model: config.embeddingModel,
    dimensions: EMBEDDING_DIMS,
  });
  return openai;
}

/**
 * Estimate tokens from text using OpenAI's guideline (~4 chars per token for English).
 * This is the standard heuristic used across the industry when tiktoken isn't available.
 * Accuracy: within ~10% for English text.
 */
function estimateTokens(texts) {
  return texts.reduce((sum, t) => sum + Math.ceil((t || '').length / 4), 0);
}

// Arrow schema for the knowledge table
function buildSchema(embeddingFn) {
  return LanceSchema({
    id: new Utf8(),
    text: embeddingFn.sourceField(new Utf8()),
    vector: embeddingFn.vectorField(),
    category: new Utf8(),
    source: new Utf8(),
    tags: new Utf8(),
    created_at: new Utf8(),
    updated_at: new Utf8(),
    // decay_exempt: "true" | "false" (string for Arrow Utf8 compatibility)
    // Entries marked "true" skip the freshness penalty — use for biographical
    // facts, permanent preferences, and knowledge that doesn't age.
    decay_exempt: new Utf8(),
    // importance: "0.0" to "1.0" — modulates effective half-life via Weibull decay
    importance: new Utf8(),
    // access_count: "0", "1", ... — incremented on retrieval for access reinforcement
    access_count: new Utf8(),
    // last_accessed: ISO timestamp — updated on retrieval, used for access recency weighting
    last_accessed: new Utf8(),
    // tier: "core" | "working" | "peripheral" — auto-assigned from category, controls Weibull β
    tier: new Utf8(),
  });
}

export async function initDB() {
  if (db && table) return { db, table };

  try {
    db = await lancedb.connect(DB_PATH);
    const embeddingFn = getEmbeddingFunction();
    const schema = buildSchema(embeddingFn);

    const existingTables = await db.tableNames();
    if (existingTables.includes(TABLE_NAME)) {
      table = await db.openTable(TABLE_NAME);
    } else {
      table = await db.createEmptyTable(TABLE_NAME, schema, { mode: 'create' });
    }

    return { db, table };
  } catch (err) {
    db = null;
    table = null;
    throw new Error(`Failed to initialize LanceDB: ${err.message}`);
  }
}

// ---------------------------------------------------------------------------
// Tier assignment
// ---------------------------------------------------------------------------

/**
 * Assign a decay tier based on category.
 * Core   (β=0.8): sub-exponential — identity facts, permanent preferences
 * Working (β=1.0): standard exponential — active project context, decisions
 * Peripheral (β=1.3): super-exponential — temporary notes, one-off observations
 *
 * @param {string} category
 * @param {string} [decayExempt='false']
 * @returns {'core'|'working'|'peripheral'}
 */
function assignTier(category, decayExempt = 'false') {
  if (decayExempt === 'true') return 'core';
  switch (category) {
    case 'personal':
    case 'preference':
      return 'core';
    case 'decision':
    case 'project':
    case 'routing':
      return 'working';
    case 'reference':
    default:
      return 'peripheral';
  }
}

/**
 * Return the Weibull β parameter for a given tier.
 * @param {'core'|'working'|'peripheral'} tier
 * @returns {number}
 */
function getTierBeta(tier) {
  switch (tier) {
    case 'core':       return config.weibullBetaCore;
    case 'working':    return config.weibullBetaWorking;
    case 'peripheral': return config.weibullBetaPeripheral;
    default:           return config.weibullBetaWorking;
  }
}

// ---------------------------------------------------------------------------
// Tiered Weibull Decay with Access Reinforcement (Technique 1)
// ---------------------------------------------------------------------------

/**
 * Compute blended Weibull decay score.
 *
 * Weibull recency formula:
 *   effectiveHL = halfLife × exp(μ × importance)
 *   λ = ln(2) / effectiveHL
 *   recency = exp(-λ × daysSince^β)
 *
 * Access reinforcement extends half-life logarithmically:
 *   effectiveCount = rawAccessCount × exp(-daysSinceLastAccess / accessDecayDays)
 *   halfLifeExtension = log(1 + effectiveCount)
 *   finalHalfLife = baseHalfLife × (1 + halfLifeExtension), capped at baseHalfLife × maxMultiplier
 *
 * @param {number} cosineSimilarity
 * @param {string} createdAt - ISO timestamp
 * @param {boolean} decayExempt
 * @param {'core'|'working'|'peripheral'} tier
 * @param {number} [importanceVal=0.5] - 0.0 to 1.0
 * @param {number} [accessCount=0]
 * @param {string|null} [lastAccessed=null] - ISO timestamp or null
 * @returns {number}
 */
function computeDecayScore(
  cosineSimilarity,
  createdAt,
  decayExempt,
  tier = 'working',
  importanceVal = 0.5,
  accessCount = 0,
  lastAccessed = null,
) {
  const alpha = config.decayAlpha;

  let freshnessScore;
  if (decayExempt) {
    freshnessScore = 1.0;
  } else {
    const createdMs = new Date(createdAt).getTime();
    if (isNaN(createdMs)) {
      // Malformed timestamp — treat as mid-life rather than brand-new or dead
      freshnessScore = decayExempt ? 1.0 : 0.5;
    } else {
    const ageMs = Date.now() - createdMs;
    const ageDays = ageMs / (1000 * 60 * 60 * 24);

    // Base half-life from config
    const baseHalfLife = config.decayHalfLife;

    // Access reinforcement — weight recent accesses more, discount stale ones
    let effectiveHL = baseHalfLife;
    if (accessCount > 0) {
      const lastAccessedDate = lastAccessed ? new Date(lastAccessed) : new Date(createdAt);
      const daysSinceLastAccess = (Date.now() - lastAccessedDate.getTime()) / (1000 * 60 * 60 * 24);
      const effectiveCount = accessCount * Math.exp(-daysSinceLastAccess / config.accessDecayDays);
      const halfLifeExtension = Math.log(1 + effectiveCount);
      const cap = baseHalfLife * config.maxHalfLifeMultiplier;
      effectiveHL = Math.min(baseHalfLife * (1 + halfLifeExtension), cap);
    }

    // Importance modulation: higher importance → longer effective half-life
    const importanceModulatedHL = effectiveHL * Math.exp(config.weibullMu * importanceVal);

    // Weibull λ and β
    const lambda = Math.LN2 / importanceModulatedHL;
    const beta = getTierBeta(tier);

    // Weibull recency: exp(-λ × t^β)
    freshnessScore = Math.exp(-lambda * Math.pow(ageDays, beta));
    } // end valid-date else
  }

  return (alpha * cosineSimilarity) + ((1 - alpha) * freshnessScore);
}

// ---------------------------------------------------------------------------
// Length normalization (Technique 3) + Type bonuses (Technique 4)
// ---------------------------------------------------------------------------

/**
 * Compute length penalty multiplier.
 * Entries at or below 500 chars get 1.0 (no penalty).
 * Longer entries are penalized: 1 / (1 + 0.5 × log₂(charLen / 500))
 *
 * @param {number} charLen
 * @returns {number} 0.0 to 1.0
 */
function computeLengthPenalty(charLen) {
  if (charLen <= 500) return 1.0;
  return 1.0 / (1.0 + 0.5 * Math.log2(charLen / 500));
}

// Additive type bonuses — break ties, never override relevance.
const TYPE_BONUSES = {
  decision:   0.08,
  personal:   0.06,
  preference: 0.06,
  project:    0.04,
  routing:    0.02,
  reference:  0.00,
};

/**
 * Get the additive type bonus for a category.
 * @param {string} category
 * @returns {number}
 */
function getTypeBonus(category) {
  return TYPE_BONUSES[category] ?? 0.00;
}

// ---------------------------------------------------------------------------
// LLM reordering (lost-in-the-middle)
// ---------------------------------------------------------------------------

/**
 * Apply lost-in-the-middle reordering for optimal LLM attention.
 * Position 1: highest score. Position N (last): second highest. Middle: rest descending.
 * LLMs attend most strongly to the beginning and end of context.
 *
 * @param {Array} results - already sorted descending by blended score
 * @returns {Array}
 */
function reorderForLLM(results) {
  if (results.length <= 2) return results;

  const [first, second, ...rest] = results;
  // rest is already descending — middle positions are less attended, so park the weaker ones there
  return [first, ...rest, second];
}

// ---------------------------------------------------------------------------
// Relevance filtering
// ---------------------------------------------------------------------------

/**
 * Apply floor + gap-based relevance filtering.
 * Drops results below minScore floor, then drops anything more than (1 - gapRatio) below top.
 *
 * @param {Array} results - sorted descending by blended score
 * @param {number} maxResults - hard cap
 * @returns {Array}
 */
function applyRelevanceFilter(results, maxResults) {
  if (results.length === 0) return results;

  const { minScore, gapRatio } = config;
  const topScore = results[0].blendedScore;
  const gapThreshold = topScore * gapRatio;

  const filtered = results.filter(r =>
    r.score >= minScore && r.blendedScore >= gapThreshold
  );

  return filtered.slice(0, maxResults);
}

// ---------------------------------------------------------------------------
// MMR-Inspired Diversity Filter (Technique 6)
// ---------------------------------------------------------------------------

/**
 * Compute dot product (= cosine similarity for unit-normalized vectors).
 * LanceDB returns unit-normalized vectors, so dot product IS cosine similarity.
 *
 * @param {Float32Array|number[]} a
 * @param {Float32Array|number[]} b
 * @returns {number}
 */
function dotProduct(a, b) {
  if (a.length !== b.length) {
    process.stderr.write(`[memex] WARNING: vector dimension mismatch in dotProduct: ${a.length} vs ${b.length}\n`);
  }
  let sum = 0;
  const len = Math.min(a.length, b.length);
  for (let i = 0; i < len; i++) {
    sum += a[i] * b[i];
  }
  return sum;
}

/**
 * MMR-inspired diversity filter.
 * Processes results in descending score order. If a result has cosine similarity > 0.85
 * to ANY already-accepted result, it's deferred to the end of the list.
 * This prevents near-duplicate content from filling prime positions.
 *
 * @param {Array} results - sorted descending by blendedScore, with .vector attached
 * @returns {Array} - accepted results first, deferred appended at end
 */
function applyDiversityFilter(results) {
  if (results.length <= 1) return results;

  const SIMILARITY_THRESHOLD = 0.85;
  const accepted = [];
  const deferred = [];

  for (const result of results) {
    let isDuplicate = false;

    // Only run pairwise check if we have vectors to compare
    if (result.vector) {
      for (const acceptedResult of accepted) {
        if (acceptedResult.vector) {
          const similarity = dotProduct(result.vector, acceptedResult.vector);
          if (similarity > SIMILARITY_THRESHOLD) {
            isDuplicate = true;
            break;
          }
        }
      }
    }

    if (isDuplicate) {
      deferred.push(result);
    } else {
      accepted.push(result);
    }
  }

  return [...accepted, ...deferred];
}

// ---------------------------------------------------------------------------
// Row mapping
// ---------------------------------------------------------------------------

/**
 * Map a raw LanceDB row to a clean result object.
 * Applies Weibull decay, length normalization, and type bonus.
 * Gracefully handles missing fields (existing tables without new columns).
 */
function mapRow(r) {
  const cosineSimilarity = r._distance !== undefined ? (1 - r._distance / 2) : null;
  const decayExempt = (r.decay_exempt || 'false') === 'true';
  const tier = r.tier || assignTier(r.category, r.decay_exempt || 'false');
  const importanceVal = parseFloat(r.importance || '0.5');
  const accessCount = parseInt(r.access_count || '0', 10);
  const lastAccessed = r.last_accessed || null;

  let blendedScore = null;
  if (cosineSimilarity !== null) {
    // Base Weibull decay blend
    const rawBlended = computeDecayScore(
      cosineSimilarity,
      r.created_at,
      decayExempt,
      tier,
      importanceVal,
      accessCount,
      lastAccessed,
    );

    // Length normalization (Technique 3) — clamp penalty at 1.0 for short entries
    const lengthPenalty = computeLengthPenalty(r.text ? r.text.length : 0);
    const normalizedScore = rawBlended * lengthPenalty;

    // Type bonus (Technique 4) — additive, small, breaks ties
    blendedScore = normalizedScore + getTypeBonus(r.category);
  }

  return {
    id: r.id,
    text: r.text,
    category: r.category,
    source: r.source,
    tags: r.tags,
    created_at: r.created_at,
    updated_at: r.updated_at,
    decay_exempt: r.decay_exempt || 'false',
    importance: r.importance || '0.5',
    access_count: r.access_count || '0',
    last_accessed: r.last_accessed || null,
    tier,
    score: cosineSimilarity,
    blendedScore,
    // Keep vector for diversity filter — stripped before returning to callers
    vector: r.vector || null,
  };
}

// ---------------------------------------------------------------------------
// Access tracking (Technique 1 — fire-and-forget)
// ---------------------------------------------------------------------------

/**
 * Update access_count and last_accessed for a set of retrieved entry IDs.
 * Fire-and-forget — never awaited, never blocks the response.
 * After MAX_ACCESS_FAILURES consecutive failures, stops logging to avoid log spam.
 *
 * @param {string[]} ids
 * @param {Map<string, number>} accessCountMap - current access_count values by id
 */
let accessTrackingFailures = 0;
const MAX_ACCESS_FAILURES = 3;

function updateAccessTracking(ids, accessCountMap) {
  // Intentionally not awaited
  (async () => {
    try {
      await initDB();
      const now = new Date().toISOString();
      for (const id of ids) {
        assertUUID(id);
        const currentCount = accessCountMap.get(id) ?? 0;
        const newCount = currentCount + 1;
        await table.update({
          where: `id = '${id}'`,
          values: { access_count: String(newCount), last_accessed: now },
        });
      }
      // Reset failure counter on success
      accessTrackingFailures = 0;
    } catch (err) {
      accessTrackingFailures++;
      if (accessTrackingFailures <= MAX_ACCESS_FAILURES) {
        process.stderr.write(`[memex] access tracking update failed: ${err.message}\n`);
      }
      // Silently skip after MAX_ACCESS_FAILURES to avoid log spam
    }
  })();
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/**
 * addEntry — Insert a new entry with admission control and deduplication.
 *
 * Admission runs BEFORE dedup (dedup only runs if admitted).
 * Pass skipAdmission=true for bulk import to bypass the gate.
 *
 * Returns:
 *   { ...entry }                                       — normal insert
 *   { ...entry, admission: 'warn', admissionReason }   — low-confidence insert
 *   { rejected: true, reason }                         — admission rejected
 *   { ...existingEntry, duplicate: true }              — exact duplicate, skipped
 *   { new: entry, existing: match, near_duplicate: true } — near duplicate, caller decides
 *
 * @param {string} text
 * @param {string} category
 * @param {string} [source='user_explicit']
 * @param {string} [tags='']
 * @param {string} [decay_exempt='false']
 * @param {string} [importance='0.5']
 * @param {boolean} [skipAdmission=false]
 */
export async function addEntry(
  text,
  category,
  source = 'user_explicit',
  tags = '',
  decay_exempt = 'false',
  importance = '0.5',
  skipAdmission = false,
) {
  await initDB();
  const now = new Date().toISOString();
  const tier = assignTier(category, decay_exempt);

  // --- Admission control gate (Technique 2) ---
  // Run BEFORE dedup. Fetch candidates for novelty scoring (same search dedup would use).
  let admissionResult = null;
  if (!skipAdmission) {
    let dedupCandidates = [];
    try {
      dedupCandidates = await table.search(text).distanceType('cosine').limit(3).toArray();
    } catch {
      // Empty table or search failure — novelty defaults to 1.0 inside evaluateAdmission
      dedupCandidates = [];
    }

    admissionResult = evaluateAdmission(text, category, dedupCandidates);

    if (admissionResult.decision === 'reject') {
      return {
        rejected: true,
        reason: admissionResult.reason,
        score: admissionResult.score,
        components: admissionResult.components,
      };
    }

    // If warn or admit, proceed — but carry the warning through for the caller
    // Dedup still runs below with the same candidates
    if (dedupCandidates && dedupCandidates.length > 0) {
      const topMatch = dedupCandidates[0];
      const similarity = topMatch._distance !== undefined ? (1 - topMatch._distance / 2) : 0;

      if (similarity >= config.dedupExact) {
        const result = {
          id: topMatch.id,
          text: topMatch.text,
          category: topMatch.category,
          source: topMatch.source,
          tags: topMatch.tags,
          created_at: topMatch.created_at,
          updated_at: topMatch.updated_at,
          decay_exempt: topMatch.decay_exempt || 'false',
          importance: topMatch.importance || '0.5',
          access_count: topMatch.access_count || '0',
          last_accessed: topMatch.last_accessed || null,
          tier: topMatch.tier || assignTier(topMatch.category, topMatch.decay_exempt),
          duplicate: true,
          similarity,
        };
        // Attach admission warning even on duplicate path so caller isn't flying blind
        if (admissionResult?.decision === 'warn') {
          result.admissionWarning = admissionResult.reason;
        }
        return result;
      }

      if (similarity >= config.dedupNear) {
        const entry = {
          id: randomUUID(),
          text,
          category,
          source,
          tags: tags || '',
          created_at: now,
          updated_at: now,
          decay_exempt: decay_exempt || 'false',
          importance: importance || '0.5',
          access_count: '0',
          last_accessed: null,
          tier,
        };
        await table.add([entry]);
        const result = {
          near_duplicate: true,
          similarity,
          new: entry,
          existing: {
            id: topMatch.id,
            text: topMatch.text,
            category: topMatch.category,
            source: topMatch.source,
            tags: topMatch.tags,
            created_at: topMatch.created_at,
            updated_at: topMatch.updated_at,
            decay_exempt: topMatch.decay_exempt || 'false',
            importance: topMatch.importance || '0.5',
            tier: topMatch.tier || assignTier(topMatch.category, topMatch.decay_exempt),
          },
        };
        if (admissionResult.decision === 'warn') {
          result.admission = 'warn';
          result.admissionReason = admissionResult.reason;
        }
        return result;
      }
    }
  } else {
    // skipAdmission=true path — still run dedup for bulk import consistency
    try {
      const dupeCheck = await table.search(text).distanceType('cosine').limit(1).toArray();
      if (dupeCheck && dupeCheck.length > 0) {
        const topMatch = dupeCheck[0];
        const similarity = topMatch._distance !== undefined ? (1 - topMatch._distance / 2) : 0;

        if (similarity >= config.dedupExact) {
          return {
            id: topMatch.id,
            text: topMatch.text,
            category: topMatch.category,
            source: topMatch.source,
            tags: topMatch.tags,
            created_at: topMatch.created_at,
            updated_at: topMatch.updated_at,
            decay_exempt: topMatch.decay_exempt || 'false',
            importance: topMatch.importance || '0.5',
            access_count: topMatch.access_count || '0',
            last_accessed: topMatch.last_accessed || null,
            tier: topMatch.tier || assignTier(topMatch.category, topMatch.decay_exempt),
            duplicate: true,
            similarity,
          };
        }

        if (similarity >= config.dedupNear) {
          const entry = {
            id: randomUUID(),
            text,
            category,
            source,
            tags: tags || '',
            created_at: now,
            updated_at: now,
            decay_exempt: decay_exempt || 'false',
            importance: importance || '0.5',
            access_count: '0',
            last_accessed: null,
            tier,
          };
          await table.add([entry]);
          return {
            near_duplicate: true,
            similarity,
            new: entry,
            existing: {
              id: topMatch.id,
              text: topMatch.text,
              category: topMatch.category,
              source: topMatch.source,
              tags: topMatch.tags,
              created_at: topMatch.created_at,
              updated_at: topMatch.updated_at,
              decay_exempt: topMatch.decay_exempt || 'false',
              importance: topMatch.importance || '0.5',
              tier: topMatch.tier || assignTier(topMatch.category, topMatch.decay_exempt),
            },
          };
        }
      }
    } catch (err) {
      process.stderr.write(`[memex] dedup check skipped: ${err.message}\n`);
    }
  }

  const entry = {
    id: randomUUID(),
    text,
    category,
    source,
    tags: tags || '',
    created_at: now,
    updated_at: now,
    decay_exempt: decay_exempt || 'false',
    importance: importance || '0.5',
    access_count: '0',
    last_accessed: null,
    tier,
  };
  // LanceDB embedding function auto-populates the vector field from text
  await table.add([entry]);
  recordUsage('store', 1, estimateTokens([text]));

  if (admissionResult && admissionResult.decision === 'warn') {
    return {
      ...entry,
      admission: 'warn',
      admissionReason: admissionResult.reason,
    };
  }

  return entry;
}

/**
 * searchEntries — Retrieve and score entries.
 *
 * Retrieves RETRIEVAL_POOL candidates from vector search, then:
 *   1. Converts cosine distance to similarity
 *   2. Applies Weibull tiered decay blending with access reinforcement
 *   3. Applies length normalization and type bonuses
 *   4. Re-sorts by blended score
 *   5. Applies MMR-inspired diversity filter (defers near-duplicate results)
 *   6. If applyFilters=true: floor + gap filter, caps at maxResults, reorders for LLM
 *   7. If applyFilters=false (reflect mode): returns all results in decay-sorted order, no cap
 *   8. Fires async access tracking update for retrieved entries
 *
 * @param {string} query
 * @param {number} [limit] - ignored in favor of config.maxResults when applyFilters=true
 * @param {string|null} [categoryFilter]
 * @param {boolean} [applyFilters=true]
 */
export async function searchEntries(query, limit = 5, categoryFilter = null, applyFilters = true) {
  await initDB();
  try {
    // Pull a larger pool so post-filtering has candidates to work with
    const poolSize = applyFilters ? RETRIEVAL_POOL : Math.max(limit, RETRIEVAL_POOL);
    let search = table.search(query).distanceType('cosine').limit(poolSize);
    if (categoryFilter) {
      search = search.where(`category = '${categoryFilter.replace(/'/g, "''")}'`);
    }
    const raw = await search.toArray();
    recordUsage('search', 1, estimateTokens([query]));

    // Map rows — computes Weibull decay, length normalization, type bonus
    let results = raw.map(mapRow);

    // Sort descending by blended score
    results.sort((a, b) => (b.blendedScore ?? b.score ?? 0) - (a.blendedScore ?? a.score ?? 0));

    // MMR-inspired diversity filter (Technique 6) — defer near-duplicate results to end
    // Run before gap filtering so deferred entries don't consume gap-filter budget
    results = applyDiversityFilter(results);

    if (applyFilters) {
      // Floor + gap filter, then LLM-optimized reordering
      // Use the caller-supplied limit (already capped at config.maxResults by callers)
      results = applyRelevanceFilter(results, limit);
      results = reorderForLLM(results);
    } else {
      // reflect mode — no floor/gap cuts, but still honour the requested limit
      results = results.slice(0, limit);
    }

    // Fire-and-forget access tracking update (Technique 1)
    if (results.length > 0) {
      const ids = results.map(r => r.id);
      const accessCountMap = new Map(results.map(r => [r.id, parseInt(r.access_count || '0', 10)]));
      updateAccessTracking(ids, accessCountMap);
    }

    // Strip vectors before returning — callers don't need them
    return results.map(({ vector, ...rest }) => rest);
  } catch (err) {
    throw new Error(`Search failed: ${err.message}`);
  }
}

export async function removeEntry(query) {
  await initDB();
  try {
    const results = await table.search(query).distanceType('cosine').limit(1).toArray();
    if (!results || results.length === 0) {
      return null;
    }
    const match = results[0];
    assertUUID(match.id);
    await table.delete(`id = '${match.id}'`);
    return {
      id: match.id,
      text: match.text,
      category: match.category,
    };
  } catch (err) {
    throw new Error(`Remove failed: ${err.message}`);
  }
}

export async function listEntries(category = null, limit = 50) {
  await initDB();
  try {
    let query = table.query();
    if (category) {
      query = query.where(`category = '${category.replace(/'/g, "''")}'`);
    }
    const results = await query.limit(limit).toArray();
    return results.map(r => ({
      id: r.id,
      text: r.text,
      category: r.category,
      source: r.source,
      tags: r.tags,
      created_at: r.created_at,
      updated_at: r.updated_at,
      decay_exempt: r.decay_exempt || 'false',
      importance: r.importance || '0.5',
      access_count: r.access_count || '0',
      last_accessed: r.last_accessed || null,
      tier: r.tier || assignTier(r.category, r.decay_exempt),
    }));
  } catch (err) {
    throw new Error(`List failed: ${err.message}`);
  }
}

export async function getStats() {
  await initDB();
  try {
    const all = await table.query().toArray();
    const total = all.length;
    const byCategory = {};
    for (const row of all) {
      byCategory[row.category] = (byCategory[row.category] || 0) + 1;
    }
    return { total, byCategory };
  } catch (err) {
    throw new Error(`Stats failed: ${err.message}`);
  }
}

export async function exportAll() {
  await initDB();
  try {
    const all = await table.query().toArray();
    // Strip vector field — too large, regenerated on import
    return all.map(({ vector, ...rest }) => rest);
  } catch (err) {
    throw new Error(`Export failed: ${err.message}`);
  }
}

export async function importEntries(entries) {
  await initDB();
  const now = new Date().toISOString();
  let count = 0;
  // Insert in batches of 20 to avoid hammering the embedding API
  const batchSize = 20;
  for (let i = 0; i < entries.length; i += batchSize) {
    const batch = entries.slice(i, i + batchSize).flatMap(e => {
      if (!e.text || typeof e.text !== 'string' || e.text.trim() === '') {
        process.stderr.write(`[memex] Skipping entry with missing text: ${e.id || 'unknown'}\n`);
        return [];
      }
      const category = e.category || 'reference';
      const decayExempt = e.decay_exempt || 'false';
      return [{
        id: e.id || randomUUID(),
        text: e.text,
        category,
        source: e.source || 'migrated',
        tags: e.tags || '',
        created_at: e.created_at || now,
        updated_at: e.updated_at || now,
        // Gracefully handle backups that predate these fields
        decay_exempt: decayExempt,
        importance: e.importance || '0.5',
        access_count: e.access_count || '0',
        last_accessed: e.last_accessed || null,
        tier: e.tier || assignTier(category, decayExempt),
      }];
    });
    await table.add(batch);
    recordUsage('seed', batch.length, estimateTokens(batch.map(b => b.text)));
    count += batch.length;
  }
  return count;
}
