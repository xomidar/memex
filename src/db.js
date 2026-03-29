import * as lancedb from '@lancedb/lancedb';
import { LanceSchema, getRegistry } from '@lancedb/lancedb/embedding';
import '@lancedb/lancedb/embedding/openai';  // registers OpenAI in the embedding registry
import { Utf8 } from 'apache-arrow';
import { randomUUID, createHash } from 'crypto';
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
    // content_type: "entry" (default) | "chunk" — distinguishes standalone entries from doc sections
    content_type: new Utf8(),
    // chunk_index: "0", "1", "2"... for ordering within a document; null for standalone entries
    chunk_index: new Utf8(),
    // superseded_at: ISO timestamp — set when chunk is replaced by resync; null = active
    superseded_at: new Utf8(),
    // context: LLM-generated contextual prefix — prepended to text for richer embeddings
    // (Contextual Retrieval, Anthropic 2024). Stored separately so recall shows clean text.
    context: new Utf8(),
    // content_hash: 16-char SHA-256 prefix — used for resync change detection
    content_hash: new Utf8(),
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
// Chunk helpers
// ---------------------------------------------------------------------------

/**
 * Extract the doc name from a comma-separated tags string.
 * Looks for "doc:<name>" tag. Returns null if not found or tags is empty/null.
 *
 * @param {string|null|undefined} tags - comma-separated tag string, e.g. "project,doc:career-plan"
 * @returns {string|null} doc name without "doc:" prefix, or null
 */
export function extractDocTag(tags) {
  if (!tags || typeof tags !== 'string') return null;
  const parts = tags.split(',');
  for (const part of parts) {
    const trimmed = part.trim();
    if (trimmed.startsWith('doc:')) {
      const name = trimmed.slice(4).trim();
      return name.length > 0 ? name : null;
    }
  }
  return null;
}

/**
 * Compute a 16-character SHA-256 hash prefix for chunk dedup.
 * Sufficient for change detection at this scale — collision probability negligible.
 *
 * @param {string} text
 * @returns {string} 16-char hex string
 */
export function computeContentHash(text) {
  return createHash('sha256').update(text).digest('hex').slice(0, 16);
}

/**
 * Auto-merge hint: when 3+ chunks from the same doc appear in results,
 * annotate them with _autoMergeHint so the caller can suggest recall().
 * This avoids an extra DB round-trip — we flag but don't fetch the parent here.
 *
 * @param {Array} results - mapped rows from searchEntries
 * @returns {Array} same array, with _autoMergeHint added where applicable
 */
function autoMergeChunks(results) {
  // Count active chunks per doc_name among results
  const docCounts = {};
  for (const r of results) {
    const docTag = extractDocTag(r.tags);
    if (docTag && r.content_type === 'chunk' && !r.superseded_at) {
      docCounts[docTag] = (docCounts[docTag] || 0) + 1;
    }
  }

  // For docs with 3+ chunk hits, annotate all their chunks unless parent already present
  for (const [docName, count] of Object.entries(docCounts)) {
    if (count >= 3) {
      const hasParent = results.some(r =>
        extractDocTag(r.tags) === docName && r.chunk_index === '0'
      );
      if (!hasParent) {
        for (const r of results) {
          if (extractDocTag(r.tags) === docName && r.content_type === 'chunk') {
            r._autoMergeHint = docName;
          }
        }
      }
    }
  }

  return results;
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
    // Skip length penalty for chunks — they're pre-sized by the chunking strategy
    // and longer chunks contain MORE useful information, not less.
    const lengthPenalty = (r.content_type === 'chunk') ? 1.0 : computeLengthPenalty(r.text ? r.text.length : 0);
    const normalizedScore = rawBlended * lengthPenalty;

    // Type bonus (Technique 4) — additive, small, breaks ties
    blendedScore = normalizedScore + getTypeBonus(r.category);

    // Parent summary demotion (Technique 13) — when searching, prefer specific sections
    // over broad parent summaries. Parent summaries (chunk_index=0) are routing entries
    // for recall, not the best match for precise queries. Small penalty so they yield
    // to children that have higher cosine similarity on specific terms.
    if (r.content_type === 'chunk' && r.chunk_index === '0') {
      blendedScore *= 0.90; // 10% demotion for parent summaries in search
    }
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
    // New chunk fields — default gracefully for pre-2.0 rows
    content_type: r.content_type || 'entry',
    chunk_index: r.chunk_index != null ? r.chunk_index : null,
    superseded_at: r.superseded_at || null,
    content_hash: r.content_hash || null,
    context: r.context || null,
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
 * Pass skipAdmission=true (or options.skipAdmission=true) to bypass the gate.
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
 * @param {string|boolean|object} [decay_exempt_or_options='false'] - legacy string/bool OR options object
 * @param {string} [importance='0.5']
 * @param {boolean} [skipAdmission=false]
 *
 * When called with an options object as 5th argument:
 * @param {object} options
 * @param {string}  [options.decay_exempt='false']
 * @param {string}  [options.importance='0.5']
 * @param {boolean} [options.skipAdmission=false]
 * @param {string}  [options.content_type='entry']
 * @param {string}  [options.chunk_index=null]
 * @param {string}  [options.content_hash=null]
 */
export async function addEntry(
  text,
  category,
  source = 'user_explicit',
  tags = '',
  decay_exempt_or_options = 'false',
  importance = '0.5',
  skipAdmission = false,
) {
  // Detect options-object call pattern (used by resyncDocument and chunk memorize handler)
  let decay_exempt, options_content_type, options_chunk_index, options_content_hash, options_context;
  if (decay_exempt_or_options && typeof decay_exempt_or_options === 'object') {
    const opts = decay_exempt_or_options;
    decay_exempt = opts.decay_exempt || 'false';
    importance = opts.importance || importance || '0.5';
    skipAdmission = opts.skipAdmission === true;
    options_content_type = opts.content_type || 'entry';
    options_chunk_index = opts.chunk_index != null ? String(opts.chunk_index) : null;
    options_content_hash = opts.content_hash || null;
    options_context = opts.context || null;
  } else {
    decay_exempt = decay_exempt_or_options || 'false';
    options_content_type = 'entry';
    options_chunk_index = null;
    options_content_hash = null;
    options_context = null;
  }

  // Contextual Retrieval (Anthropic 2024): if context is provided, prepend it to text
  // for richer embeddings. The original text is preserved via the context column.
  // Harvey (or any calling LLM) generates the context — zero extra API cost.
  if (options_context && options_context.trim()) {
    text = `${options_context.trim()}\n\n${text}`;
  }
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
          content_type: options_content_type,
          chunk_index: options_chunk_index,
          superseded_at: null,
          content_hash: options_content_hash,
          context: options_context,
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
            content_type: options_content_type,
            chunk_index: options_chunk_index,
            superseded_at: null,
            content_hash: options_content_hash,
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
    content_type: options_content_type,
    chunk_index: options_chunk_index,
    superseded_at: null,
    content_hash: options_content_hash,
    context: options_context,
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

    // Exclude superseded chunks — they've been replaced by a resync
    results = results.filter(r => !r.superseded_at);

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

    // Auto-merge hint: annotate chunks when 3+ from same doc hit (applyFilters only)
    // In reflect mode we show everything — no need to hint about recall()
    if (applyFilters) {
      results = autoMergeChunks(results);
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

    // Cascade delete: if this is a document parent chunk, delete all sibling chunks too
    const matchContentType = match.content_type || 'entry';
    const matchChunkIndex = match.chunk_index != null ? match.chunk_index : null;
    if (matchContentType === 'chunk' && matchChunkIndex === '0') {
      const docTag = extractDocTag(match.tags);
      if (docTag) {
        const allChunks = await listByDocTag(docTag);
        for (const chunk of allChunks) {
          assertUUID(chunk.id);
          await table.delete(`id = '${chunk.id}'`);
        }
        return {
          id: match.id,
          text: match.text,
          category: match.category,
          cascadeDeleted: allChunks.length,
          docName: docTag,
        };
      }
    }

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

/**
 * List all active (non-superseded) chunks belonging to a document.
 * Used by removeEntry cascade, resyncDocument, and the recall tool.
 *
 * @param {string} docName - logical document name (without "doc:" prefix)
 * @returns {Promise<Array<object>>} array of mapped row objects
 */
export async function listByDocTag(docName) {
  const { table: t } = await initDB();
  try {
    // LanceDB doesn't support LIKE on string columns efficiently — fetch and filter in JS
    // No hard limit — must scan full table to avoid silently missing chunks
    const all = await t.query().toArray();
    return all
      .filter(r => {
        const tag = extractDocTag(r.tags);
        return tag === docName && !(r.superseded_at);
      })
      .map(r => ({
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
        tier: r.tier || assignTier(r.category, r.decay_exempt || 'false'),
        content_type: r.content_type || 'entry',
        chunk_index: r.chunk_index != null ? r.chunk_index : null,
        superseded_at: r.superseded_at || null,
        content_hash: r.content_hash || null,
    context: r.context || null,
      }));
  } catch (err) {
    throw new Error(`listByDocTag failed: ${err.message}`);
  }
}

/**
 * Safely replace all chunks for a document using insert-then-supersede.
 * New chunks are written first; old ones are marked superseded only after all
 * inserts succeed. If the embedding API fails mid-batch, old data is preserved.
 *
 * Unchanged chunks (same content_hash at same chunk_index) are skipped —
 * their access patterns and creation timestamps are preserved.
 *
 * @param {string} docName - logical document name (without "doc:" prefix)
 * @param {Array<{
 *   text: string,
 *   category?: string,
 *   tags?: string,
 *   importance?: string,
 *   decay_exempt?: string
 * }>} chunks - ordered array of chunk data; chunks[0] is the parent summary
 * @param {object} [options={}]
 * @param {string} [options.source='resync']
 * @param {string} [options.importance='0.7']
 * @param {string} [options.decay_exempt='false']
 * @returns {Promise<{
 *   success: boolean,
 *   docName: string,
 *   total?: number,
 *   changed?: number,
 *   unchanged?: number,
 *   superseded?: number,
 *   error?: string,
 *   failed?: number,
 *   inserted?: number
 * }>}
 */
export async function resyncDocument(docName, chunks, options = {}) {
  await initDB();

  // Step 1: Get existing active chunks for this doc
  const existing = await listByDocTag(docName);

  // Step 2: Hash new chunks, compare with existing at same position
  const newChunks = chunks.map((chunk, i) => {
    const hash = computeContentHash(chunk.text);
    const existingMatch = existing.find(
      e => e.content_hash === hash && e.chunk_index === String(i)
    );
    return { ...chunk, hash, unchanged: !!existingMatch, existingId: existingMatch?.id };
  });

  // Step 3: Insert only CHANGED chunks (skip unchanged to preserve their access history)
  const inserted = [];
  for (let i = 0; i < newChunks.length; i++) {
    const nc = newChunks[i];

    if (nc.unchanged) {
      inserted.push({ id: nc.existingId, skipped: true });
      continue;
    }

    // Ensure doc tag is present in tags
    const baseTags = nc.tags || '';
    const docTag = `doc:${docName}`;
    const tags = baseTags.includes(docTag)
      ? baseTags
      : (baseTags ? `${baseTags},${docTag}` : docTag);

    // Parent summary (chunk_index 0) gets higher importance floor
    const chunkImportance = i === 0
      ? String(Math.max(parseFloat(nc.importance || options.importance || '0.7'), 0.7))
      : (nc.importance || options.importance || '0.7');

    let entryText = nc.text;
    // Prepend context header for non-parent chunks (improves embedding relevance)
    if (i > 0) {
      const firstLine = nc.text.split('\n')[0].replace(/^#+\s*/, '').trim();
      entryText = `[Document: ${docName} | Section: ${firstLine}]\n${nc.text}`;
    }

    const entry = await addEntry(
      entryText,
      nc.category || 'project',
      options.source || 'resync',
      tags,
      {
        content_type: 'chunk',
        chunk_index: String(i),
        importance: chunkImportance,
        decay_exempt: nc.decay_exempt || options.decay_exempt || 'false',
        skipAdmission: true,
        content_hash: nc.hash,
      },
    );
    inserted.push(entry);
  }

  // Step 4: Normalize inserted results and verify
  // addEntry can return { duplicate: true, ... }, { near_duplicate: true, new: {...} }, or a normal entry.
  // Normalize so we always have a usable id for Step 5 comparison.
  const normalizedInserts = inserted.map(e => {
    if (!e) return null;
    if (e.skipped) return e; // unchanged chunk, has .id
    if (e.duplicate) return { id: e.id, skipped: true }; // exact dup = treat as kept
    if (e.near_duplicate) return e.new || e; // near-dup: use the newly created entry
    return e; // normal insert
  });

  const failedInserts = normalizedInserts.filter(e => !e || (e.rejected && !e.skipped));
  if (failedInserts.length > 0) {
    return {
      success: false,
      error: 'Some chunks failed to insert',
      failed: failedInserts.length,
      inserted: inserted.length - failedInserts.length,
    };
  }

  // Step 5: Supersede old chunks that were NOT kept unchanged or matched as duplicates
  const now = new Date().toISOString();
  let supersededCount = 0;
  const keptIds = new Set(normalizedInserts.filter(e => e?.skipped && e?.id).map(e => e.id));
  for (const old of existing) {
    const wasKept = keptIds.has(old.id);
    if (!wasKept) {
      assertUUID(old.id);
      await table.update({
        where: `id = '${old.id}'`,
        values: { superseded_at: now },
      });
      supersededCount++;
    }
  }

  const unchangedCount = newChunks.filter(nc => nc.unchanged).length;
  const changedCount = newChunks.length - unchangedCount;

  return {
    success: true,
    docName,
    total: newChunks.length,
    changed: changedCount,
    unchanged: unchangedCount,
    superseded: supersededCount,
  };
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
      content_type: r.content_type || 'entry',
      chunk_index: r.chunk_index != null ? r.chunk_index : null,
      superseded_at: r.superseded_at || null,
      content_hash: r.content_hash || null,
    context: r.context || null,
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
        // New chunk fields — default gracefully for pre-2.0 backups
        content_type: e.content_type || 'entry',
        chunk_index: e.chunk_index != null ? e.chunk_index : null,
        superseded_at: e.superseded_at || null,
        content_hash: e.content_hash || null,
        context: e.context || null,
      }];
    });
    await table.add(batch);
    recordUsage('seed', batch.length, estimateTokens(batch.map(b => b.text)));
    count += batch.length;
  }
  return count;
}
