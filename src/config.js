/**
 * config.js — All configurable values in one place.
 *
 * Reads from environment variables with sensible defaults.
 * Users configure via .env file or shell exports.
 */

import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const projectRoot = path.resolve(__dirname, '..');

export const config = {
  // Tool name prefix — tools become {prefix}_ask, {prefix}_memorize, etc.
  // Set MEMEX_PREFIX=reza to get reza_ask, reza_memorize, etc.
  // Leave empty for clean names: ask, memorize, forget, reflect
  // (MCP already namespaces as mcp__memex__ask)
  toolPrefix: process.env.MEMEX_PREFIX || '',

  // Display name used in tool descriptions
  // Set MEMEX_OWNER=Reza to get "Search Reza's knowledge base..."
  ownerName: process.env.MEMEX_OWNER || 'the user',

  // LanceDB data directory — where the vector DB lives
  // Default: ./data/kb (relative to project root)
  dataDir: process.env.MEMEX_DATA_DIR || path.join(projectRoot, 'data', 'kb'),

  // Embedding model — full 3072 dimensions for maximum semantic resolution.
  // Handles everything from 1-sentence preferences to multi-paragraph project context.
  // IMPORTANT: Changing dims requires re-embedding all data. Delete the LanceDB
  // table directory and re-run seed. Mixing dims in the same table will cause errors.
  embeddingModel: process.env.MEMEX_EMBEDDING_MODEL || 'text-embedding-3-large',
  embeddingDims: parseInt(process.env.MEMEX_EMBEDDING_DIMS || '3072', 10),

  // Table name inside LanceDB
  tableName: process.env.MEMEX_TABLE || 'knowledge',

  // Export path for backup
  exportPath: process.env.MEMEX_EXPORT_PATH || path.join(projectRoot, 'data', 'backup.json'),

  // Temporal decay scoring — Weibull tiered decay with access reinforcement.
  // See computeDecayScore() in db.js for the full formula.
  // alpha: weight of semantic similarity vs freshness (0-1).
  // halfLife: base half-life in days before importance/access modulation.
  decayAlpha: parseFloat(process.env.MEMEX_DECAY_ALPHA || '0.7'),
  decayHalfLife: parseInt(process.env.MEMEX_DECAY_HALF_LIFE || '90', 10),

  // Relevance filtering
  // minScore: floor threshold — drop results below this cosine similarity
  // gapRatio: drop results more than this ratio below the top score (e.g. 0.75 = drop if < top * 0.75)
  // maxResults: hard cap after all filtering
  // retrievalPool: candidates pulled from vector search before post-filtering; must be > maxResults
  minScore: parseFloat(process.env.MEMEX_MIN_SCORE || '0.35'),
  gapRatio: parseFloat(process.env.MEMEX_GAP_RATIO || '0.75'),
  maxResults: parseInt(process.env.MEMEX_MAX_RESULTS || '5', 10),
  retrievalPool: parseInt(process.env.MEMEX_RETRIEVAL_POOL || '20', 10),

  // Insert-time deduplication thresholds (cosine similarity)
  // dedupExact: skip insert entirely, return existing with duplicate:true
  // dedupNear: return both with near_duplicate:true, let caller decide
  dedupExact: parseFloat(process.env.MEMEX_DEDUP_EXACT || '0.97'),
  dedupNear: parseFloat(process.env.MEMEX_DEDUP_NEAR || '0.94'),

  // Tiered Weibull decay parameters
  // mu: importance modulation strength — higher = importance has more effect on effective half-life
  // betaCore: shape for core tier (personal/preference) — sub-exponential, plateaus after initial drop
  // betaWorking: shape for working tier (decision/project/routing) — standard exponential
  // betaPeripheral: shape for peripheral tier (reference/unknown) — super-exponential, accelerates decay
  // accessDecayDays: window (days) over which access recency is weighted
  // maxHalfLifeMultiplier: cap on how much access reinforcement can extend half-life
  weibullMu: parseFloat(process.env.MEMEX_WEIBULL_MU || '1.5'),
  weibullBetaCore: parseFloat(process.env.MEMEX_WEIBULL_BETA_CORE || '0.8'),
  weibullBetaWorking: parseFloat(process.env.MEMEX_WEIBULL_BETA_WORKING || '1.0'),
  weibullBetaPeripheral: parseFloat(process.env.MEMEX_WEIBULL_BETA_PERIPHERAL || '1.3'),
  accessDecayDays: parseFloat(process.env.MEMEX_ACCESS_DECAY_DAYS || '30'),
  maxHalfLifeMultiplier: parseFloat(process.env.MEMEX_MAX_HALF_LIFE_MULTIPLIER || '3.0'),

  // Admission control thresholds
  // admissionThresholdAdmit: score at or above this → store normally
  // admissionThresholdReject: score below this → reject with explanation
  // Between the two → store but flag as low-confidence
  admissionThresholdAdmit: parseFloat(process.env.MEMEX_ADMISSION_ADMIT || '0.55'),
  admissionThresholdReject: parseFloat(process.env.MEMEX_ADMISSION_REJECT || '0.35'),

};
