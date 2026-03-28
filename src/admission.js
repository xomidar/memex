/**
 * admission.js — Multi-factor admission control gate for memory entries.
 *
 * Before any entry is stored, scoreAdmission() evaluates it on three axes:
 *   1. Type prior    (50%) — how inherently valuable is this category?
 *   2. Novelty       (30%) — how different is it from what's already stored?
 *   3. Content quality (20%) — is the text substantive and non-generic?
 *
 * Final score → three outcomes:
 *   >= admissionThresholdAdmit  : ADMIT  — store normally
 *   >= admissionThresholdReject : WARN   — store, flag low-confidence
 *   <  admissionThresholdReject : REJECT — don't store, explain why
 *
 * Sourced from patterns in CortexReach/admission-control.ts and
 * mcp-memory-service/memory-scorer.js.
 */

import { config } from './config.js';

// How inherently valuable each category is as a memory type.
// Higher = more likely to be worth storing regardless of other factors.
const TYPE_PRIORS = {
  personal:   0.95,
  preference: 0.90,
  decision:   0.85,
  project:    0.75,
  routing:    0.70,
  reference:  0.60,
};

// Generic filler phrases that inflate length without adding information.
const GENERIC_PHRASES = [
  'it is important to note',
  'it should be noted',
  'it is worth noting',
  'as mentioned above',
  'as previously stated',
  'in conclusion',
  'to summarize',
  'needless to say',
  'at the end of the day',
  'it goes without saying',
];

/**
 * Score content quality of a text string.
 * Returns a value between ~0.1 and 0.8.
 *
 * @param {string} text
 * @returns {number}
 */
function scoreContentQuality(text) {
  let quality = 0.5;

  const len = text.length;

  if (len < 10) {
    // Trivially short — almost certainly noise
    quality -= 0.5;
  } else if (len < 20) {
    // Too short to convey meaningful information
    quality -= 0.3;
  } else if (len > 50) {
    // Substantive enough to hold real signal
    quality += 0.1;
  }

  // Lexical diversity: unique word ratio above 0.7 → rich vocabulary, not repetitive
  const words = text.toLowerCase().match(/\b\w+\b/g) || [];
  if (words.length > 0) {
    const uniqueRatio = new Set(words).size / words.length;
    if (uniqueRatio > 0.7) {
      quality += 0.2;
    }
  }

  // Penalize generic filler phrases
  const lowerText = text.toLowerCase();
  for (const phrase of GENERIC_PHRASES) {
    if (lowerText.includes(phrase)) {
      quality -= 0.1;
      break; // one penalty per entry, not cumulative
    }
  }

  // Clamp to [0.0, 1.0]
  return Math.max(0.0, Math.min(1.0, quality));
}

/**
 * Score how novel the new entry is relative to existing entries.
 * Novelty = 1 - avgCosineSimilarity across the top-K matches in the DB.
 * Using an average rather than max prevents a single near-duplicate from
 * dominating the signal when other matches are dissimilar.
 *
 * When the table is empty (no candidates), novelty is 0.5 — neutral score
 * to prevent empty-table bootstrapping from bypassing quality gates.
 *
 * @param {Array} dedupCandidates - results from a pre-search against the table (may be empty)
 * @returns {number} 0.0 to 1.0
 */
function scoreNovelty(dedupCandidates) {
  if (!dedupCandidates || dedupCandidates.length === 0) {
    // No comparison data — neutral, not maximum. Prevents garbage from passing
    // on novelty alone when the KB is empty or the search failed.
    return 0.5;
  }
  const similarities = dedupCandidates.map(m =>
    m._distance !== undefined ? (1 - m._distance / 2) : 0
  );
  const avgSimilarity = similarities.reduce((a, b) => a + b, 0) / similarities.length;
  return Math.max(0.0, 1.0 - avgSimilarity);
}

/**
 * Compute the full admission score for a candidate entry.
 *
 * @param {string} text           - the content to be stored
 * @param {string} category       - memory category (personal/preference/decision/...)
 * @param {Array}  dedupCandidates - raw LanceDB results from pre-search (may be empty array)
 * @returns {{ score: number, typePrior: number, novelty: number, contentQuality: number }}
 */
export function computeAdmissionScore(text, category, dedupCandidates) {
  const typePrior = TYPE_PRIORS[category] ?? TYPE_PRIORS.reference;
  const novelty = scoreNovelty(dedupCandidates);
  const contentQuality = scoreContentQuality(text);

  const score = (typePrior * 0.50) + (novelty * 0.30) + (contentQuality * 0.20);

  return { score, typePrior, novelty, contentQuality };
}

/**
 * Evaluate admission decision for an entry.
 *
 * @param {string} text
 * @param {string} category
 * @param {Array}  dedupCandidates
 * @returns {{
 *   decision: 'admit' | 'warn' | 'reject',
 *   score: number,
 *   reason: string,
 *   components: { typePrior: number, novelty: number, contentQuality: number }
 * }}
 */
export function evaluateAdmission(text, category, dedupCandidates) {
  const { score, typePrior, novelty, contentQuality } = computeAdmissionScore(text, category, dedupCandidates);

  const components = { typePrior, novelty, contentQuality };

  if (score >= config.admissionThresholdAdmit) {
    return {
      decision: 'admit',
      score,
      reason: null,
      components,
    };
  }

  if (score >= config.admissionThresholdReject) {
    // Build a human-readable reason for the warning
    const flags = [];
    if (novelty < 0.25) flags.push('highly similar to existing entry');
    if (contentQuality < 0.4) flags.push('low content quality');
    if (typePrior < 0.70) flags.push('low-priority category');
    const reason = flags.length > 0
      ? `Low admission confidence (score: ${score.toFixed(3)}): ${flags.join(', ')}`
      : `Low admission confidence (score: ${score.toFixed(3)})`;
    return {
      decision: 'warn',
      score,
      reason,
      components,
    };
  }

  // Rejection — explain why clearly
  const flags = [];
  if (novelty < 0.20) flags.push(`near-duplicate of existing entry (similarity: ${(1 - novelty).toFixed(3)})`);
  if (contentQuality < 0.3) flags.push('content too short or low quality');
  if (typePrior < 0.65) flags.push(`low-priority category "${category}"`);
  const reason = `Admission rejected (score: ${score.toFixed(3)}): ${flags.join('; ') || 'combined score too low'}`;

  return {
    decision: 'reject',
    score,
    reason,
    components,
  };
}
