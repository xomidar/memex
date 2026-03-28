import { searchEntries, addEntry, removeEntry, listEntries } from './db.js';
import { config } from './config.js';

const prefix = config.toolPrefix;
const p = prefix ? `${prefix}_` : '';  // "reza_" or "" — no dangling underscore
const owner = config.ownerName;

/**
 * Format a date string as YYYY-MM-DD for display.
 */
function formatDate(isoString) {
  if (!isoString) return 'unknown';
  return isoString.split('T')[0];
}

/**
 * Truncate text at a sentence boundary at or before maxChars.
 * Looks for '.', '!', '?', '\n' before the limit.
 * Appends '[...]' if truncated.
 *
 * @param {string} text
 * @param {number} maxChars
 * @returns {string}
 */
function truncateAtSentenceBoundary(text, maxChars) {
  if (text.length <= maxChars) return text;

  // Look for the last sentence-ending punctuation before maxChars
  const slice = text.slice(0, maxChars);
  const lastBoundary = Math.max(
    slice.lastIndexOf('.'),
    slice.lastIndexOf('!'),
    slice.lastIndexOf('?'),
    slice.lastIndexOf('\n'),
  );

  if (lastBoundary > 0) {
    return text.slice(0, lastBoundary + 1) + ' [...]';
  }

  // No sentence boundary found — truncate hard at limit
  return slice + ' [...]';
}

/**
 * Format search results for LLM consumption.
 * Adaptive context truncation (Technique 5): fewer results → more chars per entry.
 *
 * Truncation limits:
 *   >= 5 results → 300 chars each
 *   >= 3 results → 500 chars each
 *   < 3 results  → 800 chars each
 *
 * Format:
 *   [Memory #1 — Score: 0.91 — Category: preference — Stored: 2026-02-14]
 *   Reza prefers responses under 400 words...
 */
function formatResultsForLLM(results) {
  // Adaptive truncation limit based on result count (Technique 5)
  let maxChars;
  if (results.length >= 5) {
    maxChars = 300;
  } else if (results.length >= 3) {
    maxChars = 500;
  } else {
    maxChars = 800;
  }

  return results
    .map((r, i) => {
      const score = r.blendedScore !== null ? r.blendedScore.toFixed(2)
        : r.score !== null ? r.score.toFixed(2)
        : 'n/a';
      const stored = formatDate(r.created_at);
      const header = `[Memory #${i + 1} — Score: ${score} — Category: ${r.category} — Stored: ${stored}]`;
      const tags = r.tags ? `\n  tags: ${r.tags}` : '';
      const body = truncateAtSentenceBoundary(r.text, maxChars);
      return `${header}\n${body}${tags}`;
    })
    .join('\n\n');
}

export const tools = [
  {
    name: `${p}ask`,
    definition: {
      name: `${p}ask`,
      description:
        `Search ${owner}'s knowledge base for relevant context. Use this when you need information about ${owner}'s preferences, past decisions, personal details, project context, or routing precedents.`,
      inputSchema: {
        type: 'object',
        properties: {
          query: {
            type: 'string',
            description: 'What to search for',
          },
          category: {
            type: 'string',
            description:
              'Optional category filter: personal | preference | decision | project | routing | reference',
          },
          limit: {
            type: 'number',
            description: `Max results to return (default ${config.maxResults}, hard capped by config)`,
          },
        },
        required: ['query'],
      },
    },
    async handler({ query, category, limit }) {
      try {
        // ask: full pipeline — floor filter + gap filter + LLM reordering
        // limit is honoured up to config.maxResults hard cap
        const effectiveLimit = Math.min(limit || config.maxResults, config.maxResults);
        const results = await searchEntries(query, effectiveLimit, category || null, true);
        if (results.length === 0) {
          return { content: [{ type: 'text', text: 'No matching entries found.' }] };
        }
        return { content: [{ type: 'text', text: formatResultsForLLM(results) }] };
      } catch (err) {
        return { content: [{ type: 'text', text: `Error: ${err.message}` }], isError: true };
      }
    },
  },

  {
    name: `${p}memorize`,
    definition: {
      name: `${p}memorize`,
      description:
        `Store information in ${owner}'s knowledge base. Use this when ${owner} asks you to remember something, or when you learn something important about ${owner}'s preferences, decisions, or context.`,
      inputSchema: {
        type: 'object',
        properties: {
          text: {
            type: 'string',
            description: 'The information to store',
          },
          category: {
            type: 'string',
            description:
              'Category: personal | preference | decision | project | routing | reference',
          },
          tags: {
            type: 'string',
            description: 'Optional comma-separated tags for filtering (e.g. "claude,routing,agents")',
          },
          decay_exempt: {
            type: 'string',
            description:
              'Set to "true" for biographical facts, permanent preferences, or knowledge that should not lose relevance over time. Defaults to "false".',
          },
          importance: {
            type: 'string',
            description:
              'Importance level from "0.0" to "1.0". Higher importance extends the effective memory half-life via Weibull decay modulation. Defaults to "0.5".',
          },
        },
        required: ['text', 'category'],
      },
    },
    async handler({ text, category, tags, decay_exempt, importance }) {
      try {
        const result = await addEntry(
          text,
          category,
          'user_explicit',
          tags || '',
          decay_exempt || 'false',
          importance || '0.5',
          false, // skipAdmission = false for explicit user storage
        );

        // Admission rejected — explain why and don't store
        if (result.rejected) {
          return {
            content: [
              {
                type: 'text',
                text: `Not stored — admission rejected.\nReason: ${result.reason}\nAdmission score: ${result.score.toFixed(3)}\nComponents: type_prior=${result.components.typePrior.toFixed(3)}, novelty=${result.components.novelty.toFixed(3)}, quality=${result.components.contentQuality.toFixed(3)}`,
              },
            ],
          };
        }

        // Exact duplicate — surface it clearly
        if (result.duplicate) {
          return {
            content: [
              {
                type: 'text',
                text: `Duplicate detected (similarity: ${result.similarity.toFixed(3)}) — not stored.\nExisting entry ID: ${result.id}\nText: ${result.text}`,
              },
            ],
          };
        }

        // Near duplicate — stored but flagged
        if (result.near_duplicate) {
          let msg = `Near-duplicate detected (similarity: ${result.similarity.toFixed(3)}) — stored anyway.\nNew ID: ${result.new.id}\nExisting ID: ${result.existing.id}\nExisting text: ${result.existing.text}`;
          if (result.admission === 'warn') {
            msg += `\nAdmission warning: ${result.admissionReason}`;
          }
          return { content: [{ type: 'text', text: msg }] };
        }

        // Low-confidence admission warning
        if (result.admission === 'warn') {
          return {
            content: [
              {
                type: 'text',
                text: `Stored with low confidence.\nAdmission warning: ${result.admissionReason}\nID: ${result.id}\nCategory: ${result.category}\nTier: ${result.tier}\nDecay exempt: ${result.decay_exempt}\nImportance: ${result.importance}\nText: ${result.text}`,
              },
            ],
          };
        }

        return {
          content: [
            {
              type: 'text',
              text: `Stored. ID: ${result.id}\nCategory: ${result.category}\nTier: ${result.tier}\nDecay exempt: ${result.decay_exempt}\nImportance: ${result.importance}\nText: ${result.text}`,
            },
          ],
        };
      } catch (err) {
        return { content: [{ type: 'text', text: `Error: ${err.message}` }], isError: true };
      }
    },
  },

  {
    name: `${p}forget`,
    definition: {
      name: `${p}forget`,
      description:
        `Remove an entry from ${owner}'s knowledge base. Finds the closest semantic match to the query and removes it.`,
      inputSchema: {
        type: 'object',
        properties: {
          query: {
            type: 'string',
            description: 'Describe what to remove — closest match will be deleted',
          },
        },
        required: ['query'],
      },
    },
    async handler({ query }) {
      try {
        const removed = await removeEntry(query);
        if (!removed) {
          return { content: [{ type: 'text', text: 'No close match found. Nothing removed.' }] };
        }
        return {
          content: [
            {
              type: 'text',
              text: `Removed entry:\nID: ${removed.id}\nCategory: ${removed.category}\nText: ${removed.text}`,
            },
          ],
        };
      } catch (err) {
        return { content: [{ type: 'text', text: `Error: ${err.message}` }], isError: true };
      }
    },
  },

  {
    name: `${p}reflect`,
    definition: {
      name: `${p}reflect`,
      description:
        `Show what's stored in ${owner}'s knowledge base about a topic. Use this for transparency — lets ${owner} audit what you know about any subject.`,
      inputSchema: {
        type: 'object',
        properties: {
          topic: {
            type: 'string',
            description: 'Topic to reflect on',
          },
          category: {
            type: 'string',
            description:
              'Optional category filter: personal | preference | decision | project | routing | reference',
          },
          limit: {
            type: 'number',
            description: 'Max results to return (default 10)',
          },
        },
        required: ['topic'],
      },
    },
    async handler({ topic, category, limit = 10 }) {
      try {
        // reflect: bypass gap filtering — show everything semantically relevant.
        // Still applies temporal decay for ordering (recent stuff surfaces first),
        // but no floor/gap cuts and no maxResults cap.
        const results = await searchEntries(topic, limit, category || null, false);
        if (results.length === 0) {
          return { content: [{ type: 'text', text: `Nothing found about "${topic}".` }] };
        }
        const formatted = formatResultsForLLM(results);
        return {
          content: [
            {
              type: 'text',
              text: `Knowledge about "${topic}" (${results.length} entries):\n\n${formatted}`,
            },
          ],
        };
      } catch (err) {
        return { content: [{ type: 'text', text: `Error: ${err.message}` }], isError: true };
      }
    },
  },
];
