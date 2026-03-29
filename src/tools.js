import { searchEntries, addEntry, removeEntry, listByDocTag, extractDocTag, computeContentHash } from './db.js';
import { addTodo, listTodos, completeTodo, removeTodo, getTodoStats } from './rdms.js';
import { config } from './config.js';

const prefix = config.toolPrefix;
const p = prefix ? `${prefix}_` : '';  // "reza_" or "" — no dangling underscore
const owner = config.ownerName;

/**
 * Format a date string as YYYY-MM-DD for display.
 *
 * @param {string|null} isoString
 * @returns {string}
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
 *
 * @param {Array} results
 * @returns {string}
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
  // ---------------------------------------------------------------------------
  // ask — semantic search
  // ---------------------------------------------------------------------------
  {
    name: `${p}ask`,
    definition: {
      name: `${p}ask`,
      description:
        `Search and find information in ${owner}'s knowledge base. Use this when you need to look up ${owner}'s preferences, past decisions, personal details, project context, or routing precedents. For retrieving a complete document or plan use recall instead.`,
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

        // Build the main response text
        let text = formatResultsForLLM(results);

        // Auto-merge hint: if any result carries _autoMergeHint, append recall suggestion
        const hintedDocs = [...new Set(results.filter(r => r._autoMergeHint).map(r => r._autoMergeHint))];
        if (hintedDocs.length > 0) {
          const hints = hintedDocs.map(d => `recall('${d}')`).join(', ');
          text += `\n\n[Note: Multiple sections from document${hintedDocs.length > 1 ? 's' : ''} ${hintedDocs.map(d => `'${d}'`).join(', ')} matched. Use ${hints} to see the full document${hintedDocs.length > 1 ? 's' : ''}.]`;
        }

        return { content: [{ type: 'text', text }] };
      } catch (err) {
        return { content: [{ type: 'text', text: `Error: ${err.message}` }], isError: true };
      }
    },
  },

  // ---------------------------------------------------------------------------
  // memorize — store knowledge
  // ---------------------------------------------------------------------------
  {
    name: `${p}memorize`,
    definition: {
      name: `${p}memorize`,
      description:
        `Store knowledge in ${owner}'s knowledge base. Use this when ${owner} asks you to remember something, or when you learn something important about ${owner}'s preferences, decisions, or context.`,
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
          content_type: {
            type: 'string',
            enum: ['entry', 'chunk'],
            description:
              '"entry" for standalone knowledge (default). "chunk" for document sections stored via ingest workflow.',
          },
          chunk_index: {
            type: 'string',
            description:
              'Position within document ("0" = parent summary, "1"+ = sections). Only used with content_type=chunk.',
          },
          doc_name: {
            type: 'string',
            description:
              'Logical document name. Automatically added as doc:<name> tag. Only used with content_type=chunk.',
          },
          context: {
            type: 'string',
            description:
              'Contextual Retrieval prefix — a 1-3 sentence description of what this entry is about, written by the calling agent who has full document context. Prepended to text for richer embeddings but stored separately so recall shows clean content. Example: "This describes TU Darmstadt\'s ELITE-tier AI & ML master\'s program with IELTS 7.0 requirement and MED-HIGH admission chance."',
          },
        },
        required: ['text', 'category'],
      },
    },
    async handler({ text, category, tags, decay_exempt, importance, content_type, chunk_index, doc_name, context }) {
      try {
        let finalText = text;
        let finalTags = tags || '';
        let skipAdmission = false;
        let finalImportance = importance || '0.5';
        let finalContentType = content_type || 'entry';
        let finalChunkIndex = chunk_index != null ? String(chunk_index) : null;
        let contentHash = null;

        // Chunk handling
        if (content_type === 'chunk') {
          // doc_name is required for chunks — without it, recall can never find them
          if (!doc_name) {
            return {
              content: [{ type: 'text', text: 'doc_name is required when content_type is "chunk". Without it, the chunk becomes unretrievable.' }],
              isError: true,
            };
          }

          skipAdmission = true;

          // Ensure doc tag in tags
          if (doc_name) {
            const docTag = `doc:${doc_name}`;
            finalTags = finalTags.includes(docTag)
              ? finalTags
              : (finalTags ? `${finalTags},${docTag}` : docTag);
          }

          // Parent summary gets importance floor of 0.7
          if (chunk_index === '0') {
            finalImportance = String(Math.max(parseFloat(importance || '0.5'), 0.7));
          }

          // Prepend context header for non-parent chunks
          if (chunk_index !== '0' && doc_name) {
            const firstLine = text.split('\n')[0].replace(/^#+\s*/, '').trim();
            finalText = `[Document: ${doc_name} | Section: ${firstLine}]\n${text}`;
          }

          // Compute hash on the ORIGINAL text (before header injection)
          contentHash = computeContentHash(text);
        }

        const result = await addEntry(
          finalText,
          category,
          'user_explicit',
          finalTags,
          {
            decay_exempt: decay_exempt || 'false',
            importance: finalImportance,
            skipAdmission,
            content_type: finalContentType,
            chunk_index: finalChunkIndex,
            content_hash: contentHash,
            context: context || null,
          },
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

        const chunkSuffix = result.content_type === 'chunk'
          ? `\nContent type: chunk\nChunk index: ${result.chunk_index}\nDoc: ${extractDocTag(result.tags) || 'n/a'}`
          : '';

        return {
          content: [
            {
              type: 'text',
              text: `Stored. ID: ${result.id}\nCategory: ${result.category}\nTier: ${result.tier}\nDecay exempt: ${result.decay_exempt}\nImportance: ${result.importance}${chunkSuffix}\nText: ${result.text}`,
            },
          ],
        };
      } catch (err) {
        return { content: [{ type: 'text', text: `Error: ${err.message}` }], isError: true };
      }
    },
  },

  // ---------------------------------------------------------------------------
  // forget — remove entries (cascade for document parents)
  // ---------------------------------------------------------------------------
  {
    name: `${p}forget`,
    definition: {
      name: `${p}forget`,
      description:
        `Remove an entry from ${owner}'s knowledge base. Finds the closest semantic match to the query and removes it. If the match is a document parent chunk, all sections of that document are also removed (cascade delete).`,
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

        if (removed.cascadeDeleted != null) {
          return {
            content: [
              {
                type: 'text',
                text: `Removed document "${removed.docName}" and all its sections.\nCascade deleted: ${removed.cascadeDeleted} chunk(s)\nParent ID: ${removed.id}\nCategory: ${removed.category}\nText: ${removed.text}`,
              },
            ],
          };
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

  // ---------------------------------------------------------------------------
  // reflect — audit what's stored
  // ---------------------------------------------------------------------------
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

  // ---------------------------------------------------------------------------
  // recall — retrieve full document by name
  // ---------------------------------------------------------------------------
  {
    name: `${p}recall`,
    definition: {
      name: `${p}recall`,
      description:
        `Retrieve a full document from ${owner}'s knowledge base by name. Use this when ${owner} asks to see a complete plan, strategy, or research document — not for searching specific facts (use ask for that). Returns all sections in reading order.`,
      inputSchema: {
        type: 'object',
        properties: {
          name: {
            type: 'string',
            description:
              'Document name (e.g. "career-plan", "driving-license", "ecosystem-research"). Matches the doc:<name> tag used when the document was stored.',
          },
        },
        required: ['name'],
      },
    },
    async handler({ name }) {
      try {
        // Validate name — no injection vectors
        if (!name || typeof name !== 'string' || name.trim().length === 0) {
          return { content: [{ type: 'text', text: 'Document name is required.' }], isError: true };
        }
        const safeName = name.trim();

        const chunks = await listByDocTag(safeName);
        if (chunks.length === 0) {
          return {
            content: [
              {
                type: 'text',
                text: `No document found with name "${safeName}". Use memorize with content_type=chunk and doc_name="${safeName}" to store one.`,
              },
            ],
          };
        }

        // Filter superseded (listByDocTag already does this, but be defensive)
        const active = chunks.filter(c => !c.superseded_at);
        if (active.length === 0) {
          return {
            content: [
              { type: 'text', text: `Document "${safeName}" exists but all chunks are superseded.` },
            ],
          };
        }

        // Sort by chunk_index numerically, nulls last
        active.sort((a, b) => {
          const ai = a.chunk_index != null ? parseInt(a.chunk_index, 10) : Infinity;
          const bi = b.chunk_index != null ? parseInt(b.chunk_index, 10) : Infinity;
          return ai - bi;
        });

        // Concatenate with section separators
        const sections = active.map((c) => {
          const label = c.chunk_index === '0'
            ? `[Summary]`
            : `[Section ${c.chunk_index}]`;
          // Strip context prefix from display — show clean original text
          let displayText = c.text;
          if (c.context && displayText.startsWith(c.context.trim())) {
            displayText = displayText.slice(c.context.trim().length).replace(/^\n+/, '');
          }
          return `${label}\n${displayText}`;
        });

        const docText = sections.join('\n\n---\n\n');
        // Use the most recent created_at across all chunks (handles resynced docs correctly)
        const latestDate = active.reduce((latest, c) => {
          const d = c.created_at || '';
          return d > latest ? d : latest;
        }, '');
        const header = `Document: ${safeName} (${active.length} section${active.length !== 1 ? 's' : ''}, last updated: ${formatDate(latestDate)})`;

        return {
          content: [
            {
              type: 'text',
              text: `${header}\n\n${docText}`,
            },
          ],
        };
      } catch (err) {
        return { content: [{ type: 'text', text: `Error: ${err.message}` }], isError: true };
      }
    },
  },

  // ---------------------------------------------------------------------------
  // todos — action items and short-term tasks
  // ---------------------------------------------------------------------------
  {
    name: `${p}todos`,
    definition: {
      name: `${p}todos`,
      description:
        `Manage ${owner}'s action items and short-term tasks. Use this for quick reminders, next-session tasks, and to-do items that don't belong in the knowledge base. Actions: list (default), add, done, remove.`,
      inputSchema: {
        type: 'object',
        properties: {
          action: {
            type: 'string',
            enum: ['list', 'add', 'done', 'remove'],
            description: 'Action to perform. Default: list',
          },
          text: {
            type: 'string',
            description: 'For add: the todo text.',
          },
          id: {
            type: 'number',
            description: 'For done/remove: the todo ID.',
          },
          due_date: {
            type: 'string',
            description: 'Optional due date in YYYY-MM-DD format.',
          },
          tags: {
            type: 'string',
            description: 'Optional comma-separated tags for filtering.',
          },
        },
      },
    },
    async handler({ action = 'list', text, id, due_date, tags }) {
      try {
        switch (action) {
          case 'add': {
            if (!text || typeof text !== 'string' || text.trim().length === 0) {
              return { content: [{ type: 'text', text: 'text is required for add action.' }], isError: true };
            }
            const todo = addTodo(text.trim(), due_date || null, tags || null);
            return {
              content: [
                {
                  type: 'text',
                  text: `Added todo #${todo.id}: ${todo.text}${due_date ? ` (due: ${due_date})` : ''}`,
                },
              ],
            };
          }

          case 'done': {
            const todoId = typeof id === 'number' ? Math.floor(id) : null;
            if (!todoId || todoId <= 0) {
              return { content: [{ type: 'text', text: 'id is required for done action.' }], isError: true };
            }
            const completed = completeTodo(todoId);
            if (!completed) {
              return {
                content: [{ type: 'text', text: `Todo #${todoId} not found or already completed.` }],
              };
            }
            return { content: [{ type: 'text', text: `Completed todo #${todoId}.` }] };
          }

          case 'remove': {
            const todoId = typeof id === 'number' ? Math.floor(id) : null;
            if (!todoId || todoId <= 0) {
              return { content: [{ type: 'text', text: 'id is required for remove action.' }], isError: true };
            }
            const removed = removeTodo(todoId);
            if (!removed) {
              return { content: [{ type: 'text', text: `Todo #${todoId} not found.` }] };
            }
            return { content: [{ type: 'text', text: `Removed todo #${todoId}.` }] };
          }

          case 'list':
          default: {
            const openTodos = listTodos('open');
            const stats = getTodoStats();

            if (openTodos.length === 0) {
              const summary = `No open todos. ${stats.done} completed total.`;
              return { content: [{ type: 'text', text: summary }] };
            }

            const lines = openTodos.map(t => {
              const due = t.due_date ? ` [due: ${t.due_date}]` : '';
              const overdue = t.due_date && t.due_date < new Date().toISOString().slice(0, 10) ? ' OVERDUE' : '';
              const tagStr = t.tags ? ` (${t.tags})` : '';
              return `#${t.id}${overdue} ${t.text}${due}${tagStr}`;
            });

            const overdueNote = stats.overdue > 0 ? ` (${stats.overdue} overdue)` : '';
            const header = `Open todos: ${stats.open}${overdueNote} | Done: ${stats.done}`;

            return {
              content: [
                {
                  type: 'text',
                  text: `${header}\n\n${lines.join('\n')}`,
                },
              ],
            };
          }
        }
      } catch (err) {
        return { content: [{ type: 'text', text: `Error: ${err.message}` }], isError: true };
      }
    },
  },
];
