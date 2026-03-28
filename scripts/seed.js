/**
 * seed.js — Import data into Memex from markdown files and/or a JSON backup.
 *
 * Sources (all optional, skipped if not found):
 *   1. Markdown directory — parses YAML frontmatter for type/category, body as text
 *      Set MEMEX_SEED_MD_DIR env var or pass --md-dir <path>
 *
 *   2. JSON backup file — re-imports a previous export (vectors regenerated)
 *      Set MEMEX_SEED_BACKUP env var or pass --backup <path>
 *
 *   3. SQLite routing DB — imports reviewed routing decisions
 *      Set MEMEX_SEED_ROUTING_DB env var or pass --routing-db <path>
 *
 * Run: node scripts/seed.js [--md-dir <path>] [--backup <path>] [--routing-db <path>]
 */

import { readdir, readFile } from 'fs/promises';
import { existsSync } from 'fs';
import path from 'path';
import { importEntries } from '../src/db.js';

// Parse CLI args
function getArg(flag) {
  const idx = process.argv.indexOf(flag);
  return idx !== -1 && process.argv[idx + 1] ? process.argv[idx + 1] : null;
}

const MD_DIR     = getArg('--md-dir')     || process.env.MEMEX_SEED_MD_DIR     || null;
const BACKUP     = getArg('--backup')     || process.env.MEMEX_SEED_BACKUP     || null;
const ROUTING_DB = getArg('--routing-db') || process.env.MEMEX_SEED_ROUTING_DB || null;

const TYPE_TO_CATEGORY = {
  user: 'personal',
  feedback: 'preference',
  project: 'project',
  reference: 'reference',
};

function parseFrontmatter(content) {
  const lines = content.split('\n');
  if (lines[0].trim() !== '---') return { meta: {}, body: content.trim() };

  let closeIdx = -1;
  for (let i = 1; i < lines.length; i++) {
    if (lines[i].trim() === '---') { closeIdx = i; break; }
  }
  if (closeIdx === -1) return { meta: {}, body: content.trim() };

  const frontmatterLines = lines.slice(1, closeIdx);
  const body = lines.slice(closeIdx + 1).join('\n').trim();

  const meta = {};
  for (const line of frontmatterLines) {
    const match = line.match(/^(\w+):\s*(.+)$/);
    if (match) meta[match[1].trim()] = match[2].trim();
  }
  return { meta, body };
}

async function seedFromMarkdown(dir) {
  if (!dir || !existsSync(dir)) {
    if (dir) console.log(`  Directory not found: ${dir} — skipping.`);
    return [];
  }

  const files = await readdir(dir);
  const mdFiles = files.filter(f => f.endsWith('.md') && f.toUpperCase() !== 'MEMORY.MD' && f.toUpperCase() !== 'README.MD');

  const entries = [];
  for (const file of mdFiles) {
    const raw = await readFile(path.join(dir, file), 'utf-8');
    const { meta, body } = parseFrontmatter(raw);
    if (!body) { console.log(`  Skipping ${file} — empty body`); continue; }

    const memoryType = meta.type || 'reference';
    const category = TYPE_TO_CATEGORY[memoryType] || 'reference';
    const name = meta.name || file.replace('.md', '');

    entries.push({
      text: `${name}: ${body}`,
      category,
      source: 'migrated',
      tags: `memory,${memoryType}`,
    });
    console.log(`  + [${category}] ${name}`);
  }
  return entries;
}

async function seedFromBackup(backupPath) {
  if (!backupPath || !existsSync(backupPath)) {
    if (backupPath) console.log(`  Backup not found: ${backupPath} — skipping.`);
    return [];
  }

  const raw = await readFile(backupPath, 'utf-8');
  const entries = JSON.parse(raw);
  console.log(`  Found ${entries.length} entries in backup.`);
  return entries;
}

async function seedFromRoutingDB(dbPath) {
  if (!dbPath || !existsSync(dbPath)) {
    if (dbPath) console.log(`  Routing DB not found: ${dbPath} — skipping.`);
    return [];
  }

  let rows;
  try {
    const { DatabaseSync } = await import('node:sqlite');
    const db = new DatabaseSync(dbPath);
    rows = db.prepare(`
      SELECT description, agent, verdict, correct_agent, notes, timestamp
      FROM routing_decisions WHERE verdict IS NOT NULL
    `).all();
    db.close();
  } catch (err) {
    console.error(`  Failed to read routing DB: ${err.message}`);
    return [];
  }

  if (rows.length === 0) { console.log('  No reviewed decisions found.'); return []; }

  const entries = [];
  for (const row of rows) {
    let text = `Task: ${row.description}. Routed to: ${row.agent}. Verdict: ${row.verdict}.`;
    if (row.verdict === 'incorrect' && row.correct_agent) text += ` Correct agent: ${row.correct_agent}.`;
    if (row.notes) text += ` Notes: ${row.notes}`;

    entries.push({
      text,
      category: 'routing',
      source: 'reviewed_routing',
      tags: `routing,${row.agent},${row.verdict}`,
      created_at: row.timestamp || new Date().toISOString(),
    });
    console.log(`  + [routing] ${row.description} → ${row.agent} (${row.verdict})`);
  }
  return entries;
}

async function main() {
  console.log('memex seed\n');

  // Dimension mismatch warning — if MEMEX_EMBEDDING_DIMS differs from what was used to
  // originally build the table, LanceDB will throw a shape error on the first add().
  // The fix: delete the table directory (MEMEX_DATA_DIR or ./data/kb) and re-run seed.
  const dims = process.env.MEMEX_EMBEDDING_DIMS || '3072';
  console.log(`Embedding dims: ${dims} (from MEMEX_EMBEDDING_DIMS or default 3072)`);
  console.log('WARNING: If your LanceDB table was built with different dimensions, this will');
  console.log('         fail with a shape mismatch. Delete the table directory and re-seed.\n');

  if (!MD_DIR && !BACKUP && !ROUTING_DB) {
    console.log('No sources specified. Use:\n');
    console.log('  --md-dir <path>      Directory of markdown files with YAML frontmatter');
    console.log('  --backup <path>      JSON backup from memex export');
    console.log('  --routing-db <path>  SQLite routing decisions DB\n');
    console.log('Or set env vars: MEMEX_SEED_MD_DIR, MEMEX_SEED_BACKUP, MEMEX_SEED_ROUTING_DB');
    process.exit(0);
  }

  const all = [];

  if (MD_DIR) {
    console.log(`Reading markdown files from ${MD_DIR}...`);
    const beforeMd = all.length;
    all.push(...await seedFromMarkdown(MD_DIR));
    console.log(`  ${all.length - beforeMd} entries\n`);
  }

  if (BACKUP) {
    console.log(`Reading backup from ${BACKUP}...`);
    const before = all.length;
    all.push(...await seedFromBackup(BACKUP));
    console.log(`  ${all.length - before} entries\n`);
  }

  if (ROUTING_DB) {
    console.log(`Reading routing decisions from ${ROUTING_DB}...`);
    const before = all.length;
    all.push(...await seedFromRoutingDB(ROUTING_DB));
    console.log(`  ${all.length - before} entries\n`);
  }

  if (all.length === 0) {
    console.log('Nothing to import.');
    return;
  }

  console.log(`Importing ${all.length} entries into LanceDB...`);
  console.log('(This calls OpenAI embeddings — may take a moment)\n');

  const count = await importEntries(all);
  console.log(`Done. Imported ${count} entries.`);
}

main().catch(err => {
  console.error(`Fatal: ${err.message}`);
  process.exit(1);
});
