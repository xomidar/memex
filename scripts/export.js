/**
 * export.js — Export full knowledge base to data/backup.json
 *
 * Vectors are NOT included — they're too large to diff and get regenerated
 * from the text field via the embedding function when restoring.
 *
 * This file IS committed to git. It's the human-readable, text-diffable backup.
 *
 * Superseded chunks are excluded by default (pass --all to include them).
 * Superseded entries are historical — they've been replaced by a resync and
 * including them in a restore would re-introduce stale data.
 *
 * Run: node scripts/export.js [--all]
 *   --all    Include superseded chunks in the export
 */

import { writeFile } from 'fs/promises';
import { exportAll } from '../src/db.js';
import { config } from '../src/config.js';

const OUTPUT_PATH = config.exportPath;
const includeAll = process.argv.includes('--all');

async function main() {
  console.log('memex export\n');

  let entries;
  try {
    console.log('Reading all entries from LanceDB...');
    entries = await exportAll();
  } catch (err) {
    console.error(`Failed to read from LanceDB: ${err.message}`);
    process.exit(1);
  }

  console.log(`Found ${entries.length} entries.`);

  // Filter superseded chunks unless --all is specified
  let exported = entries;
  if (!includeAll) {
    const before = entries.length;
    exported = entries.filter(e => !e.superseded_at);
    const supersededCount = before - exported.length;
    if (supersededCount > 0) {
      console.log(`Excluded ${supersededCount} superseded chunk(s). Use --all to include them.`);
    }
  } else {
    console.log('--all flag set: including superseded chunks.');
  }

  // Sort by created_at for stable diffs
  exported.sort((a, b) => (a.created_at || '').localeCompare(b.created_at || ''));

  const json = JSON.stringify(exported, null, 2);

  try {
    await writeFile(OUTPUT_PATH, json, 'utf-8');
    console.log(`\nExported ${exported.length} entries to: ${OUTPUT_PATH}`);
    console.log('Vectors excluded — regenerated from text on restore via npm run seed.');
  } catch (err) {
    console.error(`Failed to write backup: ${err.message}`);
    process.exit(1);
  }
}

main().catch(err => {
  console.error(`Fatal: ${err.message}`);
  process.exit(1);
});
