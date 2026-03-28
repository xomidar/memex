/**
 * export.js — Export full knowledge base to data/backup.json
 *
 * Vectors are NOT included — they're too large to diff and get regenerated
 * from the text field via the embedding function when restoring.
 *
 * This file IS committed to git. It's the human-readable, text-diffable backup.
 *
 * Run: node scripts/export.js
 */

import { writeFile } from 'fs/promises';
import { exportAll } from '../src/db.js';
import { config } from '../src/config.js';

const OUTPUT_PATH = config.exportPath;

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

  // Sort by created_at for stable diffs
  entries.sort((a, b) => (a.created_at || '').localeCompare(b.created_at || ''));

  const json = JSON.stringify(entries, null, 2);

  try {
    await writeFile(OUTPUT_PATH, json, 'utf-8');
    console.log(`\nExported to: ${OUTPUT_PATH}`);
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
