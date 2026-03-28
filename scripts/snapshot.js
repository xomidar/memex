/**
 * snapshot.js — Snapshot and restore the LanceDB data directory.
 *
 * Usage:
 *   npm run snapshot              Create a timestamped snapshot
 *   npm run snapshot:list         List available snapshots
 *   npm run snapshot:restore      Restore the most recent snapshot
 *   npm run snapshot:restore -- --id <timestamp>  Restore a specific snapshot
 *
 * Snapshots are full copies of the data directory stored alongside it.
 * The MCP server should be stopped (or disconnected) before restoring.
 */

import { cpSync, mkdirSync, rmSync, readdirSync, existsSync, statSync } from 'fs';
import path from 'path';
import { config } from '../src/config.js';

const DATA_DIR = config.dataDir;
const SNAPSHOT_ROOT = path.resolve(DATA_DIR, '..', 'snapshots');

function getArg(flag) {
  const idx = process.argv.indexOf(flag);
  return idx !== -1 && process.argv[idx + 1] ? process.argv[idx + 1] : null;
}

const command = process.argv[2] || 'create';

function createSnapshot() {
  if (!existsSync(DATA_DIR)) {
    console.log('Nothing to snapshot — data directory does not exist:', DATA_DIR);
    process.exit(0);
  }

  const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
  const snapshotDir = path.join(SNAPSHOT_ROOT, timestamp);

  mkdirSync(snapshotDir, { recursive: true });
  cpSync(DATA_DIR, snapshotDir, { recursive: true });

  const size = getDirSize(snapshotDir);
  console.log(`Snapshot created: ${timestamp}`);
  console.log(`Location: ${snapshotDir}`);
  console.log(`Size: ${(size / 1024 / 1024).toFixed(2)} MB`);
}

function listSnapshots() {
  if (!existsSync(SNAPSHOT_ROOT)) {
    console.log('No snapshots yet.');
    return;
  }

  const dirs = readdirSync(SNAPSHOT_ROOT)
    .filter(d => statSync(path.join(SNAPSHOT_ROOT, d)).isDirectory())
    .sort()
    .reverse();

  if (dirs.length === 0) {
    console.log('No snapshots yet.');
    return;
  }

  console.log(`\n${dirs.length} snapshot(s) in ${SNAPSHOT_ROOT}:\n`);
  for (const d of dirs) {
    const size = getDirSize(path.join(SNAPSHOT_ROOT, d));
    const label = d === dirs[0] ? ' (latest)' : '';
    console.log(`  ${d}  ${(size / 1024 / 1024).toFixed(2)} MB${label}`);
  }
  console.log();
}

function restoreSnapshot() {
  if (!existsSync(SNAPSHOT_ROOT)) {
    console.log('No snapshots available.');
    process.exit(1);
  }

  const targetId = getArg('--id');
  let snapshotDir;

  if (targetId) {
    snapshotDir = path.join(SNAPSHOT_ROOT, targetId);
    if (!existsSync(snapshotDir)) {
      console.log(`Snapshot not found: ${targetId}`);
      console.log('Run "npm run snapshot:list" to see available snapshots.');
      process.exit(1);
    }
  } else {
    // Use most recent
    const dirs = readdirSync(SNAPSHOT_ROOT)
      .filter(d => statSync(path.join(SNAPSHOT_ROOT, d)).isDirectory())
      .sort()
      .reverse();

    if (dirs.length === 0) {
      console.log('No snapshots available.');
      process.exit(1);
    }
    snapshotDir = path.join(SNAPSHOT_ROOT, dirs[0]);
    console.log(`Restoring latest snapshot: ${dirs[0]}`);
  }

  // Remove current data
  if (existsSync(DATA_DIR)) {
    rmSync(DATA_DIR, { recursive: true, force: true });
  }

  // Copy snapshot to data dir
  cpSync(snapshotDir, DATA_DIR, { recursive: true });

  console.log(`Restored from: ${snapshotDir}`);
  console.log('Restart the MCP server (/mcp) to pick up the restored data.');
}

function getDirSize(dir) {
  let total = 0;
  try {
    const entries = readdirSync(dir, { withFileTypes: true });
    for (const entry of entries) {
      const fullPath = path.join(dir, entry.name);
      if (entry.isDirectory()) {
        total += getDirSize(fullPath);
      } else {
        total += statSync(fullPath).size;
      }
    }
  } catch { /* skip inaccessible */ }
  return total;
}

// Route commands
if (command === 'create') createSnapshot();
else if (command === 'list') listSnapshots();
else if (command === 'restore') restoreSnapshot();
else {
  console.log('Usage:');
  console.log('  node scripts/snapshot.js create    Create a snapshot');
  console.log('  node scripts/snapshot.js list       List snapshots');
  console.log('  node scripts/snapshot.js restore    Restore latest (or --id <timestamp>)');
}
