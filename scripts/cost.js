/**
 * cost.js — Display embedding cost data from SQLite.
 *
 * Usage:
 *   npm run cost              Show cost breakdown (today, week, month, all-time)
 */

import { getCostSummary } from '../src/cost-tracker.js';

function fmt(cost) { return '$' + cost.toFixed(6); }
function tok(n) { return n.toLocaleString(); }

function main() {
  console.log('memex cost report\n');

  const s = getCostSummary();

  if (s.totals.calls === 0) {
    console.log('No embedding calls recorded yet.');
    console.log('The cost ledger is populated automatically as you use memex.\n');
    return;
  }

  console.log(`Model: ${s.model}`);
  console.log(`Price: $${s.pricePerKToken} per 1K tokens\n`);

  // Time-based breakdown
  console.log('=== Cost Summary ===\n');
  console.log(`  ${'Period'.padEnd(14)} ${'Calls'.padStart(8)} ${'Tokens'.padStart(12)} ${'Cost'.padStart(12)}`);
  console.log(`  ${'─'.repeat(14)} ${'─'.repeat(8)} ${'─'.repeat(12)} ${'─'.repeat(12)}`);
  console.log(`  ${'Today'.padEnd(14)} ${String(s.today.calls).padStart(8)} ${tok(s.today.tokens).padStart(12)} ${fmt(s.today.cost).padStart(12)}`);
  console.log(`  ${'This week'.padEnd(14)} ${String(s.week.calls).padStart(8)} ${tok(s.week.tokens).padStart(12)} ${fmt(s.week.cost).padStart(12)}`);
  console.log(`  ${'This month'.padEnd(14)} ${String(s.month.calls).padStart(8)} ${tok(s.month.tokens).padStart(12)} ${fmt(s.month.cost).padStart(12)}`);
  console.log(`  ${'All time'.padEnd(14)} ${String(s.totals.calls).padStart(8)} ${tok(s.totals.tokens).padStart(12)} ${fmt(s.totals.cost).padStart(12)}`);

  // By operation
  if (s.byOp.length > 0) {
    console.log('\n=== By Operation ===\n');
    for (const row of s.byOp) {
      console.log(`  ${row.operation.padEnd(12)} ${String(row.calls).padStart(6)} calls  ${tok(row.tokens).padStart(10)} tokens  ${fmt(row.cost).padStart(12)}`);
    }
  }

  // Daily breakdown
  if (s.byDay.length > 0) {
    console.log('\n=== Last 14 Days ===\n');
    for (const row of s.byDay) {
      const bar = '█'.repeat(Math.max(1, Math.round(row.tokens / 100)));
      console.log(`  ${row.date}  ${String(row.calls).padStart(4)} calls  ${tok(row.tokens).padStart(8)} tokens  ${fmt(row.cost).padStart(10)}  ${bar}`);
    }
  }

  console.log();
}

main();
