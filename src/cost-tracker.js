/**
 * cost-tracker.js — Tracks embedding API costs per call using SQLite.
 *
 * Every embedding operation (store, search, seed) is logged with token
 * count and dollar cost. Zero file rewriting — append-only inserts.
 *
 * Pricing (updated March 2026):
 *   text-embedding-3-large:  $0.00013 / 1K tokens
 *   text-embedding-3-small:  $0.00002 / 1K tokens
 */

import { DatabaseSync } from 'node:sqlite';
import { mkdirSync } from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { config } from './config.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const DB_DIR = path.resolve(__dirname, '..', 'data');
const DB_PATH = path.join(DB_DIR, 'cost.db');

const PRICING = {
  'text-embedding-3-large': 0.00013,
  'text-embedding-3-small': 0.00002,
};

let db = null;

function getDB() {
  if (db) return db;
  mkdirSync(DB_DIR, { recursive: true });
  db = new DatabaseSync(DB_PATH);
  db.exec(`
    CREATE TABLE IF NOT EXISTS embedding_costs (
      id          INTEGER PRIMARY KEY AUTOINCREMENT,
      timestamp   TEXT NOT NULL,
      date        TEXT NOT NULL,
      operation   TEXT NOT NULL,
      input_count INTEGER NOT NULL,
      tokens      INTEGER NOT NULL,
      cost        REAL NOT NULL,
      model       TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_cost_date ON embedding_costs(date);
    CREATE INDEX IF NOT EXISTS idx_cost_op ON embedding_costs(operation);
  `);
  return db;
}

/**
 * Record a single embedding API call.
 */
export function recordUsage(operation, inputCount, totalTokens) {
  try {
    const database = getDB();
    const pricePerKToken = PRICING[config.embeddingModel] || 0.00013;
    const cost = (totalTokens / 1000) * pricePerKToken;
    const now = new Date();

    database.prepare(`
      INSERT INTO embedding_costs (timestamp, date, operation, input_count, tokens, cost, model)
      VALUES (?, ?, ?, ?, ?, ?, ?)
    `).run(
      now.toISOString(),
      now.toISOString().slice(0, 10),
      operation,
      inputCount,
      totalTokens,
      cost,
      config.embeddingModel,
    );
  } catch (err) {
    process.stderr.write(`[memex] cost tracking failed: ${err.message}\n`);
  }
}

/**
 * Get a full cost summary with today/week/month breakdowns.
 */
export function getCostSummary() {
  const database = getDB();

  const now = new Date();
  const today = now.toISOString().slice(0, 10);

  // Week: last 7 days
  const weekAgo = new Date(now.getTime() - 7 * 24 * 60 * 60 * 1000).toISOString().slice(0, 10);

  // Month: last 30 days
  const monthAgo = new Date(now.getTime() - 30 * 24 * 60 * 60 * 1000).toISOString().slice(0, 10);

  // Totals
  const totals = database.prepare(`
    SELECT COUNT(*) as calls, COALESCE(SUM(tokens), 0) as tokens, COALESCE(SUM(cost), 0) as cost
    FROM embedding_costs
  `).get();

  // Today
  const todayStats = database.prepare(`
    SELECT COUNT(*) as calls, COALESCE(SUM(tokens), 0) as tokens, COALESCE(SUM(cost), 0) as cost
    FROM embedding_costs WHERE date = ?
  `).get(today);

  // This week
  const weekStats = database.prepare(`
    SELECT COUNT(*) as calls, COALESCE(SUM(tokens), 0) as tokens, COALESCE(SUM(cost), 0) as cost
    FROM embedding_costs WHERE date >= ?
  `).get(weekAgo);

  // This month
  const monthStats = database.prepare(`
    SELECT COUNT(*) as calls, COALESCE(SUM(tokens), 0) as tokens, COALESCE(SUM(cost), 0) as cost
    FROM embedding_costs WHERE date >= ?
  `).get(monthAgo);

  // By operation
  const byOp = database.prepare(`
    SELECT operation, COUNT(*) as calls, SUM(tokens) as tokens, SUM(cost) as cost
    FROM embedding_costs GROUP BY operation ORDER BY cost DESC
  `).all();

  // By day (last 14 days)
  const byDay = database.prepare(`
    SELECT date, COUNT(*) as calls, SUM(tokens) as tokens, SUM(cost) as cost
    FROM embedding_costs WHERE date >= ? GROUP BY date ORDER BY date DESC
  `).all(new Date(now.getTime() - 14 * 24 * 60 * 60 * 1000).toISOString().slice(0, 10));

  // Model info
  const model = database.prepare(`
    SELECT model FROM embedding_costs ORDER BY id DESC LIMIT 1
  `).get();

  return {
    model: model?.model || config.embeddingModel,
    pricePerKToken: PRICING[config.embeddingModel] || 0.00013,
    totals,
    today: todayStats,
    week: weekStats,
    month: monthStats,
    byOp,
    byDay,
  };
}
