/**
 * rdms.js — Relational data management for todos and ephemeral task tracking.
 *
 * Uses Node.js built-in SQLite (node:sqlite, available Node >= 22).
 * Gracefully errors on older Node versions with a clear message.
 *
 * DB location: data/rdms.db (sibling of cost.db)
 *
 * Pattern mirrors cost-tracker.js: singleton db, lazy init, sync API.
 */

import { resolve, dirname } from 'path';
import { fileURLToPath } from 'url';
import { mkdirSync } from 'fs';
import { createRequire } from 'module';

const __dirname = dirname(fileURLToPath(import.meta.url));
const _require = createRequire(import.meta.url);

/**
 * Load node:sqlite synchronously via createRequire.
 * Returns the DatabaseSync constructor, or throws a user-friendly error on Node < 22.
 *
 * @returns {{ DatabaseSync: Function }}
 */
function loadSQLite() {
  try {
    return _require('node:sqlite');
  } catch {
    throw new Error(
      '[memex] rdms.js requires Node.js >= 22 for node:sqlite support. ' +
      'Current Node version: ' + process.version + '. ' +
      'Upgrade Node to use the todos tool.'
    );
  }
}

let db = null;

/**
 * Get (or initialise) the SQLite database.
 * Creates the todos table and indexes if they don't exist.
 *
 * @returns {object} DatabaseSync instance
 * @throws {Error} if node:sqlite is unavailable (Node < 22) or DB init fails
 */
function getDB() {
  if (db) return db;

  const { DatabaseSync } = loadSQLite();

  // Resolve data directory consistently with the rest of the system.
  // MEMEX_DATA_DIR typically points to the LanceDB kb/ dir (e.g., /data/kb).
  // rdms.db lives one level up (e.g., /data/rdms.db), same as cost.db.
  // Use dirname() to strip the last path segment safely regardless of trailing slash.
  const kbDir = process.env.MEMEX_DATA_DIR || resolve(__dirname, '..', 'data', 'kb');
  const dataDir = dirname(kbDir);

  mkdirSync(dataDir, { recursive: true });
  const dbPath = resolve(dataDir, 'rdms.db');

  db = new DatabaseSync(dbPath);
  db.exec(`
    CREATE TABLE IF NOT EXISTS todos (
      id           INTEGER PRIMARY KEY AUTOINCREMENT,
      text         TEXT NOT NULL,
      created_at   TEXT NOT NULL DEFAULT (datetime('now')),
      due_date     TEXT,
      status       TEXT NOT NULL DEFAULT 'open',
      completed_at TEXT,
      tags         TEXT
    );
    CREATE INDEX IF NOT EXISTS idx_todos_status ON todos(status);
    CREATE INDEX IF NOT EXISTS idx_todos_due    ON todos(due_date);
  `);
  return db;
}

/**
 * Add a new todo item.
 *
 * @param {string} text - the todo description
 * @param {string|null} [dueDate=null] - ISO date string YYYY-MM-DD or null
 * @param {string|null} [tags=null] - comma-separated tags or null
 * @returns {{ id: number, text: string, status: string, due_date: string|null, tags: string|null }}
 */
export function addTodo(text, dueDate = null, tags = null) {
  const d = getDB();
  const stmt = d.prepare(
    "INSERT INTO todos (text, created_at, due_date, tags) VALUES (?, datetime('now'), ?, ?)"
  );
  const result = stmt.run(text, dueDate, tags);
  return { id: Number(result.lastInsertRowid), text, status: 'open', due_date: dueDate, tags };
}

/**
 * List todos by status, ordered by due date (nulls last) then creation time.
 *
 * @param {'open'|'done'} [status='open']
 * @returns {Array<object>}
 */
export function listTodos(status = 'open') {
  const d = getDB();
  const stmt = d.prepare(
    'SELECT * FROM todos WHERE status = ? ORDER BY due_date IS NULL, due_date ASC, created_at ASC'
  );
  return stmt.all(status);
}

/**
 * Mark a todo as completed.
 *
 * @param {number} id - positive integer todo ID
 * @returns {boolean} true if a row was updated, false if ID not found or already done
 * @throws {Error} if id is not a positive integer
 */
export function completeTodo(id) {
  if (!Number.isInteger(id) || id <= 0) throw new Error(`Invalid todo ID: ${id}`);
  const d = getDB();
  const stmt = d.prepare(
    "UPDATE todos SET status = 'done', completed_at = datetime('now') WHERE id = ? AND status = 'open'"
  );
  const result = stmt.run(id);
  return result.changes > 0;
}

/**
 * Permanently remove a todo regardless of status.
 *
 * @param {number} id - positive integer todo ID
 * @returns {boolean} true if deleted, false if not found
 * @throws {Error} if id is not a positive integer
 */
export function removeTodo(id) {
  if (!Number.isInteger(id) || id <= 0) throw new Error(`Invalid todo ID: ${id}`);
  const d = getDB();
  const stmt = d.prepare('DELETE FROM todos WHERE id = ?');
  const result = stmt.run(id);
  return result.changes > 0;
}

/**
 * Get summary counts: open, done, overdue.
 *
 * @returns {{ open: number, done: number, overdue: number }}
 */
export function getTodoStats() {
  const d = getDB();
  const open    = d.prepare("SELECT COUNT(*) as count FROM todos WHERE status = 'open'").get();
  const done    = d.prepare("SELECT COUNT(*) as count FROM todos WHERE status = 'done'").get();
  const overdue = d.prepare(
    "SELECT COUNT(*) as count FROM todos WHERE status = 'open' AND due_date < date('now')"
  ).get();
  return {
    open:    Number(open.count),
    done:    Number(done.count),
    overdue: Number(overdue.count),
  };
}

/**
 * Archive (delete) completed todos older than N days.
 * Keeps the done list clean without losing recent completions.
 *
 * @param {number} [days=30] - minimum age in days for archival
 * @returns {number} number of rows deleted
 * @throws {Error} if days is not a non-negative integer
 */
export function archiveOldTodos(days = 30) {
  if (!Number.isInteger(days) || days < 0) throw new Error(`Invalid days value: ${days}`);
  const d = getDB();
  const stmt = d.prepare(
    `DELETE FROM todos WHERE status = 'done' AND completed_at < datetime('now', '-' || ? || ' days')`
  );
  const result = stmt.run(days);
  return result.changes;
}
