# Memex

[![SafeSkill 89/100](https://img.shields.io/badge/SafeSkill-89%2F100_Passes%20with%20Notes-yellow)](https://safeskill.dev/scan/xomidar-memex)

Semantic memory for AI agents. An MCP server that gives your AI assistant a persistent, searchable knowledge base powered by vector embeddings.

Built for [Claude Code](https://docs.anthropic.com/en/docs/claude-code). Works with any MCP-compatible client.

## What is Memex?

In 1945, Vannevar Bush — MIT engineer and director of the US Office of Scientific Research and Development — published ["As We May Think"](https://www.theatlantic.com/magazine/archive/1945/07/as-we-may-think/303881/) in The Atlantic. He described a hypothetical device called the **memex** (memory + index):

> *"A memex is a device in which an individual stores all his books, records, and communications, and which is mechanized so that it may be consulted with exceeding speed and flexibility."*

The memex was never built. The technology didn't exist. But the concept — a personal device that extends human memory through associative retrieval — became the intellectual foundation for hypertext (Ted Nelson cited it), the World Wide Web (Tim Berners-Lee cited it), and personal computing.

This project brings Bush's vision to AI agents. Instead of losing context between sessions or guessing about user preferences, your AI assistant can store and retrieve knowledge semantically — by meaning, not keywords.

## How It Works

Memex runs as a local [MCP server](https://modelcontextprotocol.io/) that exposes four tools:

| Tool | Purpose |
|------|---------|
| `ask` | Semantic search — find relevant context by meaning |
| `memorize` | Store new knowledge with auto-generated embeddings |
| `forget` | Remove an entry by semantic match |
| `reflect` | Audit what's stored about any topic |

Under the hood:
- **[LanceDB](https://lancedb.com/)** — embedded vector database, no server needed
- **[OpenAI text-embedding-3-large](https://platform.openai.com/docs/guides/embeddings)** — 3072-dimension embeddings, auto-generated on insert and search
- **MCP SDK** — stdio transport, registers as a native tool in Claude Code

## Quick Start

```bash
git clone https://github.com/xomidar/memex.git
cd memex
npm install
cp .env.example .env
# Edit .env — set OPENAI_API_KEY and customize prefix/owner
```

Register with Claude Code:
```bash
# Linux / macOS
claude mcp add memex -- node /path/to/memex/src/server.js

# Windows
claude mcp add memex -- node C:/path/to/memex/src/server.js
```

> **Note:** Use absolute paths — Claude Code launches the server from any working directory. The server auto-loads `.env` from the project root, so no `--env-file` flag is needed.

To remove:
```bash
claude mcp remove memex
```

That's it. Your AI assistant now has `ask`, `memorize`, `forget`, and `reflect` (accessed as `mcp__memex__ask`, etc. in Claude Code). Set `MEMEX_PREFIX=reza` if you want personalized names like `reza_ask`.

## Configuration

All configuration is via environment variables (or `.env` file):

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | *(required)* | OpenAI API key for embeddings |
| `MEMEX_PREFIX` | *(empty)* | Tool name prefix — e.g., `reza` → `reza_ask`. Empty = clean names |
| `MEMEX_OWNER` | `the user` | Name in tool descriptions — e.g., `Reza` → "Search Reza's knowledge base" |
| `MEMEX_EMBEDDING_MODEL` | `text-embedding-3-large` | OpenAI embedding model |
| `MEMEX_EMBEDDING_DIMS` | `3072` | Embedding dimensions |
| `MEMEX_DATA_DIR` | `./data/kb` | LanceDB data directory |
| `MEMEX_EXPORT_PATH` | `./data/backup.json` | Export file path |

## Knowledge Categories

Entries are tagged with a category for filtered retrieval:

| Category | What it stores |
|----------|----------------|
| `personal` | User profile — role, goals, skills, background |
| `preference` | How the user likes to work — confirmed approaches, corrections |
| `decision` | Past decisions with context and rationale |
| `project` | Ongoing work, goals, deadlines, stakeholders |
| `routing` | Agent routing precedents (for multi-agent systems) |
| `reference` | External resources, documentation pointers |

## Seeding Data

Import from various sources:

```bash
# From a directory of markdown files (with YAML frontmatter)
npm run seed -- --md-dir /path/to/markdown/files

# From a previous Memex export
npm run seed -- --backup /path/to/backup.json

# From a SQLite routing decisions database
npm run seed -- --routing-db /path/to/routing.db

# All at once
npm run seed -- --md-dir ./docs --backup ./old-backup.json --routing-db ./routing.db
```

Markdown files should have YAML frontmatter with a `type` field (`user`, `feedback`, `project`, `reference`) which maps to Memex categories.

## Data Backup

Export the full knowledge base as human-readable JSON (vectors excluded — they're regenerated on import):

```bash
npm run export
```

This writes to `data/backup.json`. Commit it to a **separate private repo** for version-controlled backup:

```bash
# In your private data repo
cp /path/to/memex/data/backup.json .
git add backup.json && git commit -m "knowledge base update"
```

To restore on a new machine:
```bash
npm run seed -- --backup /path/to/backup.json
```

## Architecture

```
memex/
├── src/
│   ├── server.js     ← MCP server (stdio transport)
│   ├── config.js     ← All configurable values, reads from env
│   ├── db.js         ← LanceDB connection, schema, CRUD operations
│   └── tools.js      ← MCP tool definitions (ask, memorize, forget, reflect)
├── scripts/
│   ├── seed.js       ← Import from markdown, JSON backup, or SQLite
│   └── export.js     ← Export knowledge base to JSON
├── data/
│   └── kb/           ← LanceDB files (gitignored — use backup.json)
├── .env.example      ← Configuration template
└── package.json
```

## Optimization Strategies

See [Optimization Strategies](docs/OPTIMIZATION.md) for the research and implementation details behind memex's retrieval pipeline.

## Requirements

- Node.js 20+
- OpenAI API key (for embeddings — ~$0.00013 per entry at 3072 dims)
- npm

## License

MIT — see [LICENSE](LICENSE)

## Author

[Rezaul Hasan](https://www.rezaulhasan.me)
