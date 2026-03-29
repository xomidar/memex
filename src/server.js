import './env.js';  // MUST be first — loads .env before config.js reads process.env
import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
  ListPromptsRequestSchema,
  GetPromptRequestSchema,
} from '@modelcontextprotocol/sdk/types.js';
import { tools, prompts } from './tools.js';
import { initDB } from './db.js';

// Validate required env var at startup
if (!process.env.OPENAI_API_KEY) {
  process.stderr.write('[memex] ERROR: OPENAI_API_KEY environment variable is not set.\n');
  process.exit(1);
}

const server = new Server(
  { name: 'memex', version: '1.0.0' },
  { capabilities: { tools: {}, prompts: {} } }
);

// Build lookup maps for handlers
const toolHandlers = new Map(tools.map(t => [t.name, t.handler]));
const toolDefinitions = tools.map(t => t.definition);
const promptHandlers = new Map(prompts.map(p => [p.name, p.handler]));
const promptDefinitions = prompts.map(p => p.definition);

// List tools
server.setRequestHandler(ListToolsRequestSchema, async () => {
  return { tools: toolDefinitions };
});

// List prompts
server.setRequestHandler(ListPromptsRequestSchema, async () => {
  return { prompts: promptDefinitions };
});

// Get prompt
server.setRequestHandler(GetPromptRequestSchema, async (request) => {
  const { name, arguments: args } = request.params;
  const handler = promptHandlers.get(name);
  if (!handler) {
    throw new Error(`Unknown prompt: ${name}`);
  }
  return handler(args || {});
});

// Call tool
server.setRequestHandler(CallToolRequestSchema, async (request) => {
  const { name, arguments: args } = request.params;
  const handler = toolHandlers.get(name);
  if (!handler) {
    return {
      content: [{ type: 'text', text: `Unknown tool: ${name}` }],
      isError: true,
    };
  }
  try {
    return await handler(args || {});
  } catch (err) {
    process.stderr.write(`[memex] Unhandled error in tool "${name}": ${err.message}\n`);
    return {
      content: [{ type: 'text', text: `Internal error: ${err.message}` }],
      isError: true,
    };
  }
});

async function main() {
  // Pre-warm the DB connection so the first tool call isn't slow
  try {
    await initDB();
    process.stderr.write('[memex] LanceDB connected and ready.\n');
  } catch (err) {
    process.stderr.write(`[memex] WARNING: DB pre-warm failed: ${err.message}\n`);
    // Don't exit — let individual tool calls surface the error with context
  }

  const transport = new StdioServerTransport();
  await server.connect(transport);
  process.stderr.write('[memex] Server running on stdio.\n');
}

// Graceful shutdown
process.on('SIGINT', () => {
  process.stderr.write('[memex] Shutting down.\n');
  process.exit(0);
});
process.on('SIGTERM', () => {
  process.stderr.write('[memex] Shutting down.\n');
  process.exit(0);
});

main().catch(err => {
  process.stderr.write(`[memex] Fatal: ${err.message}\n`);
  process.exit(1);
});
