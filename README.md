# mcpify

> Your API spec walked in. An MCP server walked out.

**mcpify** converts API specs into fully working [MCP](https://modelcontextprotocol.io) servers. Point it at an OpenAPI file, a Postman collection, a HAR capture, or a GraphQL schema — it parses every endpoint, builds a local semantic search index, and spits out a TypeScript MCP server ready to run.

No boilerplate. No copy-pasting curl examples. Just tools.

## Why 2 tools, not 500

The obvious approach — one MCP tool per endpoint — falls apart fast. Connect an agent to an API with 200 endpoints and 200 tool schemas land in context before the agent asks anything. Mcpify 10,000 API docs and you've got a context bonfire.

mcpify generates exactly **2 tools** regardless of API size:

| Tool | What it does |
|---|---|
| `{api-name}_search_docs` | Semantic search over all endpoints — returns IDs + param schemas on demand |
| `{api-name}_call_api` | Executes any endpoint by ID with the params you provide |

The tool names are prefixed with the API slug (e.g. `stripe-api_search_docs`, `stripe-api_call_api`), so connecting multiple mcpified APIs to the same agent produces unambiguously named tools — no collisions, no relying on the MCP client to namespace them.

The agent searches first, gets back what it needs, then calls. Context stays clean whether the API has 10 endpoints or 10,000.

## What it does

1. **Parse** your spec (OpenAPI 2/3, Postman, HAR, GraphQL)
2. **Embed** every endpoint with [Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) (local, no API key) and store in SQLite
3. **Generate** a self-contained TypeScript MCP server with `{api-name}_search_docs` + `{api-name}_call_api` and a `.env.example` for auth
4. **Run** the server — your agent finds what it needs, calls it, done

## Supported input formats

| Format | Example |
|---|---|
| OpenAPI 3.x (JSON or YAML) | `openapi.yaml`, `https://api.example.com/openapi.json` |
| Swagger 2.x (JSON or YAML) | `swagger.json` |
| Postman Collection v2.1 | `collection.json` |
| HAR (HTTP Archive) | `captured.har` |
| GraphQL Schema | `schema.graphql` |

## Installation

```bash
npm install -g mcpify
```

Or run directly:

```bash
npx mcpify convert ./openapi.yaml -o ./my-api-mcp
```

## Usage

### CLI

```bash
# Preview what mcpify sees in a spec
mcpify inspect ./openapi.yaml

# Convert — uses local Qwen3 embeddings by default (no API key needed)
mcpify convert ./openapi.yaml -o ./my-api-mcp

# Remote spec
mcpify convert https://petstore3.swagger.io/api/v3/openapi.json -o ./petstore-mcp

# Use OpenAI embeddings instead
mcpify convert ./openapi.yaml -o ./my-api-mcp --embedding-provider openai

# Use Ollama
mcpify convert ./openapi.yaml -o ./my-api-mcp --embedding-provider ollama
```

### As an MCP server (for AI agents)

mcpify itself is an MCP server — so your agent can convert specs without dropping to a shell.

```json
{
  "mcpServers": {
    "mcpify": {
      "command": "npx",
      "args": ["mcpify", "mcp"]
    }
  }
}
```

Tools exposed:

- **`inspect_spec`** — parse a spec and return a summary of endpoints, auth, and format
- **`convert_spec`** — full conversion: parse → embed → index → generate, returns output path and required env vars

### As a library

```typescript
import { parseSpec, generateMcpServer } from 'mcpify';

const spec = await parseSpec('./openapi.yaml');
const result = await generateMcpServer(spec, {
  outputDir: './my-api-mcp',
  // embeddingProvider: 'local' is the default — no API key needed
});

console.log(`Indexed ${result.endpointsIndexed} endpoints`);
console.log('Set these env vars:', result.envVars.map(v => v.name));
```

## Generated server

```
my-api-mcp/
├── src/
│   └── server.ts        # 2-tool MCP server ({api-name}_search_docs + {api-name}_call_api)
├── db/
│   └── api.sqlite       # Pre-built semantic search index
├── .env.example         # Required auth credentials
├── package.json
├── tsconfig.json
└── README.md
```

```bash
cd my-api-mcp
npm install
cp .env.example .env   # fill in your API credentials
npm run dev
```

## Embeddings

Search uses vector similarity. Three providers — the default needs nothing installed.

### Local — Qwen3-Embedding-0.6B (default)

No API key. No external service. ~614MB download on first use, cached to `~/.cache/huggingface/`.

```bash
mcpify convert ./spec.yaml -o ./output
# that's it
```

Uses [`onnx-community/Qwen3-Embedding-0.6B-ONNX`](https://huggingface.co/onnx-community/Qwen3-Embedding-0.6B-ONNX) at `q8` quantization via `@huggingface/transformers`. 1024-dim embeddings, decoder-style model with last-token pooling.

### OpenAI

```bash
export OPENAI_API_KEY=sk-...
mcpify convert ./spec.yaml -o ./output --embedding-provider openai
```

### Ollama

```bash
ollama pull nomic-embed-text
mcpify convert ./spec.yaml -o ./output --embedding-provider ollama
```

## Contributing

Early days — all contributions welcome. Check [issues](https://github.com/bisratttt/mcpify/issues) for what's cooking.

## License

MIT
