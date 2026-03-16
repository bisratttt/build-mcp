import { DatabaseSync } from 'node:sqlite';
import type { NormalizedSpec, Endpoint, EmbedConfig } from '../types.js';
import { embed, buildEndpointText } from './embed.js';

/**
 * Strip JSON schemas from an endpoint before storing.
 * Dereferenced specs (e.g. Stripe) have circular $ref objects that
 * JSON.stringify can't handle. call_api only needs structural info —
 * method, path, param names/locations, and body content type.
 */
function sanitizeEndpoint(ep: Endpoint): Endpoint {
  return {
    ...ep,
    parameters: ep.parameters.map(p => ({
      name: p.name,
      in: p.in,
      required: p.required,
      description: p.description,
    })),
    requestBody: ep.requestBody
      ? { required: ep.requestBody.required, contentType: ep.requestBody.contentType }
      : undefined,
    responses: [],
  };
}

function cosineSimilarity(a: number[], b: number[]): number {
  let dot = 0, normA = 0, normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  return dot / (Math.sqrt(normA) * Math.sqrt(normB) || 1);
}

function float32ToBuffer(arr: number[]): Buffer {
  const buf = Buffer.allocUnsafe(arr.length * 4);
  for (let i = 0; i < arr.length; i++) buf.writeFloatLE(arr[i], i * 4);
  return buf;
}

function bufferToFloat32(raw: Uint8Array): number[] {
  const view = new DataView(raw.buffer, raw.byteOffset, raw.byteLength);
  const arr: number[] = [];
  for (let i = 0; i < raw.byteLength; i += 4) arr.push(view.getFloat32(i, true));
  return arr;
}

export class ApiIndexer {
  private db: DatabaseSync;

  constructor(dbPath: string) {
    this.db = new DatabaseSync(dbPath);
    this.db.exec(`
      CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
      CREATE TABLE IF NOT EXISTS endpoints (
        id TEXT PRIMARY KEY,
        method TEXT NOT NULL,
        path TEXT NOT NULL,
        summary TEXT,
        description TEXT,
        tags TEXT,
        endpoint_json TEXT NOT NULL,
        embedding BLOB
      );
    `);
  }

  setMeta(key: string, value: string): void {
    this.db.prepare(`INSERT OR REPLACE INTO meta VALUES (?, ?)`).run(key, value);
  }

  getMeta(): Record<string, string> {
    const rows = this.db.prepare(`SELECT key, value FROM meta`).all() as { key: string; value: string }[];
    return Object.fromEntries(rows.map(r => [r.key, r.value]));
  }

  async indexSpec(spec: NormalizedSpec, embedConfig: EmbedConfig): Promise<void> {
    this.setMeta('provider', embedConfig.provider);
    this.setMeta('model', embedConfig.model ?? '');

    const insertEndpoint = this.db.prepare(`
      INSERT OR REPLACE INTO endpoints (id, method, path, summary, description, tags, endpoint_json, embedding)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    `);

    for (const ep of spec.endpoints) {
      const text = buildEndpointText(ep);
      const { embedding } = await embed(text, embedConfig);
      const embeddingBuf = float32ToBuffer(embedding);

      insertEndpoint.run(
        ep.id, ep.method, ep.path,
        ep.summary ?? null, ep.description ?? null,
        JSON.stringify(ep.tags), JSON.stringify(sanitizeEndpoint(ep)),
        embeddingBuf,
      );
    }
  }

  search(queryEmbedding: number[], limit = 5): Endpoint[] {
    const rows = this.db.prepare(`
      SELECT endpoint_json, embedding FROM endpoints WHERE embedding IS NOT NULL
    `).all() as { endpoint_json: string; embedding: Uint8Array }[];

    const scored = rows.map(r => ({
      endpoint: JSON.parse(r.endpoint_json) as Endpoint,
      score: cosineSimilarity(queryEmbedding, bufferToFloat32(r.embedding)),
    }));

    scored.sort((a, b) => b.score - a.score);
    return scored.slice(0, limit).map(s => s.endpoint);
  }

  getAll(): Endpoint[] {
    const rows = this.db.prepare(`SELECT endpoint_json FROM endpoints`).all() as { endpoint_json: string }[];
    return rows.map(r => JSON.parse(r.endpoint_json) as Endpoint);
  }

  close(): void {
    this.db.close();
  }
}
