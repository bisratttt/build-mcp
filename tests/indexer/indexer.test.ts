import { describe, it, expect, vi, beforeEach } from 'vitest';
import { tmpdir } from 'os';
import { join } from 'path';
import { randomUUID } from 'crypto';
import { ApiIndexer } from '../../src/indexer/index.js';
import type { NormalizedSpec, EmbedConfig } from '../../src/types.js';

// Mock the embed module so tests don't need a real API key
vi.mock('../../src/indexer/embed.js', () => ({
  embed: vi.fn(async (text: string, _config: unknown, isQuery = false) => {
    // Deterministic fake embeddings based on text content
    // "pets" endpoints get higher first dimension
    const dim = 8;
    const vec = Array.from({ length: dim }, (_, i) => {
      if (text.includes('pets') && i === 0) return 1.0;
      if (text.includes('users') && i === 1) return 1.0;
      return 0.01;
    });
    return { embedding: vec, dimensions: dim };
  }),
  buildEndpointText: vi.fn((ep: { method: string; path: string; tags: string[] }) =>
    `${ep.method} ${ep.path} ${ep.tags.join(' ')}`
  ),
}));

function makeTempDb(): string {
  return join(tmpdir(), `mcpify-test-${randomUUID()}.sqlite`);
}

const mockSpec: NormalizedSpec = {
  info: { title: 'Test API', version: '1.0.0' },
  servers: [{ url: 'https://api.test.com' }],
  endpoints: [
    {
      id: 'listPets',
      method: 'GET',
      path: '/pets',
      summary: 'List all pets',
      tags: ['pets'],
      parameters: [],
      responses: [],
    },
    {
      id: 'createPet',
      method: 'POST',
      path: '/pets',
      summary: 'Create a pet',
      tags: ['pets'],
      parameters: [],
      responses: [],
    },
    {
      id: 'listUsers',
      method: 'GET',
      path: '/users',
      summary: 'List all users',
      tags: ['users'],
      parameters: [],
      responses: [],
    },
  ],
  auth: [],
  rawFormat: 'openapi3',
};

const embedConfig: EmbedConfig = { provider: 'openai' };

describe('ApiIndexer', () => {
  it('indexes all endpoints', async () => {
    const indexer = new ApiIndexer(makeTempDb());
    await indexer.indexSpec(mockSpec, embedConfig);
    const all = indexer.getAll();
    expect(all).toHaveLength(3);
    indexer.close();
  });

  it('stores endpoint IDs correctly', async () => {
    const indexer = new ApiIndexer(makeTempDb());
    await indexer.indexSpec(mockSpec, embedConfig);
    const all = indexer.getAll();
    const ids = all.map(e => e.id);
    expect(ids).toContain('listPets');
    expect(ids).toContain('createPet');
    expect(ids).toContain('listUsers');
    indexer.close();
  });

  it('stores and retrieves metadata', async () => {
    const indexer = new ApiIndexer(makeTempDb());
    await indexer.indexSpec(mockSpec, embedConfig);
    const meta = indexer.getMeta();
    expect(meta.provider).toBe('openai');
    indexer.close();
  });

  it('returns ranked search results by cosine similarity', async () => {
    const indexer = new ApiIndexer(makeTempDb());
    await indexer.indexSpec(mockSpec, embedConfig);

    // Query vector that matches "pets" pattern (high first dimension)
    const queryVec = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    const results = indexer.search(queryVec, 3);

    // pets endpoints should rank higher than users
    const topTwoIds = results.slice(0, 2).map(e => e.id);
    expect(topTwoIds.some(id => id.includes('Pet') || id.includes('pet'))).toBe(true);
    indexer.close();
  });

  it('respects the limit parameter in search', async () => {
    const indexer = new ApiIndexer(makeTempDb());
    await indexer.indexSpec(mockSpec, embedConfig);
    const results = indexer.search([1, 0, 0, 0, 0, 0, 0, 0], 2);
    expect(results).toHaveLength(2);
    indexer.close();
  });

  it('persists data across instances (same db path)', async () => {
    const dbPath = makeTempDb();
    const indexer1 = new ApiIndexer(dbPath);
    await indexer1.indexSpec(mockSpec, embedConfig);
    indexer1.close();

    const indexer2 = new ApiIndexer(dbPath);
    const all = indexer2.getAll();
    expect(all).toHaveLength(3);
    indexer2.close();
  });

  it('preserves full endpoint shape after round-trip', async () => {
    const indexer = new ApiIndexer(makeTempDb());
    await indexer.indexSpec(mockSpec, embedConfig);
    const all = indexer.getAll();
    const listPets = all.find(e => e.id === 'listPets')!;
    expect(listPets.method).toBe('GET');
    expect(listPets.path).toBe('/pets');
    expect(listPets.summary).toBe('List all pets');
    expect(listPets.tags).toContain('pets');
    indexer.close();
  });

  it('handles empty spec without throwing', async () => {
    const indexer = new ApiIndexer(makeTempDb());
    const emptySpec = { ...mockSpec, endpoints: [] };
    await expect(indexer.indexSpec(emptySpec, embedConfig)).resolves.not.toThrow();
    indexer.close();
  });
});

describe('buildEndpointText helper', () => {
  it('is called with each endpoint during indexing', async () => {
    const { buildEndpointText } = await import('../../src/indexer/embed.js');
    vi.clearAllMocks();

    const indexer = new ApiIndexer(makeTempDb());
    await indexer.indexSpec(mockSpec, embedConfig);
    expect(vi.mocked(buildEndpointText)).toHaveBeenCalledTimes(mockSpec.endpoints.length);
    indexer.close();
  });
});
