/**
 * Integration test for the local Qwen3-Embedding-0.6B provider.
 * Downloads ~614MB on first run (cached to ~/.cache/huggingface/ afterwards).
 *
 * Run with: npm run test:integration
 */
import { describe, it, expect, beforeAll } from 'vitest';
import { embed, QWEN3_DIMS } from '../../src/indexer/embed.js';
import type { EmbedConfig } from '../../src/types.js';

const config: EmbedConfig = { provider: 'local' };

function cosineSimilarity(a: number[], b: number[]): number {
  let dot = 0, normA = 0, normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i]; normA += a[i] * a[i]; normB += b[i] * b[i];
  }
  return dot / (Math.sqrt(normA) * Math.sqrt(normB) || 1);
}

// Warm up the pipeline once before all tests
beforeAll(async () => {
  console.log('Loading Qwen3-Embedding-0.6B (downloads ~614MB on first run)...');
  await embed('warmup', config);
  console.log('Model ready.');
});

describe('local embed — output shape', () => {
  it('returns a number array for document text', async () => {
    const result = await embed('GET /pets — List all pets', config);
    expect(Array.isArray(result.embedding)).toBe(true);
    expect(result.embedding.every(n => typeof n === 'number')).toBe(true);
  });

  it(`returns ${QWEN3_DIMS}-dimensional embeddings`, async () => {
    const result = await embed('POST /users — Create a user', config);
    expect(result.dimensions).toBe(QWEN3_DIMS);
    expect(result.embedding).toHaveLength(QWEN3_DIMS);
  });

  it('returns unit-norm vectors (normalize: true)', async () => {
    const result = await embed('GET /orders', config);
    const norm = Math.sqrt(result.embedding.reduce((s, x) => s + x * x, 0));
    expect(norm).toBeCloseTo(1.0, 3);
  });

  it('returns same dimensions for query mode', async () => {
    const result = await embed('list all pets', config, true);
    expect(result.dimensions).toBe(QWEN3_DIMS);
  });
});

describe('local embed — semantic quality', () => {
  it('similar endpoints score higher than dissimilar ones', async () => {
    const [docA, docB, docC] = await Promise.all([
      embed('GET /pets — List all pets in the store', config),
      embed('GET /animals — Retrieve all animals', config),
      embed('POST /payments/charge — Charge a credit card', config),
    ]);

    const simAB = cosineSimilarity(docA.embedding, docB.embedding);
    const simAC = cosineSimilarity(docA.embedding, docC.embedding);

    console.log(`pets↔animals similarity:  ${simAB.toFixed(4)}`);
    console.log(`pets↔payments similarity: ${simAC.toFixed(4)}`);

    expect(simAB).toBeGreaterThan(simAC);
  });

  it('query embedding is closer to matching doc than unrelated doc', async () => {
    const [query, matchingDoc, unrelatedDoc] = await Promise.all([
      embed('how do I list pets', config, true),
      embed('GET /pets — List all pets', config),
      embed('POST /invoices — Create a billing invoice', config),
    ]);

    const simMatch = cosineSimilarity(query.embedding, matchingDoc.embedding);
    const simUnrelated = cosineSimilarity(query.embedding, unrelatedDoc.embedding);

    console.log(`query↔matching doc similarity:   ${simMatch.toFixed(4)}`);
    console.log(`query↔unrelated doc similarity:  ${simUnrelated.toFixed(4)}`);

    expect(simMatch).toBeGreaterThan(simUnrelated);
  });

  it('document mode and query mode produce different vectors for same text', async () => {
    const text = 'list all users';
    const [asDoc, asQuery] = await Promise.all([
      embed(text, config, false),
      embed(text, config, true),
    ]);

    const sim = cosineSimilarity(asDoc.embedding, asQuery.embedding);
    // Should be similar but not identical — instruct prefix shifts the vector
    expect(sim).toBeLessThan(1.0);
    console.log(`doc vs query similarity for same text: ${sim.toFixed(4)}`);
  });
});
