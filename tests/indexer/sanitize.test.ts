import { describe, it, expect, vi } from 'vitest';
import { ApiIndexer } from '../../src/indexer/index.js';
import { tmpdir } from 'os';
import { join } from 'path';
import { randomUUID } from 'crypto';
import type { NormalizedSpec } from '../../src/types.js';

vi.mock('../../src/indexer/embed.js', () => ({
  embed: vi.fn(async () => ({ embedding: new Array(8).fill(0.1), dimensions: 8 })),
  buildEndpointText: vi.fn(() => 'test text'),
}));

describe('sanitizeEndpoint — circular schema handling', () => {
  it('indexes an endpoint whose schema contains circular references', async () => {
    // Simulate what swagger-parser produces after dereferencing Stripe-like specs:
    // schema objects that reference each other circularly
    const circular: Record<string, unknown> = { type: 'object' };
    circular['self'] = circular; // circular reference

    const spec: NormalizedSpec = {
      info: { title: 'Circular Test', version: '1.0.0' },
      servers: [],
      auth: [],
      rawFormat: 'openapi3',
      endpoints: [{
        id: 'createCharge',
        method: 'POST',
        path: '/v1/charges',
        summary: 'Create a charge',
        tags: ['charges'],
        parameters: [
          { name: 'amount', in: 'query', required: true, schema: circular },
        ],
        requestBody: {
          required: true,
          contentType: 'application/x-www-form-urlencoded',
          schema: circular,
        },
        responses: [{ statusCode: '200', schema: circular }],
      }],
    };

    const indexer = new ApiIndexer(join(tmpdir(), `mcpify-circ-${randomUUID()}.sqlite`));
    // Should not throw "Converting circular structure to JSON"
    await expect(indexer.indexSpec(spec, { provider: 'openai' })).resolves.not.toThrow();

    const all = indexer.getAll();
    expect(all).toHaveLength(1);
    expect(all[0].id).toBe('createCharge');
    // Schema should be stripped, parameter structure preserved
    expect(all[0].parameters[0].name).toBe('amount');
    expect(all[0].parameters[0].schema).toBeUndefined();
    expect(all[0].requestBody?.contentType).toBe('application/x-www-form-urlencoded');
    expect(all[0].requestBody?.schema).toBeUndefined();
    expect(all[0].responses).toHaveLength(0);
    indexer.close();
  });
});
