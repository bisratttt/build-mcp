import { describe, it, expect } from 'vitest';
import { formatEndpointForSearch } from '../../src/generator/tools.js';
import type { Endpoint } from '../../src/types.js';

function makeEndpoint(overrides: Partial<Endpoint>): Endpoint {
  return {
    id: 'testOp',
    method: 'GET',
    path: '/test',
    tags: [],
    parameters: [],
    responses: [],
    ...overrides,
  };
}

describe('formatEndpointForSearch', () => {
  it('includes endpoint ID, method, and path on first line', () => {
    const result = formatEndpointForSearch(makeEndpoint({ id: 'listPets', method: 'GET', path: '/pets' }));
    expect(result).toContain('listPets (GET /pets)');
  });

  it('includes summary', () => {
    const result = formatEndpointForSearch(makeEndpoint({ summary: 'List all pets' }));
    expect(result).toContain('List all pets');
  });

  it('falls back to description when no summary', () => {
    const result = formatEndpointForSearch(makeEndpoint({ description: 'Returns pets from the store' }));
    expect(result).toContain('Returns pets from the store');
  });

  it('marks deprecated endpoints', () => {
    const result = formatEndpointForSearch(makeEndpoint({ deprecated: true }));
    expect(result).toContain('DEPRECATED');
  });

  it('lists query parameters with type and required status', () => {
    const result = formatEndpointForSearch(makeEndpoint({
      parameters: [
        { name: 'limit', in: 'query', required: false, schema: { type: 'integer' } },
        { name: 'status', in: 'query', required: true, schema: { type: 'string' } },
      ],
    }));
    expect(result).toContain('limit (query, optional, integer)');
    expect(result).toContain('status (query, required, string)');
  });

  it('lists path parameters', () => {
    const result = formatEndpointForSearch(makeEndpoint({
      parameters: [{ name: 'petId', in: 'path', required: true, schema: { type: 'string' } }],
    }));
    expect(result).toContain('petId (path, required, string)');
  });

  it('shows enum values in type description', () => {
    const result = formatEndpointForSearch(makeEndpoint({
      parameters: [{ name: 'status', in: 'query', required: false, schema: { type: 'string', enum: ['active', 'inactive'] } }],
    }));
    expect(result).toContain('string: active|inactive');
  });

  it('omits header parameters from param list', () => {
    const result = formatEndpointForSearch(makeEndpoint({
      parameters: [
        { name: 'X-Custom', in: 'header', required: false },
        { name: 'limit', in: 'query', required: false },
      ],
    }));
    expect(result).not.toContain('X-Custom');
    expect(result).toContain('limit');
  });

  it('shows request body line with content type', () => {
    const result = formatEndpointForSearch(makeEndpoint({
      requestBody: { required: true, contentType: 'application/json', schema: {} },
    }));
    expect(result).toContain('Body: required (application/json)');
  });

  it('shows optional body', () => {
    const result = formatEndpointForSearch(makeEndpoint({
      requestBody: { required: false, contentType: 'application/json' },
    }));
    expect(result).toContain('Body: optional');
  });

  it('includes tags', () => {
    const result = formatEndpointForSearch(makeEndpoint({ tags: ['pets', 'animals'] }));
    expect(result).toContain('pets');
    expect(result).toContain('animals');
  });

  it('omits params/body/tags lines when absent', () => {
    const result = formatEndpointForSearch(makeEndpoint({ summary: 'Simple' }));
    expect(result).not.toContain('Params:');
    expect(result).not.toContain('Body:');
    expect(result).not.toContain('Tags:');
  });
});
