import { readFileSync } from 'fs';
import yaml from 'js-yaml';
import type { NormalizedSpec } from '../types.js';
import { parseOpenApi } from './openapi.js';
import { parsePostman } from './postman.js';
import { parseHar } from './har.js';
import { parseGraphQL } from './graphql.js';

export type { NormalizedSpec };

export async function parseSpec(input: string): Promise<NormalizedSpec> {
  // URL — download and detect
  if (input.startsWith('http://') || input.startsWith('https://')) {
    return parseFromUrl(input);
  }

  const ext = input.split('.').pop()?.toLowerCase();
  const content = readFileSync(input, 'utf-8');

  if (ext === 'graphql' || ext === 'gql') {
    return parseGraphQL(content, 'https://api.example.com/graphql');
  }

  const parsed = ext === 'yaml' || ext === 'yml'
    ? yaml.load(content) as Record<string, unknown>
    : JSON.parse(content) as Record<string, unknown>;

  return detectAndParse(parsed, input);
}

async function parseFromUrl(url: string): Promise<NormalizedSpec> {
  const res = await fetch(url);
  const contentType = res.headers.get('content-type') ?? '';
  const text = await res.text();

  if (contentType.includes('yaml') || url.endsWith('.yaml') || url.endsWith('.yml')) {
    const parsed = yaml.load(text) as Record<string, unknown>;
    return detectAndParse(parsed, url);
  }

  const parsed = JSON.parse(text) as Record<string, unknown>;
  return detectAndParse(parsed, url);
}

function detectAndParse(parsed: Record<string, unknown>, input: string): Promise<NormalizedSpec> {
  // OpenAPI 3.x
  if ('openapi' in parsed) return parseOpenApi(parsed);
  // Swagger 2.x
  if ('swagger' in parsed) return parseOpenApi(parsed);
  // Postman collection
  if ('info' in parsed && 'item' in parsed) return Promise.resolve(parsePostman(parsed));
  // HAR
  if ('log' in parsed && typeof parsed.log === 'object' && parsed.log !== null && 'entries' in (parsed.log as object)) {
    return Promise.resolve(parseHar(parsed));
  }

  throw new Error(`Unrecognized spec format. Supported: OpenAPI (2/3) JSON/YAML, Postman collection, HAR, GraphQL schema.`);
}
