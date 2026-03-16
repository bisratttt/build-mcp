import SwaggerParser from '@apidevtools/swagger-parser';
import type { OpenAPIV2, OpenAPIV3, OpenAPIV3_1 } from 'openapi-types';
import type {
  NormalizedSpec, Endpoint, Parameter, RequestBody,
  AuthScheme, HttpMethod, JsonSchema, SpecFormat,
} from '../types.js';

type V3Doc = OpenAPIV3.Document | OpenAPIV3_1.Document;
type V2Doc = OpenAPIV2.Document;

function slug(str: string): string {
  return str.replace(/[^a-zA-Z0-9]/g, '_').replace(/_+/g, '_').replace(/^_|_$/g, '');
}

function toEnvVar(apiTitle: string, name: string): string {
  return `${slug(apiTitle).toUpperCase()}_${slug(name).toUpperCase()}`;
}

export async function parseOpenApi(input: string | Record<string, unknown>): Promise<NormalizedSpec> {
  // Pass parsed objects directly — swagger-parser uses file extension to determine
  // parse mode, so paths with non-standard extensions (.txt, .bin, etc.) must be
  // pre-parsed and passed as an object to avoid extension-based failures.
  const api = await SwaggerParser.dereference(input as Parameters<typeof SwaggerParser.dereference>[0]) as V2Doc | V3Doc;
  const isV2 = 'swagger' in api;

  return isV2
    ? normalizeV2(api as V2Doc)
    : normalizeV3(api as V3Doc);
}

// ── OpenAPI 3.x ─────────────────────────────────────────────────────────────

function normalizeV3(api: V3Doc): NormalizedSpec {
  const endpoints: Endpoint[] = [];
  const paths = api.paths ?? {};

  for (const [path, pathItem] of Object.entries(paths)) {
    if (!pathItem) continue;
    const methods: HttpMethod[] = ['GET', 'POST', 'PUT', 'PATCH', 'DELETE', 'HEAD', 'OPTIONS'];
    for (const method of methods) {
      const op = pathItem[method.toLowerCase() as keyof typeof pathItem] as OpenAPIV3.OperationObject | undefined;
      if (!op) continue;

      const opId = op.operationId ?? `${method.toLowerCase()}_${slug(path)}`;
      endpoints.push({
        id: opId,
        operationId: op.operationId,
        method,
        path,
        summary: op.summary,
        description: op.description,
        tags: op.tags ?? [],
        deprecated: op.deprecated,
        parameters: normalizeV3Params(op.parameters as OpenAPIV3.ParameterObject[] ?? []),
        requestBody: normalizeV3Body(op.requestBody as OpenAPIV3.RequestBodyObject | undefined),
        responses: normalizeV3Responses(op.responses ?? {}),
        security: op.security?.flatMap(s => Object.keys(s)),
      });
    }
  }

  const auth = normalizeV3Auth(api as OpenAPIV3.Document, api.info.title);

  const servers = (api as OpenAPIV3.Document).servers?.map(s => ({
    url: s.url,
    description: s.description,
  })) ?? [];

  return {
    info: { title: api.info.title, version: api.info.version, description: api.info.description },
    servers,
    endpoints,
    auth,
    rawFormat: 'openapi3',
  };
}

function normalizeV3Params(params: OpenAPIV3.ParameterObject[]): Parameter[] {
  return params.map(p => ({
    name: p.name,
    in: p.in as Parameter['in'],
    required: p.required ?? false,
    description: p.description,
    schema: p.schema as JsonSchema | undefined,
  }));
}

function normalizeV3Body(body?: OpenAPIV3.RequestBodyObject): RequestBody | undefined {
  if (!body) return undefined;
  const [contentType, media] = Object.entries(body.content ?? {})[0] ?? [];
  if (!contentType) return undefined;
  return {
    required: body.required ?? false,
    description: body.description,
    contentType,
    schema: (media as OpenAPIV3.MediaTypeObject)?.schema as JsonSchema | undefined,
  };
}

function normalizeV3Responses(responses: OpenAPIV3.ResponsesObject): Array<{ statusCode: string; description?: string; schema?: JsonSchema }> {
  return Object.entries(responses).map(([code, resp]) => {
    const r = resp as OpenAPIV3.ResponseObject;
    const [, media] = Object.entries(r.content ?? {})[0] ?? [];
    const description = 'description' in r ? (r as { description?: string }).description : undefined;
    return {
      statusCode: code,
      description,
      schema: (media as OpenAPIV3.MediaTypeObject)?.schema as JsonSchema | undefined,
    };
  });
}

function normalizeV3Auth(api: OpenAPIV3.Document, title: string): AuthScheme[] {
  const schemes = api.components?.securitySchemes ?? {};
  return Object.entries(schemes).map(([name, scheme]) => {
    const s = scheme as OpenAPIV3.SecuritySchemeObject;
    if (s.type === 'apiKey') {
      return { name, type: 'apiKey', in: s.in as AuthScheme['in'], description: s.description, envVar: toEnvVar(title, name) };
    }
    if (s.type === 'http') {
      return { name, type: 'http', scheme: s.scheme, description: s.description, envVar: toEnvVar(title, name) };
    }
    if (s.type === 'oauth2') {
      return { name, type: 'oauth2', description: s.description, envVar: toEnvVar(title, `${name}_ACCESS_TOKEN`) };
    }
    return { name, type: 'openIdConnect', description: (s as OpenAPIV3.OpenIdSecurityScheme).openIdConnectUrl, envVar: toEnvVar(title, name) };
  });
}

// ── OpenAPI 2.x (Swagger) ────────────────────────────────────────────────────

function normalizeV2(api: V2Doc): NormalizedSpec {
  const endpoints: Endpoint[] = [];

  for (const [path, pathItem] of Object.entries(api.paths ?? {})) {
    const methods: HttpMethod[] = ['GET', 'POST', 'PUT', 'PATCH', 'DELETE', 'HEAD', 'OPTIONS'];
    for (const method of methods) {
      const op = pathItem[method.toLowerCase() as keyof typeof pathItem] as OpenAPIV2.OperationObject | undefined;
      if (!op) continue;

      const opId = op.operationId ?? `${method.toLowerCase()}_${slug(path)}`;
      const params = (op.parameters ?? []) as OpenAPIV2.Parameter[];
      const bodyParam = params.find(p => p.in === 'body') as OpenAPIV2.InBodyParameterObject | undefined;
      const nonBodyParams = params.filter(p => p.in !== 'body') as OpenAPIV2.GeneralParameterObject[];

      endpoints.push({
        id: opId,
        operationId: op.operationId,
        method,
        path,
        summary: op.summary,
        description: op.description,
        tags: op.tags ?? [],
        deprecated: op.deprecated,
        parameters: nonBodyParams.map(p => ({
          name: p.name,
          in: p.in as Parameter['in'],
          required: p.required ?? false,
          description: p.description,
          schema: { type: p.type, format: p.format, items: p.items } as JsonSchema,
        })),
        requestBody: bodyParam ? {
          required: bodyParam.required ?? false,
          description: bodyParam.description,
          contentType: 'application/json',
          schema: bodyParam.schema as JsonSchema | undefined,
        } : undefined,
        responses: Object.entries(op.responses ?? {}).map(([code, resp]) => {
          const r = resp as unknown as Record<string, unknown>;
          return { statusCode: code, description: typeof r?.description === 'string' ? r.description : undefined };
        }),
        security: op.security?.flatMap(s => Object.keys(s)),
      });
    }
  }

  const auth = normalizeV2Auth(api);
  const baseUrl = `${api.schemes?.[0] ?? 'https'}://${api.host ?? 'localhost'}${api.basePath ?? '/'}`;

  return {
    info: { title: api.info.title, version: api.info.version, description: api.info.description },
    servers: [{ url: baseUrl }],
    endpoints,
    auth,
    rawFormat: 'openapi2',
  };
}

function normalizeV2Auth(api: V2Doc): AuthScheme[] {
  const defs = api.securityDefinitions ?? {};
  return Object.entries(defs).map(([name, def]) => {
    if (def.type === 'apiKey') {
      return { name, type: 'apiKey', in: def.in as AuthScheme['in'], description: def.description, envVar: toEnvVar(api.info.title, name) };
    }
    if (def.type === 'basic') {
      return { name, type: 'http', scheme: 'basic', description: def.description, envVar: toEnvVar(api.info.title, name) };
    }
    // oauth2
    return { name, type: 'oauth2', description: def.description, envVar: toEnvVar(api.info.title, `${name}_ACCESS_TOKEN`) };
  });
}
