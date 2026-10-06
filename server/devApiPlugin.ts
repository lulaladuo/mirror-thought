import fs from 'node:fs/promises';
import path from 'node:path';
import type { IncomingMessage, ServerResponse } from 'node:http';
import type { Plugin } from 'vite';
import { handleChatPayload, handleHealthPayload, handleSummaryPayload, type EndpointResult } from './jingguanRuntime';

type EndpointHandler = (payload: unknown) => Promise<EndpointResult>;

async function readJsonBody(req: IncomingMessage) {
  const chunks: Buffer[] = [];

  for await (const chunk of req) {
    chunks.push(typeof chunk === 'string' ? Buffer.from(chunk) : chunk);
  }

  const rawBody = Buffer.concat(chunks).toString('utf8').trim();
  if (!rawBody) return {};
  return JSON.parse(rawBody);
}

function sendJson(res: ServerResponse, status: number, body: unknown) {
  res.statusCode = status;
  res.setHeader('content-type', 'application/json; charset=utf-8');
  res.end(JSON.stringify(body));
}

function createMiddleware(handler: EndpointHandler) {
  return async (req: IncomingMessage, res: ServerResponse) => {
    if (req.method !== 'POST') {
      sendJson(res, 405, { error: '只支持 POST 请求。' });
      return;
    }

    try {
      const result = await handler(await readJsonBody(req));
      sendJson(res, result.status, result.body);
    } catch (error) {
      sendJson(res, 400, {
        error: '请求体不是有效 JSON。',
        detail: error instanceof Error ? error.message : undefined,
      });
    }
  };
}

function createHealthMiddleware() {
  return async (req: IncomingMessage, res: ServerResponse) => {
    if (req.method !== 'GET' && req.method !== 'HEAD') {
      sendJson(res, 405, { error: '只支持 GET 请求。' });
      return;
    }

    const result = await handleHealthPayload();
    if (req.method === 'HEAD') {
      res.statusCode = result.status;
      res.end();
      return;
    }

    sendJson(res, result.status, result.body);
  };
}

export function jingguanApiPlugin(): Plugin {
  return {
    name: 'jingguan-api',
    configureServer(server) {
      server.middlewares.use('/api/local-history', async (req, res) => {
        const origin = req.headers.origin;
        if (origin && origin !== `http://${req.headers.host}`) { sendJson(res, 403, { error: '仅限本机镜观页面' }); return; }
        const file = path.resolve('.jingguan-local-history.json');
        try {
          if (req.method === 'GET') {
            const data = await fs.readFile(file, 'utf8').catch((e: NodeJS.ErrnoException) => {
              if (e.code === 'ENOENT') return '{}'; throw e;
            });
            sendJson(res, 200, JSON.parse(data));
          } else if (req.method === 'PUT') {
            let raw = '';
            for await (const chunk of req) {
              raw += chunk.toString();
              if (Buffer.byteLength(raw) > 2000000) { sendJson(res, 413, { error: '历史记录过大' }); return; }
            }
            const value = JSON.parse(raw);
            if (!Array.isArray(value?.conversation?.messages) || !Array.isArray(value?.savedRecords)) {
              sendJson(res, 400, { error: '历史格式不正确' }); return;
            }
            await fs.writeFile(file, JSON.stringify(value), { mode: 0o600 });
            sendJson(res, 200, { ok: true });
          } else sendJson(res, 405, { error: '不支持的方法' });
        } catch { sendJson(res, 500, { error: '本机历史保存失败' }); }
      });
      server.middlewares.use('/api/health' , createHealthMiddleware());
      server.middlewares.use('/api/chat', createMiddleware(handleChatPayload));
      server.middlewares.use('/api/summary', createMiddleware(handleSummaryPayload));
    },
  };
}
