import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import type { ChatMessage, ChatResponse, ResponseChoice, ResponseMode, ResponsePhase, ResponseType, RiskLevel } from '../src/types/chat';

type AnthropicRole = 'user' | 'assistant';

interface AnthropicMessage {
  role: AnthropicRole;
  content: string;
}

interface RuntimeRequest {
  mode: 'chat' | 'summary';
  text?: string;
  history: ChatMessage[];
  detectedBeliefs: string[];
  detectedTensions: string[];
  detectedAssumptions: string[];
  unclearConcepts: string[];
  currentUserTurnCount: number;
}

interface EndpointErrorBody {
  error: string;
  detail?: string;
}

interface EndpointHealthBody {
  ok: boolean;
  service: 'jingguan-api';
  mode: 'anthropic' | 'qwen' | 'mock';
  anthropic_configured: boolean;
  has_api_key: boolean;
  has_model: boolean;
  model: string | null;
  prompt_loaded: boolean;
  api_url: string;
  api_version: string;
  max_tokens: number;
  commit_sha: string | null;
  timestamp: string;
}

export interface EndpointResult {
  status: number;
  body: ChatResponse | EndpointErrorBody | EndpointHealthBody;
}

const PROJECT_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const SYSTEM_PROMPT_PATH = path.join(PROJECT_ROOT, 'prompts', 'jingguan-system-prompt.md');
const DEFAULT_ANTHROPIC_API_URL = 'https://api.anthropic.com/v1/messages';
const DEFAULT_ANTHROPIC_VERSION = '2023-06-01';
const DEFAULT_MAX_TOKENS = 1200;
const MAX_HISTORY_MESSAGES = 16;
const REPAIR_TEXT_LIMIT = 8000;
const ALLOW_SUMMARY_TURNS = 5;
const SUGGEST_SUMMARY_TURNS = 8;
const FORCE_SUMMARY_TURNS = 9;

const responseTypes = new Set<ResponseType>(['normal', 'summary', 'crisis']);
const riskLevels = new Set<RiskLevel>(['none', 'low', 'high']);
const phases = new Set<ResponsePhase>(['intake', 'mapping', 'clarification', 'tension_analysis', 'summary', 'crisis']);
const responseModes = new Set<ResponseMode>(['free_text', 'choice']);
const RESPONSE_TOOL_NAME = 'format_jingguan_response';

const responseTool = {
  name: RESPONSE_TOOL_NAME,
  description: 'Return the Jingguan thought-analysis response in the exact structured format consumed by the product.',
  input_schema: {
    type: 'object',
    additionalProperties: false,
    required: [
      'response_type',
      'risk_level',
      'phase',
      'message',
      'mapping',
      'question',
      'response_mode',
      'choices',
      'allow_free_text',
      'detected_beliefs',
      'detected_tensions',
      'detected_assumptions',
      'unclear_concepts',
      'can_summarize',
      'should_summarize',
    ],
    properties: {
      response_type: {
        type: 'string',
        enum: ['normal', 'summary', 'crisis'],
      },
      risk_level: {
        type: 'string',
        enum: ['none', 'low', 'high'],
      },
      phase: {
        type: 'string',
        enum: ['intake', 'mapping', 'clarification', 'tension_analysis', 'summary', 'crisis'],
      },
      message: {
        type: 'string',
        minLength: 1,
      },
      mapping: {
        type: ['string', 'null'],
      },
      question: {
        type: ['string', 'null'],
      },
      response_mode: {
        type: 'string',
        enum: ['free_text', 'choice'],
      },
      choices: {
        type: 'array',
        maxItems: 4,
        items: {
          type: 'object',
          additionalProperties: false,
          required: ['id', 'label', 'description', 'meaning', 'client_followup', 'requires_api_after_choice'],
          properties: {
            id: {
              type: 'string',
              minLength: 1,
            },
            label: {
              type: 'string',
              minLength: 1,
            },
            description: {
              type: 'string',
              minLength: 1,
            },
            meaning: {
              type: 'string',
              minLength: 1,
            },
            client_followup: {
              type: ['string', 'null'],
            },
            requires_api_after_choice: {
              type: 'boolean',
            },
          },
        },
      },
      allow_free_text: {
        type: 'boolean',
      },
      detected_beliefs: {
        type: 'array',
        maxItems: 5,
        items: { type: 'string', minLength: 1 },
      },
      detected_tensions: {
        type: 'array',
        maxItems: 4,
        items: { type: 'string', minLength: 1 },
      },
      detected_assumptions: {
        type: 'array',
        maxItems: 5,
        items: { type: 'string', minLength: 1 },
      },
      unclear_concepts: {
        type: 'array',
        maxItems: 5,
        items: { type: 'string', minLength: 1 },
      },
      can_summarize: {
        type: 'boolean',
      },
      should_summarize: {
        type: 'boolean',
      },
    },
  },
};

// 提示词更新后由本地服务重载，供团队逐次验收。
let systemPromptPromise: Promise<string> | null = null;

class PublicEndpointError extends Error {
  status: number;
  detail?: string;

  constructor(status: number, message: string, detail?: string) {
    super(message);
    this.status = status;
    this.detail = detail;
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function readStringArray(value: unknown, maxItems: number) {
  if (!Array.isArray(value)) return [];
  return value
    .filter((item): item is string => typeof item === 'string')
    .map((item) => item.trim())
    .filter(Boolean)
    .slice(0, maxItems);
}

function isReadableChatMessage(item: unknown): item is ChatMessage {
  return (
    isRecord(item) &&
    (item.role === 'user' || item.role === 'assistant') &&
    typeof item.content === 'string' &&
    item.content.trim().length > 0
  );
}

function readHistory(value: unknown) {
  if (!Array.isArray(value)) return [];

  return value
    .filter(isReadableChatMessage)
    .slice(-MAX_HISTORY_MESSAGES);
}

function countHistoryUserTurns(value: unknown) {
  if (!Array.isArray(value)) return 0;
  return value.filter((item) => isReadableChatMessage(item) && item.role === 'user').length;
}

function readCurrentUserTurnCount(payload: Record<string, unknown>, mode: 'chat' | 'summary') {
  const explicitTurnCount = payload.turnCount;
  if (typeof explicitTurnCount === 'number' && Number.isFinite(explicitTurnCount) && explicitTurnCount >= 0) {
    return Math.floor(explicitTurnCount);
  }

  return countHistoryUserTurns(payload.history) + (mode === 'chat' ? 1 : 0);
}

function endpointError(error: unknown): EndpointResult {
  if (error instanceof PublicEndpointError) {
    return {
      status: error.status,
      body: {
        error: error.message,
        detail: error.detail,
      },
    };
  }

  console.error(error);
  return {
    status: 500,
    body: {
      error: '服务端生成回应时出现问题。',
    },
  };
}

function getRequiredText(body: Record<string, unknown>, key: string) {
  const value = body[key];
  return typeof value === 'string' ? value.trim() : '';
}

function getRuntimeRequest(payload: unknown, mode: 'chat' | 'summary'): RuntimeRequest {
  if (!isRecord(payload)) {
    throw new PublicEndpointError(400, '请求体必须是 JSON object。');
  }

  const text = getRequiredText(payload, 'text');
  if (mode === 'chat' && !text) {
    throw new PublicEndpointError(400, '缺少本轮用户输入。');
  }

  return {
    mode,
    text,
    history: readHistory(payload.history),
    detectedBeliefs: readStringArray(payload.detectedBeliefs, 5),
    detectedTensions: readStringArray(payload.detectedTensions, 4),
    detectedAssumptions: readStringArray(payload.detectedAssumptions, 5),
    unclearConcepts: readStringArray(payload.unclearConcepts, 5),
    currentUserTurnCount: readCurrentUserTurnCount(payload, mode),
  };
}

function readConfigValues() {
  const provider: 'qwen' | 'anthropic' = process.env.JINGGUAN_PROVIDER === 'qwen' ? 'qwen' : 'anthropic';
  const apiKey = (provider === 'qwen' ? process.env.DASHSCOPE_API_KEY : process.env.ANTHROPIC_API_KEY)?.trim();
  const model = provider === 'qwen' ? (process.env.QWEN_MODEL?.trim() || 'qwen-plus') : process.env.ANTHROPIC_MODEL?.trim();
  const apiUrl = provider === 'qwen' ? (process.env.QWEN_API_URL?.trim() || 'https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions') : (process.env.ANTHROPIC_MESSAGES_URL?.trim() || DEFAULT_ANTHROPIC_API_URL);
  const apiVersion = process.env.ANTHROPIC_VERSION?.trim() || DEFAULT_ANTHROPIC_VERSION;
  const betaHeaders = process.env.ANTHROPIC_BETA_HEADERS?.trim();
  const maxTokensValue = Number(process.env.ANTHROPIC_MAX_TOKENS ?? DEFAULT_MAX_TOKENS);
  const maxTokens = Number.isFinite(maxTokensValue) && maxTokensValue > 0 ? Math.floor(maxTokensValue) : DEFAULT_MAX_TOKENS;

  return {
    provider,
    apiKey,
    model,
    apiUrl,
    apiVersion,
    betaHeaders,
    maxTokens,
  };
}

function getConfig() {
  const config = readConfigValues();
  const { apiKey, model } = config;

  if (!apiKey || !model) {
    const missing = [apiKey ? null : (config.provider === 'qwen' ? 'DASHSCOPE_API_KEY' : 'ANTHROPIC_API_KEY'), model ? null : 'ANTHROPIC_MODEL'].filter(Boolean);
    throw new PublicEndpointError(501, `模型 API 尚未配置：缺少 ${missing.join(', ')}。`);
  }

  return {
    ...config,
    apiKey,
    model,
  };
}

async function isSystemPromptReadable() {
  try {
    await fs.access(SYSTEM_PROMPT_PATH);
    return true;
  } catch {
    return false;
  }
}

export async function handleHealthPayload(): Promise<EndpointResult> {
  const config = readConfigValues();
  const mode = process.env.VITE_JINGGUAN_API_MODE?.trim() === 'mock' ? 'mock' : config.provider;
  const promptLoaded = await isSystemPromptReadable();
  const anthropicConfigured = Boolean(config.apiKey && config.model);

  return {
    status: 200,
    body: {
      ok: mode === 'mock' || (anthropicConfigured && promptLoaded),
      service: 'jingguan-api',
      mode,
      anthropic_configured: config.provider === 'anthropic' && anthropicConfigured,
      has_api_key: Boolean(config.apiKey),
      has_model: Boolean(config.model),
      model: config.model || null,
      prompt_loaded: promptLoaded,
      api_url: config.apiUrl,
      api_version: config.apiVersion,
      max_tokens: config.maxTokens,
      commit_sha: process.env.VERCEL_GIT_COMMIT_SHA ?? null,
      timestamp: new Date().toISOString(),
    },
  };
}

async function getSystemPrompt() {
  systemPromptPromise = fs.readFile(SYSTEM_PROMPT_PATH, 'utf8').then((prompt) => {
    return [
      prompt.trim(),
      '',
      '## 运行时补充',
      '',
      '- 你会收到当前会话的结构化状态，previous_detected_beliefs / previous_detected_tensions 只作为暂定上下文。',
      '- previous_detected_assumptions / previous_unclear_concepts 是本次会话已浮现的来访者预设和待澄清概念，第二轮之后必须优先检查它们是否才是当前困惑的卡点。',
      '- 如果本轮任务是 summary，response_type 必须为 "summary"，phase 必须为 "summary"，question 必须为 null。',
      '- 如果本轮任务是 chat，普通回应最多一个核心追问；用户请求整理或暂停时不得追加问题。',
      '- 面向普通用户使用自然、清楚而有思考力度的中文：保留必要的哲学概念，但首次出现时用上下文解释，不堆叠学术术语，也不要把分析写成过度口语化的安慰话。',
      '- 当 current_user_turn_count >= 9 时，本轮必须强制收束为阶段性总结：response_type 为 "summary"，phase 为 "summary"，question 为 null，不要继续追问。',
      '- 当 current_user_turn_count >= 8 且尚未强制总结时，不要开启新的细枝追问；请做收束性映射，并只询问是否先生成阶段性小结。',
      '- 第一轮或用户表达仍不清楚时，优先使用 response_mode "choice"，给 2-4 个澄清选项，并保留自由输入。',
      '- choices 必须是 JSON array，不要把数组序列化成字符串。',
      '- 如果选项点击后只需要用户补充一句，不需要立即再次调用模型，则该选项 requires_api_after_choice 设为 false，并写入 client_followup。',
      '- 你必须调用 format_jingguan_response 工具返回结构化结果，不要输出自由文本。',
    ].join('\n');
  });
  return systemPromptPromise;
}

function messageText(message: ChatMessage) {
  if (message.responseType === 'normal') {
    return [message.mapping, message.question].filter(Boolean).join('\n\n') || message.content;
  }
  return message.content;
}

function mergeConsecutiveMessages(messages: AnthropicMessage[]) {
  const merged: AnthropicMessage[] = [];

  for (const message of messages) {
    const content = message.content.trim();
    if (!content) continue;

    const previous = merged.at(-1);
    if (previous?.role === message.role) {
      previous.content = `${previous.content}\n\n${content}`;
    } else {
      merged.push({ role: message.role, content });
    }
  }

  return merged;
}

function countUserTurns(request: RuntimeRequest) {
  return request.currentUserTurnCount;
}

function shouldForceSummary(request: RuntimeRequest) {
  const alreadyHasSummary = request.history.some((message) => message.role === 'assistant' && message.responseType === 'summary');
  return request.mode === 'chat' && countUserTurns(request) >= FORCE_SUMMARY_TURNS && !alreadyHasSummary;
}

function buildRuntimeTask(request: RuntimeRequest) {
  const userTurnCount = countUserTurns(request);
  const state = {
    product: '镜观',
    mode: request.mode,
    current_user_turn_count: userTurnCount,
    summary_policy: {
      allow_after_effective_user_turns: ALLOW_SUMMARY_TURNS,
      suggest_after_effective_user_turns: SUGGEST_SUMMARY_TURNS,
      force_summary_at_user_turn: FORCE_SUMMARY_TURNS,
      can_summarize_meaning:
      '材料足够生成阶段性结构整理；第 8 轮应主动收束提醒，第 9 轮首次达到时必须生成阶段性小结，不表示已有结论；小结后允许继续对话。',
    },
    previous_detected_beliefs: request.detectedBeliefs,
    previous_detected_tensions: request.detectedTensions,
    previous_detected_assumptions: request.detectedAssumptions,
    previous_unclear_concepts: request.unclearConcepts,
  };

  if (request.mode === 'summary') {
    return [
      '【运行时状态】',
      JSON.stringify(state, null, 2),
      '',
      '【本轮任务】',
      '请立即整理当前会话，不受5轮门槛限制。message 必须含三个分段标题：已经明确的、还不确定的、暂时停在这里。已明确的只写用户亲自表达或确认的事实、感受与疑问，保留可能/不确定；还不确定的写待检验联系，不能登记成确定立场。材料不足也如实说明。response_type=summary，phase=summary，question=null，mapping=null，choices=[]，不要邀请回答，不要给建议。',
      '必须只输出符合 system prompt schema 的 JSON object。',
    ].join('\n');
  }

  return [
    '【运行时状态】',
    JSON.stringify(state, null, 2),
    '',
    '【本轮任务】',
    shouldForceSummary(request)
      ? '本轮已经达到强制收束轮次。请生成阶段性思想分析小结，response_type 必须为 "summary"，phase 必须为 "summary"，question 必须为 null，response_mode 必须为 "free_text"，choices 必须为空数组。不要继续追问，不要给建议、行动方案、安慰或结论。'
      : userTurnCount >= SUGGEST_SUMMARY_TURNS
        ? '本轮已经接近收束。请不要开启新的细枝追问；请优先整理已经显出来的来访者预设、核心信念和张力，并用一个收束性问题询问是否先生成阶段性小结。必须只输出符合 system prompt schema 的 JSON object。'
        : '请先回看历史中已确认的内容及重复追问，处理用户本轮输入；已有核心张力时优先检验判断之间的理由联系，不默认询问词语定义，同一概念澄清不得超过3轮。未确认前提只能作为待检验问题，不能当成用户立场。第一轮先确认困惑，最后给出一个能带来新增理解且容易回答的小范围追问，不要求完整定义或完整标准；用户答不上来时优先用已有材料做可修正的情境对比，不默认索要真实事件或最早记忆；连续两次卡住时先整理已知与未知，允许暂停，普通回复可以不提问。问句必须短而直接，禁止“是否完全不可想象”等双重否定和需要先想出解决方法的假设。必须只输出符合 system prompt schema 的 JSON object。',
    '',
    '【用户本轮输入】',
    request.text,
  ].join('\n');
}

function buildMessages(request: RuntimeRequest) {
  const historyMessages: AnthropicMessage[] = request.history.map((message) => ({
    role: message.role,
    content: messageText(message),
  }));

  const messages = mergeConsecutiveMessages([
    ...historyMessages,
    {
      role: 'user',
      content: buildRuntimeTask(request),
    },
  ]);

  if (messages[0]?.role === 'assistant') {
    messages.unshift({
      role: 'user',
      content: '以下是当前会话中已有的分析内容，仅作为上下文，请继续遵守镜观运行时边界。',
    });
  }

  return messages;
}

async function callAnthropic(system: string, messages: AnthropicMessage[]) {
  const config = getConfig();
  if (config.provider === 'qwen') {
    const response = await fetch(config.apiUrl, {
      method: 'POST',
      headers: { 'content-type': 'application/json', authorization: `Bearer ${config.apiKey}` },
      signal: AbortSignal.timeout(60000),
      body: JSON.stringify({
        model: config.model,
        messages: [{ role: 'system', content: system + '\n仅输出 JSON 对象，必须符合以下格式：\n' + JSON.stringify(responseTool.input_schema) }, ...messages],
        enable_thinking: false,
        // 固定采样温度，减少同一会话中措辞和选项结构的无意义波动。
        temperature: 0,
        response_format: { type: 'json_object' },
        max_tokens: config.maxTokens,
      }),
    });
    const value: unknown = await response.json();
    if (!response.ok) {
      const error = isRecord(value) && isRecord(value.error) ? value.error : {};
      throw new PublicEndpointError(response.status >= 500 ? 502 : response.status, '千问 API 请求失败。', typeof error.code === 'string' ? error.code : '请检查密钥、权限和额度。');
    }
    if (!isRecord(value) || !Array.isArray(value.choices)) throw new PublicEndpointError(502, '千问返回格式异常。');
    const first = value.choices[0];
    if (!isRecord(first) || !isRecord(first.message) || typeof first.message.content !== 'string') throw new PublicEndpointError(502, '千问未返回回答。');
    return { content: [{ type: 'text', text: first.message.content }] };
  }
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    'x-api-key': config.apiKey,
    'anthropic-version': config.apiVersion,
  };

  if (config.betaHeaders) {
    headers['anthropic-beta'] = config.betaHeaders;
  }

  const response = await fetch(config.apiUrl, {
    method: 'POST',
    headers,
    body: JSON.stringify({
      model: config.model,
      max_tokens: config.maxTokens,
      system,
      messages,
      tools: [responseTool],
      tool_choice: {
        type: 'tool',
        name: RESPONSE_TOOL_NAME,
      },
    }),
  });

  const text = await response.text();
  if (!response.ok) {
    throw new PublicEndpointError(
      response.status >= 500 ? 502 : response.status,
      'Claude API 请求失败。',
      text.slice(0, 700),
    );
  }

  try {
    return JSON.parse(text);
  } catch {
    throw new PublicEndpointError(502, 'Claude API 返回了不可解析的 JSON。', text.slice(0, 700));
  }
}

function extractAssistantText(value: unknown) {
  if (!isRecord(value) || !Array.isArray(value.content)) {
    throw new Error('Anthropic response is missing content array');
  }

  const text = value.content
    .map((block) => (isRecord(block) && block.type === 'text' && typeof block.text === 'string' ? block.text : ''))
    .join('\n')
    .trim();

  if (!text) {
    throw new Error('Anthropic response did not contain text');
  }

  return text;
}

function extractAssistantPayload(value: unknown) {
  if (!isRecord(value) || !Array.isArray(value.content)) {
    throw new Error('Anthropic response is missing content array');
  }

  const toolUse = value.content.find((block) => {
    return isRecord(block) && block.type === 'tool_use' && block.name === RESPONSE_TOOL_NAME && isRecord(block.input);
  });

  if (isRecord(toolUse) && isRecord(toolUse.input)) {
    return toolUse.input;
  }

  return parseJsonObject(extractAssistantText(value));
}

function stringifyAssistantPayload(value: unknown) {
  try {
    return JSON.stringify(extractAssistantPayload(value), null, 2);
  } catch {
    try {
      return extractAssistantText(value);
    } catch {
      return JSON.stringify(value, null, 2);
    }
  }
}

function parseJsonObject(text: string) {
  const trimmed = text.trim();
  const withoutFence = trimmed
    .replace(/^```(?:json)?\s*/i, '')
    .replace(/\s*```$/i, '')
    .trim();
  const firstBrace = withoutFence.indexOf('{');
  const lastBrace = withoutFence.lastIndexOf('}');

  if (firstBrace < 0 || lastBrace < firstBrace) {
    throw new Error('model output is not a JSON object');
  }

  return JSON.parse(withoutFence.slice(firstBrace, lastBrace + 1));
}

function assertEnum<T extends string>(value: unknown, allowed: Set<T>, field: string): T {
  if (typeof value === 'string' && allowed.has(value as T)) {
    return value as T;
  }
  throw new Error(`${field} must be one of ${[...allowed].join(', ')}`);
}

function assertNullableString(value: unknown, field: string) {
  if (value === null || typeof value === 'string') return value;
  throw new Error(`${field} must be string or null`);
}

function assertBoolean(value: unknown, field: string) {
  if (typeof value === 'boolean') return value;
  throw new Error(`${field} must be boolean`);
}

function readBoolean(value: unknown, fallback: boolean) {
  return typeof value === 'boolean' ? value : fallback;
}

function stripLooseQuotes(value: string): string {
  let item = value.trim().replace(/\\"/g, '"');
  if (item.length >= 2 && item.startsWith('"') && item.endsWith('"')) {
    item = item.slice(1, -1).trim();
  }
  return item;
}

function parseLooseStringArray(value: string): string[] | null {
  const trimmed = stripLooseQuotes(value);
  if (!trimmed.startsWith('[') || !trimmed.endsWith(']')) return null;

  const inner = trimmed.slice(1, -1).trim();
  if (!inner) return [];

  return inner
    .split(/,\s*(?=")/)
    .map(stripLooseQuotes)
    .filter(Boolean);
}

function parseSerializedStringArray(value: string, field: string, maxItems: number): string[] {
  const item = value.trim();
  if (!item) return [];

  if ((item.startsWith('[') && item.endsWith(']')) || (item.startsWith('"') && item.includes('['))) {
    try {
      const parsed = JSON.parse(item);
      return assertStringArray(parsed, field, maxItems);
    } catch {
      const looseItems = parseLooseStringArray(item);
      if (looseItems) return looseItems.slice(0, maxItems);
    }
  }

  return [stripLooseQuotes(item)].slice(0, maxItems);
}

function assertStringArray(value: unknown, field: string, maxItems: number): string[] {
  if (value === null || value === undefined) {
    return [];
  }

  if (typeof value === 'string') {
    return parseSerializedStringArray(value, field, maxItems);
  }

  if (!Array.isArray(value)) {
    throw new Error(`${field} must be an array`);
  }

  const items = value.flatMap((item) => {
    if (typeof item !== 'string' || !item.trim()) {
      throw new Error(`${field} must contain non-empty strings`);
    }
    return parseSerializedStringArray(item, field, maxItems);
  });

  if (items.length > maxItems) {
    throw new Error(`${field} must contain at most ${maxItems} items`);
  }

  return items;
}

function readResponseMode(value: unknown, fallback: ResponseMode): ResponseMode {
  return typeof value === 'string' && responseModes.has(value as ResponseMode) ? (value as ResponseMode) : fallback;
}

function cleanLooseValue(value: string) {
  return value.replace(/\\"/g, '"').trim();
}

function readLooseField(source: string, field: string, nextField?: string) {
  const escapedField = field.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  if (nextField) {
    const escapedNextField = nextField.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    const betweenFields = new RegExp(`"${escapedField}"\\s*:\\s*"([\\s\\S]*?)"\\s*,\\s*"${escapedNextField}"`);
    const betweenMatch = source.match(betweenFields);
    if (betweenMatch?.[1]) return cleanLooseValue(betweenMatch[1]);
  }

  const regular = new RegExp(`"${escapedField}"\\s*:\\s*"([^"]*)"`);
  const match = source.match(regular);
  return match?.[1] ? cleanLooseValue(match[1]) : '';
}

function parseLooseChoiceObjects(value: string): ResponseChoice[] | null {
  const firstBracket = value.indexOf('[');
  const lastBracket = value.lastIndexOf(']');
  const arrayText = firstBracket >= 0 && lastBracket > firstBracket ? value.slice(firstBracket + 1, lastBracket) : value;
  const chunks = arrayText
    .split(/\}\s*,\s*\{/)
    .map((chunk, index, items) => {
      let normalized = chunk.trim();
      if (index > 0) normalized = `{${normalized}`;
      if (index < items.length - 1) normalized = `${normalized}}`;
      return normalized;
    })
    .filter((chunk) => chunk.includes('"label"') || chunk.includes('"description"'));

  const choices = chunks
    .map((chunk, index): ResponseChoice | null => {
      const id = readLooseField(chunk, 'id') || String.fromCharCode(65 + index);
      const label = readLooseField(chunk, 'label', 'description');
      const description = readLooseField(chunk, 'description', 'meaning') || label;
      const meaning = readLooseField(chunk, 'meaning', 'client_followup') || `choice_${index + 1}`;
      const clientFollowup = readLooseField(chunk, 'client_followup', 'requires_api_after_choice');
      const requiresMatch = chunk.match(/"requires_api_after_choice"\s*:\s*(true|false)/);

      if (!label || !description) return null;
      return {
        id,
        label,
        description,
        meaning,
        client_followup: clientFollowup || undefined,
        requires_api_after_choice: requiresMatch ? requiresMatch[1] === 'true' : true,
      };
    })
    .filter((choice): choice is ResponseChoice => Boolean(choice))
    .slice(0, 4);

  return choices.length >= 2 ? choices : null;
}

function parseSerializedChoices(value: unknown): ResponseChoice[] | null {
  if (typeof value !== 'string') return null;

  const trimmed = value
    .trim()
    .replace(/^```(?:json)?\s*/i, '')
    .replace(/\s*```$/i, '')
    .trim();

  if (!trimmed.startsWith('[') && !trimmed.startsWith('{') && !trimmed.startsWith('"')) {
    return null;
  }

  try {
    const parsed = JSON.parse(trimmed);
    if (parsed === value) return null;
    return readChoices(parsed);
  } catch {
    const looseChoices = parseLooseChoiceObjects(trimmed);
    if (looseChoices) return looseChoices;

    const firstBracket = trimmed.indexOf('[');
    const lastBracket = trimmed.lastIndexOf(']');
    if (firstBracket >= 0 && lastBracket > firstBracket) {
      try {
        return readChoices(JSON.parse(trimmed.slice(firstBracket, lastBracket + 1)));
      } catch {
        return null;
      }
    }
    return null;
  }
}

function normalizeChoice(value: unknown, index: number): ResponseChoice | null {
  if (typeof value === 'string') {
    const label = value.trim();
    if (!label) return null;
    return {
      id: String.fromCharCode(65 + index),
      label,
      description: label,
      meaning: `choice_${index + 1}`,
      requires_api_after_choice: true,
    };
  }

  if (!isRecord(value)) return null;

  const idValue = typeof value.id === 'string' ? value.id.trim() : '';
  const labelValue = typeof value.label === 'string' ? value.label.trim() : '';
  const descriptionValue = typeof value.description === 'string' ? value.description.trim() : '';
  const meaningValue = typeof value.meaning === 'string' ? value.meaning.trim() : '';
  const titleValue = typeof value.title === 'string' ? value.title.trim() : '';
  const textValue = typeof value.text === 'string' ? value.text.trim() : '';
  const clientFollowupValue =
    value.client_followup === null || value.client_followup === undefined
      ? undefined
      : typeof value.client_followup === 'string'
        ? value.client_followup.trim()
        : undefined;

  const label = labelValue || titleValue || textValue || descriptionValue;
  const description = descriptionValue || textValue || label;
  if (!label || !description) return null;

  return {
    id: idValue || String.fromCharCode(65 + index),
    label,
    description,
    meaning: meaningValue || `choice_${index + 1}`,
    client_followup: clientFollowupValue || undefined,
    requires_api_after_choice: readBoolean(value.requires_api_after_choice, true),
  };
}

function readChoices(value: unknown): ResponseChoice[] {
  if (value === null || value === undefined) {
    return [];
  }

  if (typeof value === 'string') {
    const parsed = parseSerializedChoices(value);
    if (parsed) return parsed;
  }

  if (Array.isArray(value) && value.every((item) => typeof item === 'string')) {
    const joined = value.join('\n');
    const parsed = parseSerializedChoices(joined);
    if (parsed) return parsed;
  }

  if (Array.isArray(value) && value.length === 1 && isRecord(value[0])) {
    for (const key of ['choices', 'label', 'description', 'text']) {
      const parsed = parseSerializedChoices(value[0][key]);
      if (parsed) return parsed;
    }
  }

  const rawChoices = Array.isArray(value)
    ? value
    : isRecord(value)
      ? Object.values(value)
      : typeof value === 'string'
        ? value
            .split(/\n+/)
            .map((item) => item.replace(/^[A-Da-d][.、)\s]+/, '').trim())
            .filter(Boolean)
        : [];

  return rawChoices
    .map((item, index) => normalizeChoice(item, index))
    .filter((item): item is ResponseChoice => Boolean(item))
    .slice(0, 4);
}

function validateChatResponse(value: unknown): ChatResponse {
  if (!isRecord(value)) {
    throw new Error('model output must be a JSON object');
  }

  const response_type = assertEnum(value.response_type, responseTypes, 'response_type');
  const risk_level = assertEnum(value.risk_level, riskLevels, 'risk_level');
  const phase = assertEnum(value.phase, phases, 'phase');
  const message = typeof value.message === 'string' && value.message.trim() ? value.message.trim() : null;
  if (!message) {
    throw new Error('message must be a non-empty string');
  }

  const mapping = assertNullableString(value.mapping, 'mapping');
  const question = assertNullableString(value.question, 'question');
  let choices = readChoices(value.choices);
  let response_mode = choices.length >= 2 ? 'choice' : readResponseMode(value.response_mode, 'free_text');
  let allow_free_text = readBoolean(value.allow_free_text, response_type !== 'crisis');
  const detected_beliefs = assertStringArray(value.detected_beliefs, 'detected_beliefs', 5);
  const detected_tensions = assertStringArray(value.detected_tensions, 'detected_tensions', 4);
  const detected_assumptions = assertStringArray(value.detected_assumptions, 'detected_assumptions', 5);
  const unclear_concepts = assertStringArray(value.unclear_concepts, 'unclear_concepts', 5);
  const can_summarize = readBoolean(value.can_summarize, false);
  const should_summarize = readBoolean(value.should_summarize, false);

  if (response_type === 'crisis') {
    if (risk_level !== 'high') throw new Error('crisis response must use high risk_level');
    if (phase !== 'crisis') throw new Error('crisis response must use crisis phase');
    if (question !== null) throw new Error('crisis response question must be null');
    if (can_summarize || should_summarize) throw new Error('crisis response cannot summarize');
    response_mode = 'free_text';
    choices = [];
    allow_free_text = false;
  }

  if (response_type === 'summary') {
    if (phase !== 'summary') throw new Error('summary response must use summary phase');
    if (question !== null) throw new Error('summary response question must be null');
    response_mode = 'free_text';
    choices = [];
  }

  if (response_mode === 'choice' && response_type === 'normal') {
    response_mode = choices.length >= 2 ? 'choice' : 'free_text';
  }

  if (response_mode === 'free_text' && response_type === 'normal') {
    choices = [];
  }

  return {
    response_type,
    risk_level,
    phase,
    message,
    mapping,
    question,
    response_mode,
    choices,
    allow_free_text,
    has_tension: detected_tensions.length > 0,
    detected_beliefs,
    detected_tensions,
    detected_assumptions,
    unclear_concepts,
    can_summarize,
    should_summarize,
  };
}

function hasSummaryReadyStructure(response: ChatResponse) {
  const hasBeliefTensionStructure = response.detected_beliefs.length >= 2 && response.detected_tensions.length >= 1;
  const hasAssumptionTensionStructure =
    response.detected_beliefs.length >= 1 &&
    response.detected_tensions.length >= 1 &&
    (response.detected_assumptions.length >= 1 || response.unclear_concepts.length >= 1);

  return (
    (hasBeliefTensionStructure || hasAssumptionTensionStructure) &&
    response.phase !== 'intake' &&
    response.response_type !== 'crisis'
  );
}

function fallbackList(primary: string[], secondary: string[], fallback: string) {
  const items = primary.length ? primary : secondary;
  return items.length ? items : [fallback];
}

function bulletLines(items: string[]) {
  return items.map((item) => `- ${item}`);
}

function createForcedSummaryResponse(request: RuntimeRequest, response?: ChatResponse): ChatResponse {
  const detected_beliefs = fallbackList(
    response?.detected_beliefs ?? [],
    request.detectedBeliefs,
    '用户正在尝试澄清一个尚未完全命名的核心信念',
  ).slice(0, 5);
  const detected_tensions = fallbackList(
    response?.detected_tensions ?? [],
    request.detectedTensions,
    '当前困惑中已经出现张力，但张力双方仍需要更精确命名',
  ).slice(0, 4);
  const detected_assumptions = fallbackList(
    response?.detected_assumptions ?? [],
    request.detectedAssumptions,
    '真正需要检验的前提仍需由用户确认',
  ).slice(0, 5);
  const unclear_concepts = fallbackList(
    response?.unclear_concepts ?? [],
    request.unclearConcepts,
    '核心概念的含义仍需进一步澄清',
  ).slice(0, 5);

  return {
    response_type: 'summary',
    risk_level: 'none',
    phase: 'summary',
    message: [
      '我先在这里做阶段性收束。',
      '',
      '1. 目前已经显出来的核心信念',
      ...bulletLines(detected_beliefs),
      '',
      '2. 它们之间的张力',
      ...bulletLines(detected_tensions),
      '',
      '3. 已显出来的前提',
      ...bulletLines(detected_assumptions),
      '',
      '4. 仍需澄清的概念',
      ...bulletLines(unclear_concepts),
      '',
      '5. 这次对话暂时抵达的位置',
      '- 这还不是结论，而是当前困惑结构的阶段性整理；继续分析时，应优先检验这些前提是否准确。',
    ].join('\n'),
    mapping: null,
    question: null,
    response_mode: 'free_text',
    choices: [],
    allow_free_text: true,
    has_tension: detected_tensions.length > 0,
    detected_beliefs,
    detected_tensions,
    detected_assumptions,
    unclear_concepts,
    can_summarize: true,
    should_summarize: false,
  };
}

function applySummaryPolicy(request: RuntimeRequest, response: ChatResponse): ChatResponse {
  if (response.response_type === 'crisis') {
    return {
      ...response,
      can_summarize: false,
      should_summarize: false,
    };
  }

  if (response.response_type === 'summary' || request.mode === 'summary') {
    return {
      ...response,
      can_summarize: true,
      should_summarize: false,
    };
  }

  const userTurnCount = countUserTurns(request);
  const canSummarize = userTurnCount >= ALLOW_SUMMARY_TURNS && hasSummaryReadyStructure(response);

  return {
    ...response,
    can_summarize: canSummarize,
    should_summarize: canSummarize && userTurnCount >= SUGGEST_SUMMARY_TURNS,
  };
}

function assertRuntimeSummaryPolicy(request: RuntimeRequest, response: ChatResponse) {
  if (request.mode === 'summary' && (response.response_type !== 'summary' || response.question !== null || !['已经明确的', '还不确定的', '暂时停在这里'].every(title => response.message.includes(title)))) {
    throw new Error('整理必须直接输出summary，message包含已经明确的、还不确定的、暂时停在这里三个标题，不能提问');
  }
  if (shouldForceSummary(request) && response.response_type !== 'summary') {
    throw new Error(`turn ${FORCE_SUMMARY_TURNS} and later must return a summary response`);
  }
}

function ensureSummarySections(response: ChatResponse): ChatResponse {
  if (response.response_type !== 'summary') return response;
  const message = response.message;
  if (message.includes('已经明确的') && message.includes('还不确定的') && message.includes('暂时停在这里')) return response;
  const bullets = (items: string[], fallback: string) => (items.length ? items : [fallback]).map(item => `- ${item}`).join('\n');
  return {
    ...response,
    question: null,
    response_mode: 'free_text',
    choices: [],
    message: [
      '阶段性小结',
      '',
      '已经明确的',
      bullets(response.detected_beliefs, '目前还没有得到用户确认的确定信念。'),
      '',
      '还不确定的',
      bullets([...response.detected_tensions, ...response.detected_assumptions, ...response.unclear_concepts], '上述联系和概念仍需用户确认。'),
      '',
      '暂时停在这里',
      '这份整理不构成结论，也不要求现在继续回答。',
    ].join('\n'),
  };
}

function applyRuntimeSummaryPolicy(request: RuntimeRequest, response: ChatResponse) {
  if (shouldForceSummary(request) && response.response_type !== 'summary') {
    return ensureSummarySections(createForcedSummaryResponse(request, response));
  }

  assertRuntimeSummaryPolicy(request, response);
  return ensureSummarySections(applySummaryPolicy(request, response));
}

async function generateWithClaude(request: RuntimeRequest) {
  const system = await getSystemPrompt();
  const messages = buildMessages(request);
  const firstResponse = await callAnthropic(system, messages);
  const firstPayload = extractAssistantPayload(firstResponse);

  try {
    const response = validateChatResponse(firstPayload);
    return applyRuntimeSummaryPolicy(request, response);
  } catch (firstError) {
    const repairMessages = mergeConsecutiveMessages([
      ...messages,
      {
        role: 'user',
        content: [
          '上一条输出无法通过镜观 JSON schema 校验。',
          `校验错误：${firstError instanceof Error ? firstError.message : 'unknown validation error'}`,
          '上一条输出如下：',
          stringifyAssistantPayload(firstResponse).slice(0, REPAIR_TEXT_LIMIT),
          '请重新调用 format_jingguan_response 工具，返回修正后的结构化结果。',
        ].join('\n'),
      },
    ]);
    const repairedResponse = await callAnthropic(system, repairMessages);
    const response = validateChatResponse(extractAssistantPayload(repairedResponse));
    return applyRuntimeSummaryPolicy(request, response);
  }
}

export async function handleChatPayload(payload: unknown): Promise<EndpointResult> {
  try {
    const request = getRuntimeRequest(payload, 'chat');
    const text = request.text ?? '';
    if (/(整理|总结|小结)/.test(text) && !/(不要|不想|不用)(?:再|先)?(?:整理|总结|小结)/.test(text)) request.mode = 'summary';
    return {
      status: 200,
      body: await generateWithClaude(request),
    };
  } catch (error) {
    return endpointError(error);
  }
}

export async function handleSummaryPayload(payload: unknown): Promise<EndpointResult> {
  try {
    const request = getRuntimeRequest(payload, 'summary');
    return {
      status: 200,
      body: await generateWithClaude(request),
    };
  } catch (error) {
    return endpointError(error);
  }
}
