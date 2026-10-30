// Apple Foundation Models as a pi-ai provider: Swift runs each request in a `LanguageModelSession` over the app's
// `LanguageModel` (the on-device system model, Private Cloud Compute, or any other), and this module turns what Swift
// reports into pi-ai's assistant message events. pi-durable runs the tools; Swift only reports the calls.
import {
	type Api,
	type AssistantMessage,
	type Context,
	createProvider,
	type Model,
	type ProviderStreams,
	type SimpleStreamOptions,
	type TextContent,
	type ThinkingContent,
	type ToolCall,
	type Usage,
} from "@earendil-works/pi-ai";
import { createAssistantMessageEventStream } from "@earendil-works/pi-ai/utils/event-stream";
import { getSystemMessageText, renderSystemMessageUpdate } from "@earendil-works/pi-ai/utils/text";
import { getCurrentTools } from "@earendil-works/pi-ai/utils/transcript";
import type { AssistantMessageEventStream } from "@earendil-works/pi-ai/utils/event-stream";
import { type Args, hostStream, register } from "./core.ts";
import { modelsState } from "./models.ts";

const ZERO_USAGE: Usage = {
	input: 0,
	output: 0,
	cacheRead: 0,
	cacheWrite: 0,
	totalTokens: 0,
	cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
};

type JsonSchema = Record<string, unknown>;

/**
 * A JSON schema whose object properties keep their order as `x-order`, which Swift's dictionaries would lose. Swift
 * normalizes the rest for Foundation Models' `GenerationSchema`.
 */
export function foundationSchema(schema: unknown): unknown {
	if (Array.isArray(schema)) return schema.map((item) => foundationSchema(item));
	if (schema === null || typeof schema !== "object") return schema;
	const result: JsonSchema = {};
	for (const [key, value] of Object.entries(schema as JsonSchema)) {
		if (key === "$schema") continue;
		result[key] =
			key === "properties" && value !== null && typeof value === "object"
				? Object.fromEntries(Object.entries(value as JsonSchema).map(([name, property]) => [name, foundationSchema(property)]))
				: foundationSchema(value);
	}
	if (result.properties !== undefined && result["x-order"] === undefined) result["x-order"] = Object.keys(result.properties as JsonSchema);
	return result;
}

/** What Swift answers for one request. */
type Answer = {
	/** The response and reasoning as finally written; streamed deltas only append. */
	text: string;
	thinking: string;
	/** Arguments as the model wrote them, so their key order survives. */
	toolCalls: { id: string; name: string; argumentsJSON: string }[];
	stopReason: AssistantMessage["stopReason"];
	usage: { input: number; cachedInput: number; output: number; reasoning: number };
	errorMessage?: string;
	/** A transient failure, and when the model asked to be tried again, if it said (`FoundationModelsResponder.retry`). */
	retryable?: boolean;
	retryAfterMs?: number;
};

function toolDefinition(tool: { name: string; description: string; parameters: unknown }) {
	return { name: tool.name, description: tool.description, parameters: foundationSchema(tool.parameters) };
}

function respond(models: number, provider: string, model: Model<Api>, context: Context, options: SimpleStreamOptions | undefined) {
	const stream = createAssistantMessageEventStream();
	void run(stream, models, provider, model, context, options);
	return stream;
}

async function run(
	stream: AssistantMessageEventStream,
	models: number,
	provider: string,
	model: Model<Api>,
	context: Context,
	options: SimpleStreamOptions | undefined,
): Promise<void> {
	const signal = options?.signal;
	const base = { role: "assistant" as const, api: model.api, provider: model.provider, model: model.id, timestamp: Date.now() };
	const content: (TextContent | ThinkingContent | ToolCall)[] = [];
	const partial = (): AssistantMessage => ({ ...base, content: content.map((block) => ({ ...block })), usage: ZERO_USAGE, stopReason: "stop" });
	let text = -1;
	let thinking = -1;
	const close = () => {
		if (thinking >= 0) {
			stream.push({ type: "thinking_end", contentIndex: thinking, content: (content[thinking] as ThinkingContent).thinking, partial: partial() });
			thinking = -1;
		}
		if (text >= 0) {
			stream.push({ type: "text_end", contentIndex: text, content: (content[text] as TextContent).text, partial: partial() });
			text = -1;
		}
	};
	/** Makes a block's text the final one, creating the block when nothing streamed. */
	const reconcile = (type: "text" | "thinking", final: string) => {
		let index = type === "text" ? text : thinking;
		if (index < 0) {
			if (final === "") return;
			content.push(type === "text" ? { type, text: "" } : { type, thinking: "" });
			index = content.length - 1;
			if (type === "text") text = index;
			else thinking = index;
			stream.push({ type: type === "text" ? "text_start" : "thinking_start", contentIndex: index, partial: partial() });
		}
		const block = content[index] as unknown as { text: string; thinking: string };
		const current = type === "text" ? block.text : block.thinking;
		if (current === final) return;
		if (final.startsWith(current)) {
			const delta = final.slice(current.length);
			if (type === "text") block.text = final;
			else block.thinking = final;
			stream.push({ type: type === "text" ? "text_delta" : "thinking_delta", contentIndex: index, delta, partial: partial() });
		} else if (type === "text") block.text = final;
		else block.thinking = final;
	};
	stream.push({ type: "start", partial: partial() });
	try {
		const messages = context.messages;
		const request = () => hostStream<Answer>(
			"foundationModels.respond",
			{
				models,
				provider,
				model: model.id,
				// Every tool offered now; the session needs them all to resolve calls.
				tools: getCurrentTools(messages as never).map(toolDefinition),
				// System messages stay where they are, as pi-ai sends them to APIs that accept mid-conversation system
				// messages: later ones become instructions entries in place, so a section change extends the session
				// instead of rewriting its start, which keeps its cache. The first is the whole prompt (pi-durable
				// restates it after a compaction or reset, possibly after the run's first messages): Swift makes it the
				// session's leading instructions.
				messages: messages.map((message, index) => {
					if (message.role !== "system") return message;
					const removed = (message.toolsRemoved ?? []).map((tool) => tool.name);
					const first = messages.findIndex((other) => other.role === "system") === index;
					const text = [
						first ? getSystemMessageText(message) : renderSystemMessageUpdate(message),
						...(removed.length === 0 ? [] : [`These tools are no longer available: ${removed.join(", ")}.`]),
					]
						.filter((part) => part.length > 0)
						.join("\n\n");
					return { role: "system", text, tools: (message.toolsAdded ?? []).map(toolDefinition), ...(first ? { first } : {}) };
				}),
				// The conversation's provider session: requests that extend what its Foundation Models session holds
				// reuse that session and its cache.
				sessionId: options?.sessionId ?? null,
				cacheRetention: options?.cacheRetention ?? null,
				options: {
					temperature: options?.temperature ?? null,
					maxTokens: options?.maxTokens ?? null,
					reasoning: options?.reasoning ?? null,
					toolChoice: options?.toolChoice ?? null,
				},
			},
			(event: Args) => {
				if (event.type === "text") {
					if (thinking >= 0) close();
					if (text < 0) {
						content.push({ type: "text", text: "" });
						text = content.length - 1;
						stream.push({ type: "text_start", contentIndex: text, partial: partial() });
					}
					(content[text] as TextContent).text += event.delta;
					stream.push({ type: "text_delta", contentIndex: text, delta: event.delta, partial: partial() });
				} else if (event.type === "thinking") {
					if (thinking < 0) {
						content.push({ type: "thinking", thinking: "" });
						thinking = content.length - 1;
						stream.push({ type: "thinking_start", contentIndex: thinking, partial: partial() });
					}
					(content[thinking] as ThinkingContent).thinking += event.delta;
					stream.push({ type: "thinking_delta", contentIndex: thinking, delta: event.delta, partial: partial() });
				}
			},
			signal === undefined ? undefined : { abortSignal: signal },
		);
		// A transient failure before anything streamed is tried again within the request, as pi-ai's providers retry an
		// HTTP failure (`retryProviderRequest`): up to `maxRetries` times (none by default), waiting for the model's
		// reset date as a `retry-after`, unless that's longer than `maxRetryDelayMs` (60 seconds by default), or else
		// pi-ai's backoff.
		const maxRetries = options?.maxRetries ?? 0;
		const maxRetryDelayMs = options?.maxRetryDelayMs ?? 60_000;
		let answer = await request();
		for (let attempt = 0; answer.stopReason === "error" && answer.retryable === true && content.length === 0 && attempt < maxRetries; attempt++) {
			if (signal?.aborted) break;
			const requested = answer.retryAfterMs;
			if (requested !== undefined && maxRetryDelayMs > 0 && requested > maxRetryDelayMs) {
				answer = {
					...answer,
					errorMessage: `Server requested ${Math.ceil(requested / 1000)}s retry delay (max: ${Math.ceil(maxRetryDelayMs / 1000)}s). ${answer.errorMessage ?? ""}`,
				};
				break;
			}
			const delay = requested ?? Math.min(0.5 * 2 ** attempt, 8) * 1000 * (1 - Math.random() * 0.25);
			await new Promise<void>((resolve) => {
				const timer = setTimeout(resolve, delay);
				signal?.addEventListener("abort", () => {
					clearTimeout(timer);
					resolve();
				}, { once: true });
			});
			if (signal?.aborted) break;
			answer = await request();
		}
		if (signal?.aborted) throw new Error("Request was aborted");
		// The final text replaces what streamed, in case the model revised it mid-stream.
		reconcile("thinking", answer.thinking);
		reconcile("text", answer.text);
		close();
		for (const call of answer.toolCalls) {
			const block: ToolCall = { type: "toolCall", id: call.id, name: call.name, arguments: JSON.parse(call.argumentsJSON) };
			content.push({ ...block, arguments: {} });
			const index = content.length - 1;
			stream.push({ type: "toolcall_start", contentIndex: index, partial: partial() });
			stream.push({ type: "toolcall_delta", contentIndex: index, delta: call.argumentsJSON, partial: partial() });
			content[index] = block;
			stream.push({ type: "toolcall_end", contentIndex: index, toolCall: block, partial: partial() });
		}
		const input = Math.max(0, answer.usage.input - answer.usage.cachedInput);
		const usage: Usage = {
			...ZERO_USAGE,
			input,
			cacheRead: answer.usage.cachedInput,
			output: answer.usage.output,
			...(answer.usage.reasoning > 0 ? { reasoning: answer.usage.reasoning } : {}),
			totalTokens: answer.usage.input + answer.usage.output,
		};
		const message: AssistantMessage = {
			...partial(),
			usage,
			stopReason: answer.stopReason,
			...(answer.errorMessage === undefined ? {} : { errorMessage: answer.errorMessage }),
		};
		if (message.stopReason === "error") {
			stream.push({ type: "error", reason: "error", error: message });
		} else {
			stream.push({ type: "done", reason: message.stopReason as "stop" | "length" | "toolUse", message });
		}
		stream.end(message);
	} catch (error) {
		close();
		const aborted = signal?.aborted === true;
		const message: AssistantMessage = {
			...partial(),
			stopReason: aborted ? "aborted" : "error",
			errorMessage: aborted ? "Request was aborted" : error instanceof Error ? error.message : String(error),
		};
		stream.push({ type: "error", reason: aborted ? "aborted" : "error", error: message });
		stream.end(message);
	}
}

register({
	/** A provider whose models are `LanguageModel`s Swift registered for this models collection. */
	"models.addFoundationModels": (args) => {
		const state = modelsState(args.id);
		const api = `foundation-models:${args.provider}`;
		const streams: ProviderStreams = {
			stream: (model, context, options) => respond(args.id, args.provider, model, context as Context, options as SimpleStreamOptions),
			streamSimple: (model, context, options) => respond(args.id, args.provider, model, context as Context, options),
		};
		state.models.setProvider(
			createProvider({
				id: args.provider,
				name: args.name ?? args.provider,
				baseUrl: "apple://foundation-models",
				// On-device and Private Cloud Compute models need no credentials.
				auth: { apiKey: { name: args.name ?? args.provider, resolve: async () => ({ auth: { apiKey: "unused" } }) } },
				models: (args.models as Args[]).map((model) => ({
					id: model.id,
					name: model.name ?? model.id,
					api,
					provider: args.provider,
					baseUrl: "apple://foundation-models",
					reasoning: model.reasoning === true,
					...(model.thinkingLevelMap ? { thinkingLevelMap: model.thinkingLevelMap } : {}),
					input: model.input ?? ["text"],
					cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
					contextWindow: model.contextWindow,
					maxTokens: model.maxTokens,
				})),
				api: { [api]: streams },
			}) as never,
		);
	},
});
