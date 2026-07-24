import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";
import { createPowerPrompterQueueManager } from "./Umbra.powerprompter.queue-manager.js";

const NODE_CLASS = "UmbraPowerPrompterReader";
const UNIFIED_NODE_CLASS = "UmbraPowerPrompter";
const LORA_NODE_CLASS = "UmbraA1111LoraSyntax";
const CHECKPOINT_NODE_CLASSES = new Set(["UmbraLoadCheckpoint", "CheckpointLoaderSimple"]);
const KSAMPLER_NODE_CLASSES = new Set(["UmbraKSampler", "UmbraKSamplerNormal", "KSampler", "KSamplerAdvanced"]);
const SEED_VALUE_NODE_CLASSES = new Set(["UmbraSeedValue"]);
const LORA_NONE_OPTION = "[None]";
const LORA_METADATA_ENDPOINTS = ["/easyuse/metadata/", "/pysssss/metadata/", "/umbra/metadata/"];
const WS_ROLE = "comfy_bridge";
const RECONNECT_MS = 2500;
const SAVE_NODE_TYPES = new Set(["UmbraLabSaveImage"]);
const MAX_SAFE_SEED = Number.MAX_SAFE_INTEGER;
const MAX_QUEUE_SETS = 10;
const PREVIEW_FRAME_THROTTLE_MS = 180;
const PREVIEW_MAX_DATA_URL_LENGTH = 8_000_000;
const POWER_PROMPTER_PREVIEW_METHOD = "taesd";
const PREVIEW_EVENT_NAMES = [
  "b_preview",
  "preview",
  "progress_preview",
  "preview_image",
  "unencoded_preview_image",
  "latent_preview",
  "binary_preview",
  "jpeg_preview",
  "image_preview",
];
const PREVIEW_BINARY_HEADER_BYTES = 24;
const previewBinaryHeaderDecoder = typeof TextDecoder !== "undefined" ? new TextDecoder() : null;
const IDLE_FALLBACK_DELAY_MS = 1500;
const QUEUE_SUBMIT_BATCH_SIZE = 1;
const QUEUE_SUBMIT_BETWEEN_PROMPTS_MS = 55;
const QUEUE_SUBMIT_BETWEEN_BATCHES_MS = 320;
const PROMPT_COMPLETION_TIMEOUT_MS = 180000;
const BRIDGE_STATE_HEARTBEAT_MS = 2200;
const LORA_DESCRIPTION_ALLOWED_TAGS = new Set([
  "p", "br",
  "strong", "b", "em", "i", "u", "s",
  "ul", "ol", "li",
  "h1", "h2", "h3", "h4", "h5", "h6",
  "blockquote", "code", "pre",
  "a",
]);
const LORA_DESCRIPTION_ALLOWED_PROTOCOLS = new Set(["http:", "https:"]);
const liveLoraCatalogHintsCache = {
  loaded: false,
  items: [],
};
const liveModelCatalogHintsCache = {
  loaded: false,
  items: [],
};
const loraMetadataCache = new Map();
const modelMetadataCache = new Map();
const civitaiHashCache = new Map();
const civitaiModelCache = new Map();

let prompterWs = null;
let reconnectTimer = null;
let prompterWsStatus = "disconnected";
let isQueueing = false;
let queuePumpActive = false;
let executionListenersAttached = false;
let previewFrameInFlight = false;
let lastPreviewFrameAt = 0;
let currentExecutingPromptId = "";
let currentQueueExecution = null;
let queueCancelAllRequested = false;
let queuePaused = false;
const queueCancelRequestIds = new Set();
let bridgeStateTimer = null;
let lastBridgeStateSignature = "";
const BRIDGE_ID = (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function")
  ? crypto.randomUUID()
  : `umbra-bridge-${Date.now()}-${Math.random().toString(36).slice(2, 9)}`;

const activeQueueTrackers = new Map();
const trackerProgressByKey = new Map();
const trackerProgressSentByKey = new Map();
const idleFallbackTimers = new Map();
const pendingQueueRequests = [];

const DEFAULT_DETAILER_PIPELINE = [
  {
    id: "detail-person", enabled: true, label: "Person", detectorModel: "segm/person_yolov8m-seg.pt",
    guideSize: 1024, guideSizeFor: "bbox", maxSize: 1536, seedOffset: 1, steps: 8, cfg: 4,
    samplerName: "er_sde", scheduler: "simple", denoise: 0.18, feather: 10, noiseMask: true,
    forceInpaint: true, bboxThreshold: 0.5, bboxDilation: 10, bboxCropFactor: 2.2, useSam: true,
    samModel: "sam_vit_b_01ec64.pth", samDeviceMode: "AUTO", samDetectionHint: "center-1",
    samDilation: 0, samThreshold: 0.93, samBboxExpansion: 0, samMaskHintThreshold: 0.7,
    samMaskHintUseNegative: "False", dropSize: 10,
    wildcard: "[CONCAT] coherent anatomy, natural body proportions, coherent clothing folds",
    cycle: 1, noiseMaskFeather: 24, tiledEncode: false, tiledDecode: false,
  },
  {
    id: "detail-face", enabled: true, label: "Face", detectorModel: "bbox/face_yolov8m.pt",
    guideSize: 512, guideSizeFor: "crop_region", maxSize: 1024, seedOffset: 2, steps: 8, cfg: 4,
    samplerName: "er_sde", scheduler: "simple", denoise: 0.18, feather: 5, noiseMask: true,
    forceInpaint: true, bboxThreshold: 0.5, bboxDilation: 10, bboxCropFactor: 2.5, useSam: true,
    samModel: "sam_vit_b_01ec64.pth", samDeviceMode: "AUTO", samDetectionHint: "center-1",
    samDilation: 0, samThreshold: 0.93, samBboxExpansion: 0, samMaskHintThreshold: 0.7,
    samMaskHintUseNegative: "False", dropSize: 10, wildcard: "", cycle: 1,
    noiseMaskFeather: 20, tiledEncode: false, tiledDecode: false,
  },
  {
    id: "detail-eyes", enabled: true, label: "Eyes", detectorModel: "bbox/Eyes.pt",
    guideSize: 384, guideSizeFor: "bbox", maxSize: 512, seedOffset: 3, steps: 7, cfg: 4,
    samplerName: "er_sde", scheduler: "simple", denoise: 0.16, feather: 4, noiseMask: true,
    forceInpaint: true, bboxThreshold: 0.4, bboxDilation: 5, bboxCropFactor: 2.4, useSam: true,
    samModel: "sam_vit_b_01ec64.pth", samDeviceMode: "AUTO", samDetectionHint: "center-1",
    samDilation: 0, samThreshold: 0.93, samBboxExpansion: 0, samMaskHintThreshold: 0.7,
    samMaskHintUseNegative: "False", dropSize: 4,
    wildcard: "[CONCAT] detailed symmetrical eyes, sharp irises, natural pupils", cycle: 1,
    noiseMaskFeather: 12, tiledEncode: false, tiledDecode: false,
  },
  {
    id: "detail-hands", enabled: true, label: "Hands", detectorModel: "bbox/hand_yolov8s.pt",
    guideSize: 512, guideSizeFor: "bbox", maxSize: 768, seedOffset: 4, steps: 10, cfg: 4,
    samplerName: "er_sde", scheduler: "simple", denoise: 0.28, feather: 10, noiseMask: true,
    forceInpaint: true, bboxThreshold: 0.35, bboxDilation: 14, bboxCropFactor: 2.8, useSam: true,
    samModel: "sam_vit_b_01ec64.pth", samDeviceMode: "AUTO", samDetectionHint: "center-1",
    samDilation: 0, samThreshold: 0.93, samBboxExpansion: 0, samMaskHintThreshold: 0.7,
    samMaskHintUseNegative: "False", dropSize: 10,
    wildcard: "[CONCAT] detailed hands, anatomically correct hands, five fingers, natural finger spacing",
    cycle: 1, noiseMaskFeather: 20, tiledEncode: false, tiledDecode: false,
  },
];

const DEFAULT_GENERATION_STATE = {
  detailerPipeline: DEFAULT_DETAILER_PIPELINE.map((stage) => ({ ...stage })),
  negativePrompt: "",
  seed: 0,
  controlAfterGenerate: "fixed",
  incrementStep: 1,
  steps: 20,
  cfg: 7,
  samplerName: "euler",
  scheduler: "normal",
  modelType: "checkpoint",
  checkpointName: "",
  aspectRatio: "SDXL - 1:1 square 1024x1024",
  swapDimensions: false,
  width: 1024,
  height: 1024,
  batchSize: 1,
  loras: [],
  thumbnailOverrides: {},
};

const latestSyncState = {
  prompts: [],
  activePrompt: "",
  joinedPrompt: "",
  file: "",
  activeQueueSet: 1,
  styleSeedMode: "same",
  promptSetIds: [],
  generation: { ...DEFAULT_GENERATION_STATE },
};

let queueManager = null;

function createQueueTracker(requestId, prompts) {
  const cleanPrompts = normalizePrompts(prompts, { dedupe: false });
  const tracker = {
    requestId,
    prompts: cleanPrompts,
    total: cleanPrompts.length,
    completedCount: 0,
    completedBySaveCount: 0,
    completedFlags: new Array(cleanPrompts.length).fill(false),
    promptIds: new Array(cleanPrompts.length).fill(""),
    promptSeeds: new Array(cleanPrompts.length).fill(0),
    promptIdToIndex: new Map(),
    createdAt: Date.now(),
  };
  activeQueueTrackers.set(requestId, tracker);
  return tracker;
}

function clearQueueTrackers() {
  queueManager?.clearQueueTrackers();
}

function clearQueueCancelState() {
  queueCancelAllRequested = false;
  queueCancelRequestIds.clear();
}

function emitQueuePauseState(source = "bridge") {
  queueManager?.emitQueuePauseState(source);
}

function markQueueCancelRequested(rawRequestIds) {
  const ids = Array.isArray(rawRequestIds)
    ? rawRequestIds.map((entry) => String(entry || "").trim()).filter((entry) => entry.length > 0)
    : [];
  if (ids.length <= 0) {
    queueCancelAllRequested = true;
    return;
  }
  for (const requestId of ids) {
    queueCancelRequestIds.add(requestId);
  }
}

function isQueueCancelRequestedFor(requestId) {
  if (queueCancelAllRequested) return true;
  const key = String(requestId || "").trim();
  if (!key) return false;
  return queueCancelRequestIds.has(key);
}

async function waitWhileQueuePaused(requestId) {
  while (queuePaused === true) {
    if (isQueueCancelRequestedFor(requestId)) {
      return false;
    }
    await sleep(150);
  }
  return true;
}

function clearIdleFallbackTimer(requestId) {
  const key = String(requestId || "");
  const timer = idleFallbackTimers.get(key);
  if (!timer) return;
  clearTimeout(timer);
  idleFallbackTimers.delete(key);
}

function clearAllIdleFallbackTimers() {
  for (const requestId of Array.from(idleFallbackTimers.keys())) {
    clearIdleFallbackTimer(requestId);
  }
}

function scheduleIdleFallback(tracker) {
  if (!tracker) return;
  if (tracker.completedCount >= tracker.total) return;
  if (tracker.completedBySaveCount > 0) return;

  const requestId = String(tracker.requestId || "");
  const expectedCompleted = tracker.completedCount;
  const expectedSaveCompleted = tracker.completedBySaveCount;
  clearIdleFallbackTimer(requestId);

  const timer = setTimeout(() => {
    idleFallbackTimers.delete(requestId);
    const latest = activeQueueTrackers.get(requestId);
    if (!latest) return;
    if (latest.completedCount >= latest.total) return;
    if (latest.completedBySaveCount > 0) return;
    if (
      latest.completedCount !== expectedCompleted ||
      latest.completedBySaveCount !== expectedSaveCompleted
    ) {
      return;
    }
    applyIdleFallbackToTracker(latest);
  }, IDLE_FALLBACK_DELAY_MS);

  idleFallbackTimers.set(requestId, timer);
}

function parsePromptIdFromQueueResult(value) {
  if (typeof value === "string") return value.trim();
  if (!value || typeof value !== "object") return "";
  const raw =
    value.prompt_id ??
    value.promptId ??
    value.id ??
    value?.data?.prompt_id ??
    value?.response?.prompt_id ??
    "";
  return String(raw || "").trim();
}

function setTrackerPromptId(tracker, promptIndex, promptIdRaw) {
  if (!tracker) return;
  const promptId = String(promptIdRaw || "").trim();
  if (!promptId) return;
  if (promptIndex < 0 || promptIndex >= tracker.total) return;
  tracker.promptIds[promptIndex] = promptId;
  tracker.promptIdToIndex.set(promptId, promptIndex);
}

function setTrackerPromptSeed(tracker, promptIndex, seedRaw) {
  if (!tracker) return;
  const seedNum = Number(seedRaw);
  if (!Number.isFinite(seedNum)) return;
  if (promptIndex < 0 || promptIndex >= tracker.total) return;
  tracker.promptSeeds[promptIndex] = Math.max(0, Math.floor(seedNum));
}

function trackerProgressKey(requestId, promptIndex) {
  return `${String(requestId || "")}:${Math.max(0, Math.floor(Number(promptIndex) || 0))}`;
}

function setTrackerProgress(tracker, promptIndex, stepRaw, maxStepRaw) {
  if (!tracker) return;
  const step = Number(stepRaw);
  const maxStep = Number(maxStepRaw);
  const normalizedStep = Number.isFinite(step) ? Math.max(0, Math.floor(step)) : 0;
  const normalizedMax = Number.isFinite(maxStep) ? Math.max(0, Math.floor(maxStep)) : 0;
  trackerProgressByKey.set(
    trackerProgressKey(tracker.requestId, promptIndex),
    { step: normalizedStep, maxStep: normalizedMax }
  );
}

function getTrackerProgress(tracker, promptIndex) {
  if (!tracker) return { step: 0, maxStep: 0 };
  const progress = trackerProgressByKey.get(trackerProgressKey(tracker.requestId, promptIndex));
  if (!progress) return { step: 0, maxStep: 0 };
  return {
    step: Number.isFinite(progress.step) ? Math.max(0, Math.floor(progress.step)) : 0,
    maxStep: Number.isFinite(progress.maxStep) ? Math.max(0, Math.floor(progress.maxStep)) : 0,
  };
}

function clearTrackerProgress(requestId) {
  const prefix = `${String(requestId || "")}:`;
  for (const key of Array.from(trackerProgressByKey.keys())) {
    if (key.startsWith(prefix)) trackerProgressByKey.delete(key);
  }
  for (const key of Array.from(trackerProgressSentByKey.keys())) {
    if (key.startsWith(prefix)) trackerProgressSentByKey.delete(key);
  }
}

function resolvePromptIndexFromTracker(tracker, promptId) {
  if (!tracker || tracker.total <= 0) return -1;
  if (promptId) {
    const mapped = tracker.promptIdToIndex.get(promptId);
    if (Number.isFinite(mapped) && mapped >= 0 && mapped < tracker.total && !tracker.completedFlags[mapped]) {
      return mapped;
    }
  }
  for (let idx = 0; idx < tracker.total; idx += 1) {
    if (!tracker.completedFlags[idx]) return idx;
  }
  return -1;
}

function resolveTrackerForPromptId(promptId) {
  if (activeQueueTrackers.size === 0) return null;
  if (promptId) {
    for (const tracker of activeQueueTrackers.values()) {
      const promptIndex = resolvePromptIndexFromTracker(tracker, promptId);
      if (promptIndex >= 0 && tracker.promptIdToIndex.get(promptId) === promptIndex) {
        return { tracker, promptIndex };
      }
    }
  }
  if (currentQueueExecution) {
    const activeRequestId = String(currentQueueExecution.requestId || "").trim();
    const activePromptIndex = Number.isFinite(currentQueueExecution.promptIndex)
      ? Math.max(0, Math.floor(currentQueueExecution.promptIndex))
      : -1;
    const tracker = activeQueueTrackers.get(activeRequestId);
    if (
      tracker
      && activePromptIndex >= 0
      && activePromptIndex < tracker.total
      && !tracker.completedFlags[activePromptIndex]
    ) {
      return { tracker, promptIndex: activePromptIndex };
    }
  }
  for (const tracker of activeQueueTrackers.values()) {
    const promptIndex = resolvePromptIndexFromTracker(tracker, "");
    if (promptIndex >= 0) return { tracker, promptIndex };
  }
  return null;
}

function extractPromptIdFromExecution(detail) {
  if (!detail || typeof detail !== "object") return "";
  const raw =
    detail.prompt_id ??
    detail.promptId ??
    detail?.extra_data?.prompt_id ??
    detail?.data?.prompt_id ??
    "";
  return String(raw || "").trim();
}

function extractNodeTypeFromExecution(detail) {
  if (!detail || typeof detail !== "object") return "";
  const rawNode = detail.node ?? detail.node_id ?? detail.nodeId ?? detail.id;
  if (rawNode && typeof rawNode === "object") {
    const typed = String(rawNode.type || rawNode.class_type || "").trim();
    if (typed) return typed;
  }
  const nodeId = Number(rawNode);
  if (!Number.isFinite(nodeId)) return "";
  try {
    const node = app?.graph?.getNodeById?.(nodeId);
    return String(node?.type || "").trim();
  } catch {
    return "";
  }
}

function payloadContainsImageOutput(payload, depth = 0) {
  if (depth > 4 || payload == null) return false;
  if (Array.isArray(payload)) {
    return payload.some((entry) => payloadContainsImageOutput(entry, depth + 1));
  }
  if (typeof payload !== "object") return false;
  const images = payload.images;
  if (Array.isArray(images) && images.length > 0) return true;
  for (const value of Object.values(payload)) {
    if (payloadContainsImageOutput(value, depth + 1)) return true;
  }
  return false;
}

function extractProgressValue(detail) {
  if (!detail || typeof detail !== "object") return { step: 0, maxStep: 0, hasProgress: false };
  const rawStep =
    detail.value ??
    detail.step ??
    detail.progress ??
    detail.current ??
    detail?.data?.value ??
    detail?.data?.step ??
    detail?.data?.progress ??
    detail?.detail?.value ??
    detail?.detail?.step ??
    detail?.detail?.progress;
  const rawMax =
    detail.max ??
    detail.total ??
    detail.progress_max ??
    detail.progressMax ??
    detail?.data?.max ??
    detail?.data?.total ??
    detail?.detail?.max ??
    detail?.detail?.total;
  const stepNum = Number(rawStep);
  const maxNum = Number(rawMax);
  const hasStep = Number.isFinite(stepNum);
  const hasMax = Number.isFinite(maxNum);
  return {
    step: hasStep ? Math.max(0, Math.floor(stepNum)) : 0,
    maxStep: hasMax ? Math.max(0, Math.floor(maxNum)) : 0,
    hasProgress: hasStep || hasMax,
  };
}

function normalizePreviewArrayBuffer(bufferLike) {
  const buffer = bufferLike instanceof ArrayBuffer
    ? bufferLike
    : bufferLike.buffer.slice(bufferLike.byteOffset, bufferLike.byteOffset + bufferLike.byteLength);
  if (!(buffer instanceof ArrayBuffer) || buffer.byteLength <= 0) return null;
  if (buffer.byteLength < 8) {
    return { type: "blob", value: new Blob([buffer], { type: "image/jpeg" }) };
  }

  const view = new DataView(buffer);
  const eventType = view.getUint32(0, false);
  if (eventType === 1 && buffer.byteLength > 8) {
    const imageType = view.getUint32(4, false);
    const mimeType = imageType === 2 ? "image/png" : "image/jpeg";
    return {
      type: "blob",
      value: new Blob([buffer.slice(8)], { type: mimeType }),
    };
  }

  if (eventType === 4 && buffer.byteLength > 8) {
    const metadataLength = view.getUint32(4, false);
    const imageStart = 8 + Math.max(0, metadataLength);
    if (imageStart >= buffer.byteLength) return null;
    let mimeType = "image/jpeg";
    try {
      const metadataText = previewBinaryHeaderDecoder
        ? previewBinaryHeaderDecoder.decode(buffer.slice(8, imageStart))
        : "";
      const metadata = metadataText ? JSON.parse(metadataText) : null;
      const imageType = String(metadata?.image_type || "").trim();
      if (imageType) mimeType = imageType;
    } catch {
      // Metadata is optional; keep the image frame usable.
    }
    return {
      type: "blob",
      value: new Blob([buffer.slice(imageStart)], { type: mimeType }),
    };
  }

  return { type: "blob", value: new Blob([buffer], { type: "image/jpeg" }) };
}

function normalizePreviewPayload(detail, depth = 0) {
  if (!detail || depth > 3) return null;
  if (typeof detail === "string") {
    const trimmed = detail.trim();
    if (trimmed.startsWith("data:image/")) {
      return { type: "data_url", value: trimmed };
    }
    return null;
  }
  if (detail instanceof Blob) return { type: "blob", value: detail };
  if (detail instanceof ArrayBuffer) return normalizePreviewArrayBuffer(detail);
  if (ArrayBuffer.isView(detail)) {
    return normalizePreviewArrayBuffer(detail);
  }
  if (typeof detail !== "object") return null;

  const directDataUrl =
    detail.data_url ??
    detail.dataUrl ??
    detail.url ??
    detail.src ??
    detail.imageDataUrl;
  if (typeof directDataUrl === "string" && directDataUrl.trim().startsWith("data:image/")) {
    return { type: "data_url", value: directDataUrl.trim() };
  }

  const binaryCandidates = [
    detail.buffer,
    detail.bytes,
    detail.binary,
    detail.jpeg,
    detail.jpg,
    detail.png,
  ];
  for (const candidate of binaryCandidates) {
    const normalizedBinary = normalizePreviewPayload(candidate, depth + 1);
    if (normalizedBinary) return normalizedBinary;
  }

  const nestedCandidates = [
    detail.detail,
    detail.data,
    detail.blob,
    detail.preview,
    detail.image,
    detail.payload,
    detail.output,
    detail.output?.preview,
    detail.output?.image,
    detail.output?.images?.[0],
    detail.preview_image,
    detail.unencoded_preview_image,
    detail.latents,
  ];
  for (const candidate of nestedCandidates) {
    const normalized = normalizePreviewPayload(candidate, depth + 1);
    if (normalized) return normalized;
  }
  return null;
}

async function normalizePreviewPayloadFromEvent(detail, eventType = "") {
  if (eventType === "b_preview" && detail instanceof Blob) {
    return {
      previewId: "",
      payload: { type: "blob", value: detail },
    };
  }

  return {
    previewId: "",
    payload: normalizePreviewPayload(detail),
  };
}

function blobToDataUrl(blob) {
  return new Promise((resolve, reject) => {
    if (!blob) {
      resolve("");
      return;
    }
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result || ""));
    reader.onerror = () => reject(reader.error || new Error("Failed to read preview blob."));
    reader.readAsDataURL(blob);
  });
}

function emitJobProgress(tracker, promptIndex) {
  if (!tracker) return;
  if (promptIndex < 0 || promptIndex >= tracker.total) return;
  const promptId = String(tracker.promptIds[promptIndex] || "");
  const progress = getTrackerProgress(tracker, promptIndex);
  const key = trackerProgressKey(tracker.requestId, promptIndex);
  const progressSignature = `${progress.step}:${progress.maxStep}`;
  if (trackerProgressSentByKey.get(key) === progressSignature) return;
  trackerProgressSentByKey.set(key, progressSignature);
  sendWs({
    type: "job_progress",
    requestId: tracker.requestId,
    promptIndex,
    promptId,
    progress: progress.step,
    progressMax: progress.maxStep,
  });
}

function emitGenerationPreview(tracker, promptIndex, imageDataUrl) {
  if (!tracker) return;
  if (promptIndex < 0 || promptIndex >= tracker.total) return;
  const image = String(imageDataUrl || "").trim();
  if (!image) return;
  const prompt = String(tracker.prompts[promptIndex] || "");
  const promptId = String(tracker.promptIds[promptIndex] || "");
  const progress = getTrackerProgress(tracker, promptIndex);
  sendWs({
    type: "generation_preview",
    requestId: tracker.requestId,
    promptIndex,
    prompt,
    promptId,
    imageDataUrl: image,
    step: progress.step,
    maxStep: progress.maxStep,
    updatedAt: Date.now(),
  });
}

function emitQueueProgress(tracker, promptIndex, source) {
  const prompt = String(tracker.prompts[promptIndex] || "");
  const promptId = String(tracker.promptIds[promptIndex] || "");
  const seed = Number(tracker.promptSeeds[promptIndex] || 0);
  sendWs({
    type: "queue_progress",
    requestId: tracker.requestId,
    promptIndex,
    prompt,
    promptId,
    seed,
    completed: tracker.completedCount,
    total: tracker.total,
    source,
  });
}

function finalizeTracker(tracker, reason = "completed") {
  if (!tracker) return;
  if (currentQueueExecution && String(currentQueueExecution.requestId || "") === String(tracker.requestId || "")) {
    currentQueueExecution = null;
  }
  clearIdleFallbackTimer(tracker.requestId);
  sendWs({
    type: "job_idle",
    requestId: tracker.requestId,
    completed: tracker.completedCount,
    total: tracker.total,
    reason,
  });
  clearTrackerProgress(tracker.requestId);
  activeQueueTrackers.delete(tracker.requestId);
}

function cancelTrackerByRequestId(requestId, reason = "canceled") {
  const key = String(requestId || "").trim();
  if (!key) return;
  if (currentQueueExecution && String(currentQueueExecution.requestId || "") === key) {
    currentQueueExecution = null;
  }
  const tracker = activeQueueTrackers.get(key);
  if (tracker) {
    finalizeTracker(tracker, reason);
    return;
  }
  clearTrackerProgress(key);
  activeQueueTrackers.delete(key);
}

function markActivePromptInterrupted() {
  const activeRequestId = String(currentQueueExecution?.requestId || "").trim();
  const activePromptIndex = Number.isFinite(currentQueueExecution?.promptIndex)
    ? Math.max(0, Math.floor(currentQueueExecution.promptIndex))
    : -1;

  if (activeRequestId) {
    const tracker = activeQueueTrackers.get(activeRequestId);
    if (tracker && activePromptIndex >= 0 && activePromptIndex < tracker.total) {
      return markPromptCompleted(tracker, activePromptIndex, "interrupted");
    }
  }

  const resolved = resolveTrackerForPromptId(currentExecutingPromptId || "");
  if (!resolved) return false;
  return markPromptCompleted(resolved.tracker, resolved.promptIndex, "interrupted");
}

function markPromptCompleted(tracker, promptIndex, source) {
  if (!tracker) return false;
  if (promptIndex < 0 || promptIndex >= tracker.total) return false;
  if (tracker.completedFlags[promptIndex]) return false;
  const existingProgress = getTrackerProgress(tracker, promptIndex);
  if (existingProgress.maxStep > 0) {
    setTrackerProgress(tracker, promptIndex, existingProgress.maxStep, existingProgress.maxStep);
  }
  tracker.completedFlags[promptIndex] = true;
  tracker.completedCount += 1;
  if (source === "save_output") tracker.completedBySaveCount += 1;
  emitQueueProgress(tracker, promptIndex, source);
  if (tracker.completedCount >= tracker.total) {
    finalizeTracker(tracker, source === "save_output" ? "save_output_complete" : "completed");
  }
  return true;
}

function applyIdleFallbackToTracker(tracker) {
  if (!tracker) return;
  for (let idx = 0; idx < tracker.total; idx += 1) {
    if (tracker.completedFlags[idx]) continue;
    markPromptCompleted(tracker, idx, "idle_fallback");
  }
  if (activeQueueTrackers.has(tracker.requestId)) {
    finalizeTracker(tracker, "idle_fallback");
  }
}

function handleComfyExecuted(event) {
  queueManager?.handleComfyExecuted(event);
}

function handleComfyExecutionSuccess(event) {
  queueManager?.handleComfyExecutionSuccess(event);
}

function handleComfyProgress(event) {
  queueManager?.handleComfyProgress(event);
}

async function handleComfyPreview(event) {
  await queueManager?.handleComfyPreview(event);
}

function handleComfyIdle(event) {
  queueManager?.handleComfyIdle(event);
}

function waitForPromptCompletion(requestId, promptIndex, timeoutMs = PROMPT_COMPLETION_TIMEOUT_MS) {
  return new Promise((resolve) => {
    const id = String(requestId || "").trim();
    const index = Number.isFinite(promptIndex) ? Math.max(0, Math.floor(promptIndex)) : 0;
    const startedAt = Date.now();
    const poll = () => {
      if (!id) {
        resolve("missing_request");
        return;
      }
      if (isQueueCancelRequestedFor(id)) {
        resolve("canceled");
        return;
      }
      const tracker = activeQueueTrackers.get(id);
      if (!tracker) {
        resolve("tracker_missing");
        return;
      }
      if (tracker.completedFlags[index]) {
        if (currentQueueExecution && String(currentQueueExecution.requestId || "") === id && Number(currentQueueExecution.promptIndex) === index) {
          currentQueueExecution = null;
        }
        resolve("completed");
        return;
      }
      if (Date.now() - startedAt >= Math.max(5000, Math.floor(Number(timeoutMs) || PROMPT_COMPLETION_TIMEOUT_MS))) {
        resolve("timeout");
        return;
      }
      setTimeout(poll, 120);
    };
    poll();
  });
}

function ensureExecutionListeners() {
  if (executionListenersAttached) return;
  executionListenersAttached = true;
  try {
    api.addEventListener("executed", handleComfyExecuted);
  } catch {
    // best effort
  }
  try {
    api.addEventListener("execution_success", handleComfyExecutionSuccess);
  } catch {
    // best effort
  }
  try {
    api.addEventListener("progress", handleComfyProgress);
  } catch {
    // best effort
  }
  try {
    for (const eventName of PREVIEW_EVENT_NAMES) {
      api.addEventListener(eventName, handleComfyPreview);
    }
  } catch {
    // best effort
  }
  try {
    for (const eventName of PREVIEW_EVENT_NAMES) {
      window.addEventListener(eventName, handleComfyPreview);
      document.addEventListener(eventName, handleComfyPreview);
    }
  } catch {
    // best effort
  }
  try {
    api.addEventListener("executing", handleComfyIdle);
  } catch {
    // best effort
  }
}

function createPrompterWsUrl() {
  const protocol = window.location.protocol === "https:" ? "wss" : "ws";
  const host = window.location.hostname || "127.0.0.1";
  return `${protocol}://${host}:8212/ws/prompter`;
}

function cleanPromptText(value) {
  return String(value || "").replace(/^\s*\[(?:x|X| )?\]\s*/, "").trim();
}

function normalizePrompts(value, options) {
  const dedupe = options?.dedupe !== false;
  if (!Array.isArray(value)) return [];
  const next = [];
  const unique = new Set();
  for (const raw of value) {
    const text = cleanPromptText(raw);
    if (!text) continue;
    if (!dedupe) {
      next.push(text);
      continue;
    }
    unique.add(text);
  }
  return dedupe ? Array.from(unique) : next;
}

function clampQueueSetId(rawSetId, fallback = 1) {
  const next = Number(rawSetId);
  if (!Number.isFinite(next)) return Math.max(1, Math.min(MAX_QUEUE_SETS, Math.floor(Number(fallback) || 1)));
  return Math.max(1, Math.min(MAX_QUEUE_SETS, Math.floor(next)));
}

function normalizePromptSetIds(rawPromptSetIds, promptCount, fallbackSetId = 1) {
  const total = Math.max(0, Math.floor(Number(promptCount) || 0));
  const fallback = clampQueueSetId(fallbackSetId, 1);
  if (total <= 0) return [];
  if (!Array.isArray(rawPromptSetIds) || rawPromptSetIds.length <= 0) {
    return Array.from({ length: total }, () => fallback);
  }
  const normalized = [];
  for (let idx = 0; idx < total; idx += 1) {
    normalized.push(clampQueueSetId(rawPromptSetIds[idx], fallback));
  }
  return normalized;
}

function normalizeGenerationState(value) {
  const incoming = value && typeof value === "object" ? value : {};
  const numeric = (raw, fallback, min, max) => {
    const next = Number(raw);
    if (!Number.isFinite(next)) return fallback;
    return Math.max(min, Math.min(max, Math.floor(next)));
  };
  const numericFloat = (raw, fallback, min, max) => {
    const next = Number(raw);
    if (!Number.isFinite(next)) return fallback;
    return Math.max(min, Math.min(max, next));
  };
  const normalizeDetailerBoolean = (raw, fallback) => {
    if (raw === undefined || raw === null) return fallback;
    if (typeof raw === "string") {
      const normalized = raw.trim().toLowerCase();
      if (["true", "on", "enabled"].includes(normalized)) return true;
      if (["false", "off", "disabled"].includes(normalized)) return false;
    }
    return raw === true;
  };
  const normalizeDetailerDimension = (raw, fallback) => {
    const clamped = numeric(raw, fallback, 64, 16384);
    return Math.max(64, Math.round(clamped / 8) * 8);
  };
  const normalizeDetailerPipeline = (rawPipeline, rawLegacyStages) => {
    const legacyStages = rawLegacyStages && typeof rawLegacyStages === "object"
      ? rawLegacyStages
      : null;
    const source = Array.isArray(rawPipeline)
      ? rawPipeline
      : DEFAULT_DETAILER_PIPELINE.map((stage) => ({
        ...stage,
        enabled: legacyStages
          ? normalizeDetailerBoolean(
            legacyStages[stage.label === "Eyes" ? "eyes" : stage.label.toLowerCase()],
            stage.enabled
          )
          : stage.enabled,
      }));
    const usedIds = new Set();
    const samHints = new Set([
      "center-1", "horizontal-2", "vertical-2", "rect-4", "diamond-4",
      "mask-area", "mask-points", "mask-point-bbox", "none",
    ]);

    return source.flatMap((entry, index) => {
      if (!entry || typeof entry !== "object") return [];
      const detectorModel = String(entry.detectorModel || "").trim().replace(/\\/g, "/");
      const preset = DEFAULT_DETAILER_PIPELINE.find((candidate) => (
        candidate.detectorModel.toLowerCase() === detectorModel.toLowerCase()
        || candidate.label.toLowerCase() === String(entry.label || "").trim().toLowerCase()
      )) || DEFAULT_DETAILER_PIPELINE[1] || DEFAULT_DETAILER_PIPELINE[0];
      let id = String(entry.id || `detail-stage-${index + 1}`).trim().slice(0, 128) || `detail-stage-${index + 1}`;
      if (usedIds.has(id)) id = `${id}-${index + 1}`;
      usedIds.add(id);
      const samDeviceRaw = String(entry.samDeviceMode || preset.samDeviceMode).trim();
      const samDeviceMode = samDeviceRaw === "CPU"
        ? "CPU"
        : samDeviceRaw === "Prefer GPU" ? "Prefer GPU" : "AUTO";
      const samNegativeRaw = String(entry.samMaskHintUseNegative || preset.samMaskHintUseNegative).trim();
      const samMaskHintUseNegative = samNegativeRaw === "Small"
        ? "Small"
        : samNegativeRaw === "Outter" ? "Outter" : "False";
      const samDetectionHint = String(entry.samDetectionHint || preset.samDetectionHint).trim();
      return [{
        id,
        enabled: normalizeDetailerBoolean(entry.enabled, preset.enabled),
        label: String(entry.label || preset.label).trim().slice(0, 80) || preset.label,
        detectorModel: detectorModel || preset.detectorModel,
        guideSize: normalizeDetailerDimension(entry.guideSize, preset.guideSize),
        guideSizeFor: String(entry.guideSizeFor || "").trim().toLowerCase() === "crop_region" ? "crop_region" : "bbox",
        maxSize: normalizeDetailerDimension(entry.maxSize, preset.maxSize),
        seedOffset: numeric(entry.seedOffset, index + 1, 0, 1000000),
        steps: numeric(entry.steps, preset.steps, 1, 10000),
        cfg: numericFloat(entry.cfg, preset.cfg, 0, 100),
        samplerName: String(entry.samplerName || preset.samplerName).trim() || preset.samplerName,
        scheduler: String(entry.scheduler || preset.scheduler).trim() || preset.scheduler,
        denoise: numericFloat(entry.denoise, preset.denoise, 0.0001, 1),
        feather: numeric(entry.feather, preset.feather, 0, 100),
        noiseMask: normalizeDetailerBoolean(entry.noiseMask, preset.noiseMask),
        forceInpaint: normalizeDetailerBoolean(entry.forceInpaint, preset.forceInpaint),
        bboxThreshold: numericFloat(entry.bboxThreshold, preset.bboxThreshold, 0, 1),
        bboxDilation: numeric(entry.bboxDilation, preset.bboxDilation, -512, 512),
        bboxCropFactor: numericFloat(entry.bboxCropFactor, preset.bboxCropFactor, 1, 10),
        useSam: normalizeDetailerBoolean(entry.useSam, preset.useSam),
        samModel: String(entry.samModel || preset.samModel).trim().replace(/\\/g, "/"),
        samDeviceMode,
        samDetectionHint: samHints.has(samDetectionHint) ? samDetectionHint : preset.samDetectionHint,
        samDilation: numeric(entry.samDilation, preset.samDilation, -512, 512),
        samThreshold: numericFloat(entry.samThreshold, preset.samThreshold, 0, 1),
        samBboxExpansion: numeric(entry.samBboxExpansion, preset.samBboxExpansion, 0, 1000),
        samMaskHintThreshold: numericFloat(entry.samMaskHintThreshold, preset.samMaskHintThreshold, 0, 1),
        samMaskHintUseNegative,
        dropSize: numeric(entry.dropSize, preset.dropSize, 1, 16384),
        wildcard: String(entry.wildcard || "").replace(/\r\n/g, "\n"),
        cycle: numeric(entry.cycle, preset.cycle, 1, 10),
        noiseMaskFeather: numeric(entry.noiseMaskFeather, preset.noiseMaskFeather, 0, 100),
        tiledEncode: normalizeDetailerBoolean(entry.tiledEncode, preset.tiledEncode),
        tiledDecode: normalizeDetailerBoolean(entry.tiledDecode, preset.tiledDecode),
      }];
    });
  };
  const modeRaw = String(
    incoming.controlAfterGenerate ??
    incoming.control_after_generate ??
    ""
  ).trim().toLowerCase();
  let controlAfterGenerate = DEFAULT_GENERATION_STATE.controlAfterGenerate;
  if (["fixed", "increment", "decrement", "randomize"].includes(modeRaw)) {
    controlAfterGenerate = modeRaw;
  } else if (["1", "true", "yes", "on"].includes(modeRaw)) {
    controlAfterGenerate = "increment";
  } else if (["0", "false", "no", "off"].includes(modeRaw)) {
    controlAfterGenerate = "fixed";
  }
  const samplerName = String(incoming.samplerName || "").trim() || DEFAULT_GENERATION_STATE.samplerName;
  const scheduler = String(incoming.scheduler || "").trim() || DEFAULT_GENERATION_STATE.scheduler;
  const aspectRatio = String(incoming.aspectRatio || "").trim() || DEFAULT_GENERATION_STATE.aspectRatio;
  const modelType = normalizeGenerationModelType(incoming.modelType ?? incoming.model_type);
  const swapDimensions = incoming.swapDimensions === true || String(incoming.swapDimensions || "").trim().toLowerCase() === "on";
  const normalizeLoraName = (rawName) => String(rawName || "").trim().replace(/\\/g, "/");
  const normalizeLoraStrength = (rawValue, fallback) => {
    const next = Number(rawValue);
    if (!Number.isFinite(next)) return fallback;
    return Math.max(-10, Math.min(10, next));
  };
  const normalizeLoraQueueSetIds = (rawSetIds, fallbackEnabled = true) => {
    if (!Array.isArray(rawSetIds)) return fallbackEnabled ? [1] : [];
    const normalized = Array.from(new Set(
      rawSetIds
        .map((entry) => Number(entry))
        .filter((entry) => Number.isFinite(entry))
        .map((entry) => Math.floor(entry))
        .filter((entry) => entry >= 1 && entry <= MAX_QUEUE_SETS)
    )).sort((a, b) => a - b);
    if (normalized.length === 0 && fallbackEnabled) return [1];
    return normalized;
  };
  const normalizeLoras = (rawLoras) => {
    if (!Array.isArray(rawLoras)) return [];
    const result = [];
    const seenIds = new Set();
    for (const rawEntry of rawLoras) {
      if (!rawEntry || typeof rawEntry !== "object") continue;
      const name = normalizeLoraName(rawEntry.name);
      if (!name) continue;
      const id = String(rawEntry.id || "").trim() || `pp-lora-${Date.now()}-${Math.random().toString(36).slice(2, 9)}`;
      if (seenIds.has(id)) continue;
      seenIds.add(id);
      const strengthModel = normalizeLoraStrength(rawEntry.strengthModel, 1.0);
      const strengthClip = normalizeLoraStrength(rawEntry.strengthClip, strengthModel);
      const queueEnabled = rawEntry.queueEnabled !== false;
      const queueSetIds = normalizeLoraQueueSetIds(rawEntry.queueSetIds, queueEnabled);
      result.push({
        id,
        name,
        strengthModel,
        strengthClip,
        enabled: rawEntry.enabled !== false,
        queueEnabled: queueSetIds.length > 0,
        queueSetIds,
      });
      if (result.length >= 24) break;
    }
    return result;
  };
  const normalizeThumbnailOverrides = (rawOverrides) => {
    if (!rawOverrides || typeof rawOverrides !== "object" || Array.isArray(rawOverrides)) return {};
    const normalized = {};
    for (const [rawKey, rawValue] of Object.entries(rawOverrides)) {
      const key = String(rawKey || "").trim().replace(/\\/g, "/").replace(/^\/+/, "").toLowerCase();
      if (!key) continue;
      const values = (Array.isArray(rawValue) ? rawValue : [rawValue])
        .map((entry) => String(entry || "").trim())
        .filter((entry) => entry.length > 0 && entry.length <= 2200000)
        .filter((entry) => (
          entry.startsWith("data:image/") ||
          entry.startsWith("data:video/") ||
          entry.startsWith("/api/fs/read?path=") ||
          entry.startsWith("/api/fs/image?path=") ||
          entry.startsWith("/api/fs/thumbnail?path=")
        ));
      if (values.length === 0) continue;
      normalized[key] = Array.from(new Set(values)).slice(0, 12);
      if (Object.keys(normalized).length >= 200) break;
    }
    return normalized;
  };
  return {
    detailerPipeline: normalizeDetailerPipeline(
      incoming.detailerPipeline,
      incoming.umbraUiDetailStages
    ),
    negativePrompt: cleanPromptText(incoming.negativePrompt ?? incoming.negative_prompt ?? ""),
    seed: numeric(incoming.seed, DEFAULT_GENERATION_STATE.seed, 0, Number.MAX_SAFE_INTEGER),
    controlAfterGenerate,
    incrementStep: numeric(
      incoming.incrementStep ?? incoming.increment_step,
      DEFAULT_GENERATION_STATE.incrementStep,
      1,
      Number.MAX_SAFE_INTEGER
    ),
    steps: numeric(incoming.steps, DEFAULT_GENERATION_STATE.steps, 1, 10000),
    cfg: numericFloat(incoming.cfg, DEFAULT_GENERATION_STATE.cfg, 0, 100),
    samplerName,
    scheduler,
    modelType,
    checkpointName: String(incoming.checkpointName || "").trim().replace(/\\/g, "/"),
    aspectRatio,
    swapDimensions,
    width: numeric(incoming.width, DEFAULT_GENERATION_STATE.width, 64, 8192),
    height: numeric(incoming.height, DEFAULT_GENERATION_STATE.height, 64, 8192),
    batchSize: numeric(incoming.batchSize, DEFAULT_GENERATION_STATE.batchSize, 1, 64),
    loras: normalizeLoras(incoming.loras),
    thumbnailOverrides: normalizeThumbnailOverrides(incoming.thumbnailOverrides),
  };
}

function normalizeGenerationModelType(rawType) {
  const candidate = String(rawType || "").trim().toLowerCase().replace(/[\s-]+/g, "_");
  if (candidate === "checkpoints" || candidate === "ckpt" || candidate === "checkpoint_loader") return "checkpoint";
  if (candidate === "diffuser" || candidate === "diffusers_model") return "diffusers";
  if (candidate === "diffusion_models") return "diffusion_model";
  if (candidate === "unets" || candidate === "unet_model") return "unet";
  if (candidate === "checkpoint" || candidate === "diffusers" || candidate === "diffusion_model" || candidate === "unet") return candidate;
  return "checkpoint";
}

function stripModelFolderPrefixForType(modelName, modelType) {
  const normalized = String(modelName || "").trim().replace(/\\/g, "/").replace(/^\/+/, "");
  const prefixes = {
    checkpoint: ["checkpoints"],
    diffusers: ["diffusers"],
    diffusion_model: ["diffusion_models"],
    unet: ["unet"],
  };
  const parts = normalized.split("/").filter(Boolean);
  if (parts.length <= 1) return normalized;
  const allowed = prefixes[modelType] || [];
  if (!allowed.includes(String(parts[0] || "").toLowerCase())) return normalized;
  return parts.slice(1).join("/");
}

function applyGenerationModelToApiNode(node, generation) {
  const modelType = normalizeGenerationModelType(generation?.modelType ?? generation?.model_type);
  const modelName = stripModelFolderPrefixForType(generation?.checkpointName, modelType);
  setApiNodeInput(node, "model_type", modelType);
  setApiNodeInput(node, "weight_dtype", "default");
  setApiNodeInput(node, "checkpoint_name", modelType === "checkpoint" ? (modelName || "[None]") : "[None]");
  setApiNodeInput(node, "diffusers_model", modelType === "diffusers" ? modelName : "");
  setApiNodeInput(node, "diffusion_model_name", modelType === "diffusion_model" ? modelName : "");
  setApiNodeInput(node, "unet_name", modelType === "unet" ? modelName : "");
  setApiNodeInput(node, "gguf_name", "");
}

function applyGenerationModelToWidgetNode(node, generation) {
  const modelType = normalizeGenerationModelType(generation?.modelType ?? generation?.model_type);
  const modelName = stripModelFolderPrefixForType(generation?.checkpointName, modelType);
  setWidgetChoiceValue(node, "model_type", modelType);
  setWidgetChoiceValue(node, "weight_dtype", "default");
  setWidgetChoiceValue(node, "checkpoint_name", modelType === "checkpoint" ? modelName : "");
  setWidgetChoiceValue(node, "diffusers_model", modelType === "diffusers" ? modelName : "");
  setWidgetChoiceValue(node, "diffusion_model_name", modelType === "diffusion_model" ? modelName : "");
  setWidgetChoiceValue(node, "unet_name", modelType === "unet" ? modelName : "");
  setWidgetChoiceValue(node, "gguf_name", "");
}

function randomPositiveSeed() {
  return Math.max(1, Math.floor(Math.random() * MAX_SAFE_SEED));
}

function resolveSeedForQueueRun(generation, promptIndex) {
  const cleanGeneration = normalizeGenerationState(generation);
  const idx = Number.isFinite(promptIndex) ? Math.max(0, Math.floor(promptIndex)) : 0;
  const baseSeed = Math.max(0, Math.floor(Number(cleanGeneration.seed) || 0));
  const step = Math.max(1, Math.floor(Number(cleanGeneration.incrementStep) || 1));
  const mode = String(cleanGeneration.controlAfterGenerate || "fixed").trim().toLowerCase();

  if (mode === "randomize") {
    return randomPositiveSeed();
  }
  if (mode === "increment") {
    return Math.max(0, Math.min(MAX_SAFE_SEED, baseSeed + (idx * step)));
  }
  if (mode === "decrement") {
    return Math.max(0, baseSeed - (idx * step));
  }
  return baseSeed;
}

function getPrompterNodes() {
  const graphNodes = app?.graph?._nodes;
  if (!Array.isArray(graphNodes)) return [];
  return graphNodes.filter((node) => String(node?.type || "") === NODE_CLASS);
}

function getUnifiedPowerPrompterNodes() {
  const graphNodes = app?.graph?._nodes;
  if (!Array.isArray(graphNodes)) return [];
  return graphNodes.filter((node) => String(node?.type || "") === UNIFIED_NODE_CLASS);
}

function getKSamplerNodes() {
  const graphNodes = app?.graph?._nodes;
  if (!Array.isArray(graphNodes)) return [];
  return graphNodes.filter((node) => KSAMPLER_NODE_CLASSES.has(String(node?.type || "")));
}

function getSeedValueNodes() {
  const graphNodes = app?.graph?._nodes;
  if (!Array.isArray(graphNodes)) return [];
  return graphNodes.filter((node) => SEED_VALUE_NODE_CLASSES.has(String(node?.type || "")));
}

function getCheckpointNodes() {
  const graphNodes = app?.graph?._nodes;
  if (!Array.isArray(graphNodes)) return [];
  return graphNodes.filter((node) => CHECKPOINT_NODE_CLASSES.has(String(node?.type || "")));
}

function getLoraSyntaxNodes() {
  const graphNodes = app?.graph?._nodes;
  if (!Array.isArray(graphNodes)) return [];
  return graphNodes.filter((node) => String(node?.type || "") === LORA_NODE_CLASS);
}

function getSaveNodes() {
  const graphNodes = app?.graph?._nodes;
  if (!Array.isArray(graphNodes)) return [];
  return graphNodes.filter((node) => SAVE_NODE_TYPES.has(String(node?.type || "")));
}

function sanitizeWorkflowName(rawValue) {
  return String(rawValue || "")
    .replace(/\s+/g, " ")
    .trim()
    .slice(0, 120);
}

function getWorkflowName() {
  const candidates = [
    app?.graph?.extra?.workflow?.name,
    app?.graph?.extra?.name,
    app?.graph?.name,
    app?.graph?.filename,
    typeof document !== "undefined" ? document.title : "",
  ];
  for (const candidate of candidates) {
    const value = sanitizeWorkflowName(candidate);
    if (value) return value;
  }
  return "Comfy Workflow";
}

function getWorkflowFingerprint() {
  const workflowName = getWorkflowName();
  const nodeCount = Array.isArray(app?.graph?._nodes) ? app.graph._nodes.length : 0;
  return `${workflowName}|${nodeCount}`;
}

function emitBridgeState(force = false) {
  const validation = validateQueueWorkflow();
  const compatible = validation?.ok === true;
  const missing = Array.isArray(validation?.missing) ? validation.missing.filter(Boolean) : [];
  const payload = {
    type: "bridge_state",
    bridgeId: BRIDGE_ID,
    workflowId: getWorkflowFingerprint(),
    workflowName: getWorkflowName(),
    compatible,
    missing,
    updatedAt: Date.now(),
  };
  const signature = `${payload.workflowId}|${payload.workflowName}|${compatible ? "1" : "0"}|${missing.join("|")}`;
  if (!force && signature === lastBridgeStateSignature) return;
  if (sendWs(payload)) {
    lastBridgeStateSignature = signature;
  }
}

function startBridgeStateHeartbeat() {
  if (bridgeStateTimer) return;
  bridgeStateTimer = setInterval(() => {
    emitBridgeState(false);
  }, BRIDGE_STATE_HEARTBEAT_MS);
}

function stopBridgeStateHeartbeat() {
  if (!bridgeStateTimer) return;
  clearInterval(bridgeStateTimer);
  bridgeStateTimer = null;
}

function validateQueueWorkflow() {
  const missing = [];
  const hasUnifiedNode = getUnifiedPowerPrompterNodes().length > 0;
  if (!hasUnifiedNode && getPrompterNodes().length <= 0) missing.push(NODE_CLASS);
  if (!hasUnifiedNode && getKSamplerNodes().length <= 0) missing.push("UmbraKSampler/UmbraKSamplerNormal");
  if (getSaveNodes().length <= 0) missing.push("UmbraLabSaveImage");
  return {
    ok: missing.length <= 0,
    missing,
  };
}

function cloneSerializable(value) {
  if (typeof structuredClone === "function") {
    try {
      return structuredClone(value);
    } catch {
      // fallback below
    }
  }
  return JSON.parse(JSON.stringify(value));
}

function extractApiWorkflowPromptGraph(rawWorkflow) {
  const isPromptGraph = (value) => {
    if (!value || typeof value !== "object" || Array.isArray(value)) return false;
    const entries = Object.values(value);
    if (entries.length <= 0) return false;
    return entries.every((entry) => (
      !!entry
      && typeof entry === "object"
      && !Array.isArray(entry)
      && String(entry.class_type || "").trim().length > 0
    ));
  };

  if (isPromptGraph(rawWorkflow)) return rawWorkflow;
  if (rawWorkflow && typeof rawWorkflow === "object" && isPromptGraph(rawWorkflow.prompt)) {
    return rawWorkflow.prompt;
  }
  if (rawWorkflow && typeof rawWorkflow === "object" && isPromptGraph(rawWorkflow.output)) {
    return rawWorkflow.output;
  }
  return null;
}

function validateApiWorkflow(rawWorkflow) {
  const promptGraph = extractApiWorkflowPromptGraph(rawWorkflow);
  if (!promptGraph) {
    return { ok: false, missing: ["prompt graph"] };
  }
  const classTypes = new Set(
    Object.values(promptGraph)
      .map((entry) => String(entry?.class_type || "").trim())
      .filter((entry) => entry.length > 0)
  );
  const missing = [];
  const hasUnifiedNode = classTypes.has(UNIFIED_NODE_CLASS);
  if (!hasUnifiedNode) {
    if (!classTypes.has(NODE_CLASS)) missing.push(NODE_CLASS);
    if (!classTypes.has("UmbraKSampler") && !classTypes.has("UmbraKSamplerNormal")) {
      missing.push("UmbraKSampler/UmbraKSamplerNormal");
    }
  }
  if (!classTypes.has("UmbraLabSaveImage")) {
    missing.push("UmbraLabSaveImage");
  }
  return {
    ok: missing.length <= 0,
    missing,
  };
}

function collectApiWorkflowSaveNodeIds(rawWorkflow) {
  const promptGraph = extractApiWorkflowPromptGraph(rawWorkflow);
  if (!promptGraph) return [];
  return Object.entries(promptGraph)
    .filter(([, node]) => SAVE_NODE_TYPES.has(String(node?.class_type || "").trim()))
    .map(([nodeId]) => String(nodeId || "").trim())
    .filter((nodeId) => nodeId.length > 0);
}

function ensureApiNodeInputs(node) {
  if (!node || typeof node !== "object") return null;
  if (!node.inputs || typeof node.inputs !== "object" || Array.isArray(node.inputs)) {
    node.inputs = {};
  }
  return node.inputs;
}

function setApiNodeInput(node, inputName, value) {
  const inputs = ensureApiNodeInputs(node);
  if (!inputs) return;
  inputs[inputName] = value;
}

function extractApiWorkflowMetadataPayload(rawWorkflow, promptGraph) {
  if (rawWorkflow && typeof rawWorkflow === "object") {
    const directWorkflow = cloneSerializable(
      rawWorkflow.workflow
      ?? rawWorkflow?.extra_data?.extra_pnginfo?.workflow
      ?? rawWorkflow?.extra_pnginfo?.workflow
    );
    if (directWorkflow && typeof directWorkflow === "object") {
      return directWorkflow;
    }
  }
  void promptGraph;
  return null;
}

function normalizePowerPrompterSourceFile(value) {
  const trimmed = String(value || "").trim();
  if (!trimmed) return "";
  return trimmed.replace(/\\/g, "/").replace(/\/+$/, "").replace(/\.ppcards\.json$/i, "");
}

function buildQueuedApiWorkflow(rawWorkflow, state, options = {}) {
  const cloned = cloneSerializable(rawWorkflow);
  const promptGraph = extractApiWorkflowPromptGraph(cloned);
  if (!promptGraph) {
    throw new Error("Selected API workflow does not contain a valid prompt graph.");
  }

  const prompts = normalizePrompts(state?.prompts || []);
  const activePrompt = cleanPromptText(options?.prompt || state?.activePrompt || prompts[0] || "");
  const generation = normalizeGenerationState(options?.generation || state?.generation);
  const promptSetId = clampQueueSetId(
    options?.promptSetId ?? state?.activeQueueSet ?? latestSyncState.activeQueueSet ?? 1,
    1
  );
  const outputSubfolder = String(options?.outputSubfolder || "").trim().replace(/\\/g, "/");
  const setLabel = `Set ${promptSetId}`;
  const styleLabel = outputSubfolder;
  const styleSeedMode = String(options?.styleSeedMode || state?.styleSeedMode || "same").trim().toLowerCase() === "different" ? "different" : "same";
  const shouldFreezeRepeatSeeds = styleSeedMode === "same" && outputSubfolder.length > 0;

  for (const node of Object.values(promptGraph)) {
    const classType = String(node?.class_type || "").trim();
    if (!classType) continue;

    if (classType === UNIFIED_NODE_CLASS) {
      setApiNodeInput(node, "prompt_text", activePrompt);
      setApiNodeInput(node, "negative_prompt", generation.negativePrompt || "");
      applyGenerationModelToApiNode(node, generation);
      setApiNodeInput(node, "seed", generation.seed);
      setApiNodeInput(node, "control_after_generate", generation.controlAfterGenerate || "fixed");
      setApiNodeInput(node, "increment_step", generation.incrementStep);
      setApiNodeInput(node, "style_seed_behavior", shouldFreezeRepeatSeeds ? "same_seed_style_cycle" : "normal");
      setApiNodeInput(node, "aspect_ratio", generation.aspectRatio);
      setApiNodeInput(node, "swap_dimensions", generation.swapDimensions ? "On" : "Off");
      setApiNodeInput(node, "width", generation.width);
      setApiNodeInput(node, "height", generation.height);
      setApiNodeInput(node, "batch_size", generation.batchSize);
      setApiNodeInput(node, "steps", generation.steps);
      setApiNodeInput(node, "cfg", generation.cfg);
      setApiNodeInput(node, "sampler_name", generation.samplerName);
      setApiNodeInput(node, "scheduler", generation.scheduler);
      continue;
    }

    if (classType === NODE_CLASS) {
      setApiNodeInput(node, "prompt_text", activePrompt);
      setApiNodeInput(node, "negative_prompt", generation.negativePrompt || "");
      setApiNodeInput(node, "seed", generation.seed);
      setApiNodeInput(node, "control_after_generate", generation.controlAfterGenerate || "fixed");
      setApiNodeInput(node, "increment_step", generation.incrementStep);
      setApiNodeInput(node, "style_seed_behavior", shouldFreezeRepeatSeeds ? "same_seed_style_cycle" : "normal");
      setApiNodeInput(node, "aspect_ratio", generation.aspectRatio);
      setApiNodeInput(node, "swap_dimensions", generation.swapDimensions ? "On" : "Off");
      setApiNodeInput(node, "width", generation.width);
      setApiNodeInput(node, "height", generation.height);
      setApiNodeInput(node, "batch_size", generation.batchSize);
      continue;
    }

    if (KSAMPLER_NODE_CLASSES.has(classType)) {
      if (classType === "KSamplerAdvanced") {
        setApiNodeInput(node, "noise_seed", generation.seed);
      } else {
        setApiNodeInput(node, "seed", generation.seed);
      }
      setApiNodeInput(node, "steps", generation.steps);
      setApiNodeInput(node, "cfg", generation.cfg);
      setApiNodeInput(node, "sampler_name", generation.samplerName);
      setApiNodeInput(node, "scheduler", generation.scheduler);
      if (shouldFreezeRepeatSeeds) {
        setApiNodeInput(node, "repeat_behavior", "none");
        setApiNodeInput(node, "style_seed_behavior", "same_seed_style_cycle");
      } else {
        setApiNodeInput(node, "style_seed_behavior", "normal");
      }
      continue;
    }

    if (SEED_VALUE_NODE_CLASSES.has(classType)) {
      setApiNodeInput(node, "seed", generation.seed);
      if (shouldFreezeRepeatSeeds) {
        setApiNodeInput(node, "repeat_behavior", "none");
        setApiNodeInput(node, "style_seed_behavior", "same_seed_style_cycle");
      } else {
        setApiNodeInput(node, "style_seed_behavior", "normal");
      }
      continue;
    }

    if (classType === "UmbraImageDetailer") {
      const pipeline = Array.isArray(generation.detailerPipeline)
        ? generation.detailerPipeline
        : [];
      const stageEnabled = (label) => pipeline.some((stage) => (
        stage?.enabled === true && String(stage?.label || "").trim().toLowerCase() === label
      ));
      setApiNodeInput(node, "pipeline_json", JSON.stringify(pipeline));
      setApiNodeInput(node, "person_detail", stageEnabled("person"));
      setApiNodeInput(node, "face_detail", stageEnabled("face"));
      setApiNodeInput(node, "eye_detail", stageEnabled("eyes"));
      setApiNodeInput(node, "hand_detail", stageEnabled("hands"));
      continue;
    }

    if (CHECKPOINT_NODE_CLASSES.has(classType)) {
      const checkpointName = String(generation.checkpointName || "").trim();
      if (classType === "UmbraLoadCheckpoint") {
        applyGenerationModelToApiNode(node, generation);
      } else if (classType === "CheckpointLoaderSimple" && generation.modelType === "checkpoint" && checkpointName) {
        setApiNodeInput(node, "ckpt_name", checkpointName);
      }
      continue;
    }

    if (classType === LORA_NODE_CLASS) {
      setApiNodeInput(node, "prompt_text", activePrompt);
      setApiNodeInput(node, "lora_syntax_text", "");
      setApiNodeInput(node, "lora_name", LORA_NONE_OPTION);
      continue;
    }

    if (SAVE_NODE_TYPES.has(classType)) {
      setApiNodeInput(node, "save_to_yyyy_mm_dd_folder", true);
      setApiNodeInput(node, "save_to_set_subfolder", true);
      setApiNodeInput(node, "set_subfolder", setLabel);
      setApiNodeInput(node, "save_set_to_style_subfolder", styleLabel);
    }
  }

  const workflowPayload = extractApiWorkflowMetadataPayload(cloned, promptGraph);

  return {
    promptGraph,
    workflowPayload,
  };
}

function normalizeLoraNameForTag(rawName) {
  const normalized = String(rawName || "").trim().replace(/\\/g, "/");
  if (!normalized || normalized === LORA_NONE_OPTION) return "";
  return normalized.replace(/\.[^/.]+$/, "");
}

function formatLoraTag(name, strengthModel = 1.0, strengthClip = 1.0) {
  const normalizedName = normalizeLoraNameForTag(name);
  if (!normalizedName) return "";
  const modelStrength = Number(strengthModel);
  const clipStrength = Number(strengthClip);
  const safeModel = Number.isFinite(modelStrength) ? modelStrength : 1.0;
  const safeClip = Number.isFinite(clipStrength) ? clipStrength : safeModel;
  return `<lora:${normalizedName}:${safeModel}:${safeClip}>`;
}

function getEnabledLoras(generation, activeQueueSet = 1) {
  const loras = Array.isArray(generation?.loras) ? generation.loras : [];
  const targetSet = Math.max(1, Math.min(MAX_QUEUE_SETS, Math.floor(Number(activeQueueSet) || 1)));
  return loras
    .filter((entry) => entry && entry.enabled !== false)
    .filter((entry) => {
      const queueSetIds = Array.isArray(entry.queueSetIds) ? entry.queueSetIds : [1];
      return queueSetIds.includes(targetSet);
    })
    .map((entry) => ({
      id: String(entry.id || "").trim() || `pp-lora-${Math.random().toString(36).slice(2, 9)}`,
      rawName: String(entry.name || "").trim().replace(/\\/g, "/"),
      tagName: normalizeLoraNameForTag(entry.name),
      strengthModel: Number(entry.strengthModel),
      strengthClip: Number(entry.strengthClip),
    }))
    .filter((entry) => entry.tagName);
}

function buildPromptTextWithGenerationLoras(promptText, generation, activeQueueSet = 1) {
  const cleanPrompt = cleanPromptText(promptText);
  // Prompt text is the LoRA source of truth. LoRA controls can track metadata,
  // but the bridge must not auto-enable a LoRA unless the queued prompt carries
  // its explicit <lora:...> tag.
  void generation;
  void activeQueueSet;
  return cleanPrompt;
}

function findWidget(node, widgetName) {
  if (!node || !Array.isArray(node.widgets)) return null;
  return node.widgets.find((widget) => String(widget?.name || "") === widgetName) || null;
}

function setWidgetValue(node, widgetName, nextValue) {
  const widget = findWidget(node, widgetName);
  if (!widget) return false;
  widget.value = nextValue;
  try {
    if (typeof widget.callback === "function") {
      widget.callback(nextValue, app, node);
    }
  } catch {
    // Keep assignment resilient.
  }
  try {
    if (typeof node.onWidgetChanged === "function") {
      node.onWidgetChanged(widgetName, nextValue, null, widget);
    }
  } catch {
    // Best effort only.
  }
  return true;
}

function setWidgetChoiceValue(node, widgetName, nextValue) {
  const widget = findWidget(node, widgetName);
  if (!widget) return false;
  const candidate = String(nextValue || "").trim();
  const values = Array.isArray(widget?.options?.values)
    ? widget.options.values.map((entry) => String(entry || ""))
    : [];
  if (values.length > 0) {
    if (!candidate || !values.includes(candidate)) {
      return setWidgetValue(node, widgetName, values[0]);
    }
  }
  return setWidgetValue(node, widgetName, candidate);
}

function markCanvasDirty() {
  try {
    if (typeof app?.graph?.setDirtyCanvas === "function") {
      app.graph.setDirtyCanvas(true, true);
    }
    if (typeof app?.canvas?.setDirty === "function") {
      app.canvas.setDirty(true, true);
    }
  } catch {
    // Best effort only.
  }
}

function applySyncToNode(node, state, options = {}) {
  if (!node) return;
  const prompts = normalizePrompts(state?.prompts || []);
  const activePrompt = cleanPromptText(state?.activePrompt || prompts[0] || "");
  const generation = normalizeGenerationState(state?.generation);
  const outputSubfolder = String(options?.outputSubfolder || "").trim();
  const shouldFreezeStyleSeed = String(state?.styleSeedMode || "same").trim().toLowerCase() === "same" && outputSubfolder.length > 0;
  setWidgetValue(node, "prompt_text", activePrompt);
  setWidgetValue(node, "negative_prompt", generation.negativePrompt || "");
  setWidgetValue(node, "seed", generation.seed);
  setWidgetValue(node, "control_after_generate", generation.controlAfterGenerate);
  setWidgetValue(node, "increment_step", generation.incrementStep);
  setWidgetChoiceValue(node, "style_seed_behavior", shouldFreezeStyleSeed ? "same_seed_style_cycle" : "normal");
  setWidgetValue(node, "aspect_ratio", generation.aspectRatio);
  setWidgetValue(node, "swap_dimensions", generation.swapDimensions ? "On" : "Off");
  setWidgetValue(node, "width", generation.width);
  setWidgetValue(node, "height", generation.height);
  setWidgetValue(node, "batch_size", generation.batchSize);
}

function applySyncToUnifiedPowerPrompterNode(node, state, options = {}) {
  if (!node) return;
  applySyncToNode(node, state, options);
  applySyncToCheckpointNode(node, state);
  applySyncToKSamplerNode(node, state, options);
}

function applySyncToKSamplerNode(node, state, options = {}) {
  if (!node) return;
  const generation = normalizeGenerationState(state?.generation);
  const classType = String(node?.type || "").trim();
  if (classType === "KSamplerAdvanced") {
    setWidgetValue(node, "noise_seed", generation.seed);
  } else {
    setWidgetValue(node, "seed", generation.seed);
  }
  setWidgetValue(node, "steps", generation.steps);
  setWidgetValue(node, "cfg", generation.cfg);
  setWidgetChoiceValue(node, "sampler_name", generation.samplerName);
  setWidgetChoiceValue(node, "scheduler", generation.scheduler);
  const outputSubfolder = String(options?.outputSubfolder || "").trim();
  if (String(state?.styleSeedMode || "same").trim().toLowerCase() === "same" && outputSubfolder.length > 0) {
    setWidgetChoiceValue(node, "repeat_behavior", "none");
    setWidgetChoiceValue(node, "style_seed_behavior", "same_seed_style_cycle");
  } else {
    setWidgetChoiceValue(node, "style_seed_behavior", "normal");
  }
}

function applySyncToSeedValueNode(node, state, options = {}) {
  if (!node) return;
  const generation = normalizeGenerationState(state?.generation);
  setWidgetValue(node, "seed", generation.seed);
  const outputSubfolder = String(options?.outputSubfolder || "").trim();
  if (String(state?.styleSeedMode || "same").trim().toLowerCase() === "same" && outputSubfolder.length > 0) {
    setWidgetChoiceValue(node, "repeat_behavior", "none");
    setWidgetChoiceValue(node, "style_seed_behavior", "same_seed_style_cycle");
  } else {
    setWidgetChoiceValue(node, "style_seed_behavior", "normal");
  }
}

function applySyncToCheckpointNode(node, state) {
  if (!node) return;
  const generation = normalizeGenerationState(state?.generation);
  const checkpointName = String(generation.checkpointName || "").trim();
  const nodeType = String(node?.type || "").trim();
  if (nodeType === "UmbraLoadCheckpoint" || nodeType === UNIFIED_NODE_CLASS) {
    applyGenerationModelToWidgetNode(node, generation);
    return;
  }
  if (nodeType === "CheckpointLoaderSimple" && generation.modelType === "checkpoint" && checkpointName) {
    setWidgetChoiceValue(node, "ckpt_name", checkpointName);
  }
}

function applySyncToLoraNode(node, state) {
  if (!node) return;
  const prompts = normalizePrompts(state?.prompts || []);
  const activePrompt = cleanPromptText(state?.activePrompt || prompts[0] || "");
  setWidgetValue(node, "prompt_text", activePrompt);
  // Keep node stateless: parse only explicit <lora:...> tags from prompt_text.
  setWidgetValue(node, "lora_syntax_text", "");
  setWidgetValue(node, "lora_name", LORA_NONE_OPTION);
}

function applySyncToSaveNode(node, state, options = {}) {
  if (!node) return;
  const activeSetId = clampQueueSetId(
    options?.promptSetId ?? state?.activeQueueSet ?? latestSyncState.activeQueueSet ?? 1,
    1
  );
  const outputSubfolder = String(options?.outputSubfolder || "").trim().replace(/\\/g, "/");
  const setLabel = `Set ${activeSetId}`;
  const styleLabel = outputSubfolder;
  setWidgetValue(node, "save_to_yyyy_mm_dd_folder", true);
  setWidgetValue(node, "save_to_set_subfolder", true);
  setWidgetValue(node, "set_subfolder", setLabel);
  setWidgetValue(node, "save_set_to_style_subfolder", styleLabel);
}

function applySyncToAllNodes(state, options = {}) {
  const unifiedNodes = getUnifiedPowerPrompterNodes();
  for (const node of unifiedNodes) {
    applySyncToUnifiedPowerPrompterNode(node, state, options);
  }
  const readerNodes = getPrompterNodes();
  for (const node of readerNodes) {
    applySyncToNode(node, state, options);
  }
  const checkpointNodes = getCheckpointNodes();
  for (const node of checkpointNodes) {
    applySyncToCheckpointNode(node, state);
  }
  const kSamplerNodes = getKSamplerNodes();
  for (const node of kSamplerNodes) {
    applySyncToKSamplerNode(node, state, options);
  }
  const seedValueNodes = getSeedValueNodes();
  for (const node of seedValueNodes) {
    applySyncToSeedValueNode(node, state, options);
  }
  const loraNodes = getLoraSyntaxNodes();
  for (const node of loraNodes) {
    applySyncToLoraNode(node, state);
  }
  const saveNodes = getSaveNodes();
  for (const node of saveNodes) {
    applySyncToSaveNode(node, state, options);
  }
  markCanvasDirty();
}

function hasMeaningfulSyncState(state) {
  if (!state || typeof state !== "object") return false;
  if (normalizePrompts(state?.prompts || []).length > 0) return true;
  if (cleanPromptText(state?.activePrompt || "")) return true;
  if (cleanPromptText(state?.generation?.negativePrompt ?? state?.generation?.negative_prompt ?? "")) return true;
  return false;
}

function applySyncToNewNode(node, state, syncCallback) {
  if (!node || typeof syncCallback !== "function") return;
  if (!hasMeaningfulSyncState(state)) return;
  syncCallback(node, state);
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function queueCurrentWorkflow(options = {}) {
  const queueTargetType = String(options?.queueTargetType || "").trim();
  const sourceFile = normalizePowerPrompterSourceFile(options?.state?.sourceFile);
  if (queueTargetType === "api_workflow") {
      const queuedApiWorkflow = buildQueuedApiWorkflow(options?.apiWorkflow, options?.state || latestSyncState, {
        prompt: options?.prompt,
        generation: options?.state?.generation,
        promptSetId: options?.promptSetId,
        outputSubfolder: options?.outputSubfolder,
        styleSeedMode: options?.state?.styleSeedMode,
    });
    const promptGraph = queuedApiWorkflow?.promptGraph || {};
    const workflowPayload = queuedApiWorkflow?.workflowPayload ?? null;
    const extraPngInfo = {
      ...(workflowPayload ? { workflow: workflowPayload } : {}),
      ...(sourceFile ? { source_file: sourceFile } : {}),
    };
    const nextClientId = String(
      api?.clientId ||
      api?.client_id ||
      app?.clientId ||
      app?.client_id ||
      ""
    ).trim();
    const response = await fetch("/prompt", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        client_id: nextClientId,
        prompt: promptGraph,
        extra_data: {
          extra_pnginfo: extraPngInfo,
          preview_method: POWER_PROMPTER_PREVIEW_METHOD,
        },
      }),
    });
    if (!response.ok) {
      const detail = await response.text().catch(() => "");
      throw new Error(detail || `ComfyUI rejected API workflow queue (${response.status}).`);
    }
    return await response.json().catch(() => ({}));
  }

  let graphPayload = null;
  if (typeof app?.graphToPrompt === "function") {
    graphPayload = await app.graphToPrompt();
  }

  let graphPayloadWithMetadata = graphPayload;
  if (graphPayload && typeof graphPayload === "object") {
    const nextExtraData = graphPayload?.extra_data && typeof graphPayload.extra_data === "object"
      ? { ...graphPayload.extra_data }
      : {};
    const nextExtraPngInfo = nextExtraData.extra_pnginfo && typeof nextExtraData.extra_pnginfo === "object"
      ? { ...nextExtraData.extra_pnginfo }
      : {};
    if (!nextExtraPngInfo.workflow && graphPayload?.workflow) {
      nextExtraPngInfo.workflow = graphPayload.workflow;
    }
    if (sourceFile) {
      nextExtraPngInfo.source_file = sourceFile;
    }
    nextExtraData.extra_pnginfo = nextExtraPngInfo;
    nextExtraData.preview_method = POWER_PROMPTER_PREVIEW_METHOD;
    graphPayloadWithMetadata = {
      ...graphPayload,
      extra_data: nextExtraData,
    };
  }

  if (typeof api?.queuePrompt === "function" && graphPayloadWithMetadata) {
    try {
      return await Promise.resolve(api.queuePrompt(0, graphPayloadWithMetadata, {
        previewMethod: POWER_PROMPTER_PREVIEW_METHOD,
      }));
    } catch {
      // fallback below
    }
  }

  if (graphPayloadWithMetadata) {
    try {
      const nextClientId = String(
        api?.clientId ||
        api?.client_id ||
        app?.clientId ||
        app?.client_id ||
        graphPayloadWithMetadata?.client_id ||
        ""
      ).trim();
      const nextExtraData = graphPayloadWithMetadata?.extra_data && typeof graphPayloadWithMetadata.extra_data === "object"
        ? { ...graphPayloadWithMetadata.extra_data }
        : {};
      const nextExtraPngInfo = nextExtraData.extra_pnginfo && typeof nextExtraData.extra_pnginfo === "object"
        ? { ...nextExtraData.extra_pnginfo }
        : {};
      if (!nextExtraPngInfo.workflow && graphPayloadWithMetadata?.workflow) {
        nextExtraPngInfo.workflow = graphPayloadWithMetadata.workflow;
      }
      if (sourceFile) {
        nextExtraPngInfo.source_file = sourceFile;
      }
      nextExtraData.extra_pnginfo = nextExtraPngInfo;
      nextExtraData.preview_method = POWER_PROMPTER_PREVIEW_METHOD;
      const response = await fetch("/prompt", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          client_id: nextClientId,
          prompt: graphPayloadWithMetadata?.output ?? graphPayloadWithMetadata?.prompt ?? {},
          extra_data: nextExtraData,
        }),
      });
      if (response.ok) {
        return await response.json().catch(() => ({}));
      }
    } catch {
      // fallback below
    }
  }

  if (typeof app?.queuePrompt === "function") {
    return await Promise.resolve(app.queuePrompt(0));
  }

  const queueButton = document.querySelector(
    "#queue-button, button[title*='Queue Prompt'], button.comfyui-button"
  );
  if (queueButton && typeof queueButton.click === "function") {
    queueButton.click();
    return;
  }

  throw new Error("Queue action is unavailable in this ComfyUI build.");
}

function sendWs(payload) {
  if (!prompterWs || prompterWs.readyState !== WebSocket.OPEN) return false;
  try {
    prompterWs.send(JSON.stringify(payload));
    return true;
  } catch {
    return false;
  }
}

queueManager = createPowerPrompterQueueManager({
  sendWs,
  sleep,
  normalizePrompts,
  normalizeGenerationState,
  cleanPromptText,
  resolveSeedForQueueRun,
  clampQueueSetId,
  latestSyncState,
  applySyncToAllNodes,
  queueCurrentWorkflow,
  saveNodeTypes: SAVE_NODE_TYPES,
  maxQueueSets: MAX_QUEUE_SETS,
  idleFallbackDelayMs: IDLE_FALLBACK_DELAY_MS,
  queueSubmitBatchSize: QUEUE_SUBMIT_BATCH_SIZE,
  queueSubmitBetweenPromptsMs: QUEUE_SUBMIT_BETWEEN_PROMPTS_MS,
  queueSubmitBetweenBatchesMs: QUEUE_SUBMIT_BETWEEN_BATCHES_MS,
  promptCompletionTimeoutMs: PROMPT_COMPLETION_TIMEOUT_MS,
  previewFrameThrottleMs: PREVIEW_FRAME_THROTTLE_MS,
  previewMaxDataUrlLength: PREVIEW_MAX_DATA_URL_LENGTH,
  extractPromptIdFromExecution,
  extractNodeTypeFromExecution,
  payloadContainsImageOutput,
  extractProgressValue,
  normalizePreviewPayload,
  normalizePreviewPayloadFromEvent,
  blobToDataUrl,
  hasSaveNodes: () => getSaveNodes().length > 0,
  validateQueueWorkflow,
  validateApiWorkflow,
});

function getLoraCatalogItems() {
  const options = new Set();
  for (const node of getLoraSyntaxNodes()) {
    const widget = findWidget(node, "lora_name") || findWidget(node, "selected_lora");
    if (!widget) continue;
    const candidates = Array.isArray(widget?.options?.values)
      ? widget.options.values
      : Array.isArray(widget?.options)
        ? widget.options
        : [];
    for (const rawValue of candidates) {
      const value = String(rawValue || "").trim().replace(/\\/g, "/");
      if (!value || value === LORA_NONE_OPTION) continue;
      options.add(value);
    }
  }
  return Array.from(options).sort((a, b) => a.localeCompare(b));
}

function normalizeLoraCatalogItem(rawValue, assumeSafetensors = false) {
  const value = String(rawValue || "").trim().replace(/\\/g, "/").replace(/^\/+/, "");
  if (!value || value === LORA_NONE_OPTION) return "";
  if (assumeSafetensors && !/\.(safetensors|ckpt|pt|pth)$/i.test(value)) {
    return `${value}.safetensors`;
  }
  return value;
}

async function getMergedLoraCatalogItems() {
  const options = new Set(getLoraCatalogItems().map((entry) => normalizeLoraCatalogItem(entry)).filter(Boolean));
  const liveHints = await fetchLiveLoraCatalogHints(true);
  for (const hint of liveHints) {
    const value = normalizeLoraCatalogItem(hint, true);
    if (value) options.add(value);
  }
  return Array.from(options).sort((a, b) => a.localeCompare(b));
}

function getWidgetChoiceCandidates(widget) {
  if (!widget) return [];
  return Array.isArray(widget?.options?.values)
    ? widget.options.values
    : Array.isArray(widget?.options)
      ? widget.options
      : [];
}

function addModelCatalogWidgetOptions(options, node, widgetName, modelType) {
  const widget = findWidget(node, widgetName);
  for (const rawValue of getWidgetChoiceCandidates(widget)) {
    const value = String(rawValue || "").trim().replace(/\\/g, "/");
    if (!value || value === "[None]") continue;
    const firstSegment = value.split("/").filter(Boolean)[0]?.toLowerCase() || "";
    const routedType = modelType === "diffusion_model" && firstSegment === "unet" ? "unet" : modelType;
    options.add(`${routedType}|${value}`);
  }
}

function getObjectInfoChoiceCandidates(objectInfo, nodeType, inputName) {
  const nodeInfo = objectInfo?.[nodeType];
  const inputInfo = nodeInfo?.input;
  const sections = [
    inputInfo?.required,
    inputInfo?.optional,
  ];

  for (const section of sections) {
    const rawInput = section?.[inputName];
    if (!Array.isArray(rawInput)) continue;
    const choices = rawInput[0];
    if (Array.isArray(choices)) return choices;
  }
  return [];
}

function addModelCatalogObjectInfoOptions(options, objectInfo, nodeType, inputName, modelType) {
  for (const rawValue of getObjectInfoChoiceCandidates(objectInfo, nodeType, inputName)) {
    const value = String(rawValue || "").trim().replace(/\\/g, "/");
    if (!value || value === "[None]") continue;
    const firstSegment = value.split("/").filter(Boolean)[0]?.toLowerCase() || "";
    const routedType = modelType === "diffusion_model" && firstSegment === "unet" ? "unet" : modelType;
    options.add(`${routedType}|${value}`);
  }
}

function getModelCatalogItems() {
  const options = new Set();
  for (const node of getCheckpointNodes()) {
    const nodeType = String(node?.type || "").trim();
    if (nodeType === "UmbraLoadCheckpoint") {
      addModelCatalogWidgetOptions(options, node, "checkpoint_name", "checkpoint");
      addModelCatalogWidgetOptions(options, node, "diffusers_model", "diffusers");
      addModelCatalogWidgetOptions(options, node, "diffusion_model_name", "diffusion_model");
      addModelCatalogWidgetOptions(options, node, "unet_name", "unet");
    } else {
      addModelCatalogWidgetOptions(options, node, "ckpt_name", "checkpoint");
    }
  }
  for (const node of getUnifiedPowerPrompterNodes()) {
    addModelCatalogWidgetOptions(options, node, "checkpoint_name", "checkpoint");
    addModelCatalogWidgetOptions(options, node, "diffusers_model", "diffusers");
    addModelCatalogWidgetOptions(options, node, "diffusion_model_name", "diffusion_model");
    addModelCatalogWidgetOptions(options, node, "unet_name", "unet");
  }
  return Array.from(options).sort((a, b) => a.localeCompare(b));
}

async function fetchLiveModelCatalogHints(forceRefresh = false) {
  if (!forceRefresh && liveModelCatalogHintsCache.loaded) {
    return Array.isArray(liveModelCatalogHintsCache.items) ? [...liveModelCatalogHintsCache.items] : [];
  }

  try {
    const response = await api.fetchApi("/object_info", { cache: "no-store" });
    if (!response || response.status !== 200) {
      liveModelCatalogHintsCache.loaded = true;
      liveModelCatalogHintsCache.items = [];
      return [];
    }

    const objectInfo = await response.json().catch(() => null);
    const options = new Set();
    addModelCatalogObjectInfoOptions(options, objectInfo, "UmbraLoadCheckpoint", "checkpoint_name", "checkpoint");
    addModelCatalogObjectInfoOptions(options, objectInfo, "UmbraLoadCheckpoint", "diffusers_model", "diffusers");
    addModelCatalogObjectInfoOptions(options, objectInfo, "UmbraLoadCheckpoint", "diffusion_model_name", "diffusion_model");
    addModelCatalogObjectInfoOptions(options, objectInfo, "UmbraLoadCheckpoint", "unet_name", "unet");
    addModelCatalogObjectInfoOptions(options, objectInfo, UNIFIED_NODE_CLASS, "checkpoint_name", "checkpoint");
    addModelCatalogObjectInfoOptions(options, objectInfo, UNIFIED_NODE_CLASS, "diffusers_model", "diffusers");
    addModelCatalogObjectInfoOptions(options, objectInfo, UNIFIED_NODE_CLASS, "diffusion_model_name", "diffusion_model");
    addModelCatalogObjectInfoOptions(options, objectInfo, UNIFIED_NODE_CLASS, "unet_name", "unet");
    addModelCatalogObjectInfoOptions(options, objectInfo, "CheckpointLoaderSimple", "ckpt_name", "checkpoint");

    const items = Array.from(options).sort((a, b) => a.localeCompare(b));
    liveModelCatalogHintsCache.loaded = true;
    liveModelCatalogHintsCache.items = items;
    return items;
  } catch {
    liveModelCatalogHintsCache.loaded = true;
    liveModelCatalogHintsCache.items = [];
    return [];
  }
}

async function getMergedModelCatalogItems() {
  const options = new Set(getModelCatalogItems());
  const liveHints = await fetchLiveModelCatalogHints(true);
  for (const hint of liveHints) {
    const value = String(hint || "").trim().replace(/\\/g, "/");
    if (value) options.add(value);
  }
  return Array.from(options).sort((a, b) => a.localeCompare(b));
}

async function fetchLiveLoraCatalogHints(forceRefresh = false) {
  if (!forceRefresh && liveLoraCatalogHintsCache.loaded) {
    return Array.isArray(liveLoraCatalogHintsCache.items) ? [...liveLoraCatalogHintsCache.items] : [];
  }
  try {
    const response = await api.fetchApi("/pysssss/loras", { cache: "no-store" });
    if (!response || response.status !== 200) {
      liveLoraCatalogHintsCache.loaded = true;
      liveLoraCatalogHintsCache.items = [];
      return [];
    }
    const payload = await response.json().catch(() => []);
    if (!Array.isArray(payload)) {
      liveLoraCatalogHintsCache.loaded = true;
      liveLoraCatalogHintsCache.items = [];
      return [];
    }
    const items = payload
      .map((entry) => String(entry || "").trim())
      .filter((entry) => entry.length > 0);
    liveLoraCatalogHintsCache.loaded = true;
    liveLoraCatalogHintsCache.items = items;
    return [...items];
  } catch {
    liveLoraCatalogHintsCache.loaded = true;
    liveLoraCatalogHintsCache.items = [];
    return [];
  }
}

function normalizeLoraMetadataName(rawName) {
  return String(rawName || "").trim().replace(/\\/g, "/").replace(/^\/+/, "");
}

function normalizeModelMetadataName(rawName) {
  return String(rawName || "").trim().replace(/\\/g, "/").replace(/^\/+/, "");
}

function buildModelMetadataCandidates(rawName) {
  const normalized = normalizeModelMetadataName(rawName);
  if (!normalized) return [];
  const parts = normalized.split("/").filter(Boolean);
  const basename = parts.length > 0 ? parts[parts.length - 1] : normalized;
  const noExtNormalized = removeLoraFileExtension(normalized);
  const noExtBasename = removeLoraFileExtension(basename);
  const knownType = parts.length > 1 && ["checkpoints", "diffusion_models", "unet", "diffusers"].includes(parts[0]?.toLowerCase())
    ? parts[0]
    : "";
  const relNormalized = knownType ? parts.slice(1).join("/") : normalized;
  const relNoExt = removeLoraFileExtension(relNormalized);
  const prefixed = knownType
    ? [normalized, noExtNormalized]
    : [
      `checkpoints/${relNormalized}`,
      `checkpoints/${relNoExt}`,
      `checkpoints/${basename}`,
      `checkpoints/${noExtBasename}`,
      `diffusion_models/${relNormalized}`,
      `diffusion_models/${relNoExt}`,
      `diffusion_models/${basename}`,
      `diffusion_models/${noExtBasename}`,
      `unet/${relNormalized}`,
      `unet/${relNoExt}`,
      `unet/${basename}`,
      `unet/${noExtBasename}`,
      `diffusers/${relNormalized}`,
      `diffusers/${relNoExt}`,
    ];
  const unique = [];
  const seen = new Set();
  for (const candidate of prefixed) {
    const clean = String(candidate || "").trim().replace(/\\/g, "/");
    if (!clean) continue;
    const key = clean.toLowerCase();
    if (seen.has(key)) continue;
    seen.add(key);
    unique.push(clean);
  }
  return unique;
}

function removeLoraFileExtension(rawName) {
  return String(rawName || "").replace(/\.(safetensors|ckpt|pt|pth|bin)$/i, "");
}

function getPathBasename(rawName) {
  const cleaned = String(rawName || "").trim().replace(/\\/g, "/");
  if (!cleaned) return "";
  const parts = cleaned.split("/").filter(Boolean);
  return parts.length > 0 ? parts[parts.length - 1] : cleaned;
}

function buildLoraMetadataCandidates(rawName, catalogHints = []) {
  const normalized = normalizeLoraMetadataName(rawName);
  if (!normalized) return [];
  const basename = getPathBasename(normalized);
  const noExtNormalized = removeLoraFileExtension(normalized);
  const noExtBasename = removeLoraFileExtension(basename);
  const requestedKey = noExtBasename.toLowerCase();
  const hasExplicitPath = normalized.includes("/");
  const candidates = hasExplicitPath
    ? [normalized, noExtNormalized]
    : [normalized, noExtNormalized, basename, noExtBasename];
  for (const rawHint of Array.isArray(catalogHints) ? catalogHints : []) {
    const hintNormalized = normalizeLoraMetadataName(rawHint);
    if (!hintNormalized) continue;
    const hintKey = removeLoraFileExtension(getPathBasename(hintNormalized)).toLowerCase();
    if (!requestedKey || hintKey !== requestedKey) continue;
    candidates.push(hintNormalized, removeLoraFileExtension(hintNormalized));
  }
  const unique = [];
  const seen = new Set();
  for (const candidate of candidates) {
    const clean = String(candidate || "").trim();
    if (!clean) continue;
    const normalizedVariant = String(clean || "").trim().replace(/^[/\\]+/, "").replace(/\\/g, "/");
    if (!normalizedVariant) continue;
    const key = normalizedVariant.toLowerCase();
    if (seen.has(key)) continue;
    seen.add(key);
    unique.push(normalizedVariant);
    if (unique.length >= 6) break;
  }
  return unique;
}

function extractLoraHash(metadata) {
  if (!metadata || typeof metadata !== "object") return "";
  const raw =
    metadata["pysssss.sha256"] ??
    metadata["easyuse.sha256"] ??
    metadata.sha256 ??
    metadata["sshs_model_hash"] ??
    metadata["ss_sd_model_hash"] ??
    metadata["modelspec.hash_sha256"] ??
    "";
  return String(raw || "").trim();
}

async function fetchLocalLoraMetadata(loraName) {
  const cacheKey = normalizeLoraMetadataName(loraName).toLowerCase();
  if (cacheKey && loraMetadataCache.has(cacheKey)) {
    return loraMetadataCache.get(cacheKey);
  }
  const widgetHints = getLoraCatalogItems();
  const liveHints = await fetchLiveLoraCatalogHints();
  const catalogHints = Array.from(new Set([...widgetHints, ...liveHints]));
  const candidates = buildLoraMetadataCandidates(loraName, catalogHints);
  if (candidates.length === 0) throw new Error("LoRA name is required.");

  const errors = [];
  for (const endpoint of LORA_METADATA_ENDPOINTS) {
    for (const candidate of candidates) {
      const rel = `loras/${candidate}`;
      const requestPath = endpoint + encodeURIComponent(rel);
      try {
        const response = await api.fetchApi(requestPath);
        if (!response || response.status !== 200) {
          errors.push(`${requestPath} -> ${response?.status ?? "unknown"}`);
          continue;
        }
        const payload = await response.json();
        if (!payload || typeof payload !== "object") {
          errors.push(`${requestPath} -> invalid json`);
          continue;
        }
        if (cacheKey) loraMetadataCache.set(cacheKey, payload);
        return payload;
      } catch (error) {
        errors.push(`${endpoint}${rel} -> ${String(error?.message || error || "request failed")}`);
      }
    }
  }

  const detail = errors.length > 0 ? ` (${errors.join("; ")})` : "";
  const unavailable = {
    "__umbra_metadata_unavailable": true,
    "__umbra_metadata_error": `metadata request failed${detail}`,
  };
  if (cacheKey) loraMetadataCache.set(cacheKey, unavailable);
  return unavailable;
}

async function fetchLocalModelMetadata(modelName) {
  const cacheKey = normalizeModelMetadataName(modelName).toLowerCase();
  if (cacheKey && modelMetadataCache.has(cacheKey)) {
    return modelMetadataCache.get(cacheKey);
  }
  const candidates = buildModelMetadataCandidates(modelName);
  if (candidates.length === 0) throw new Error("Model name is required.");

  const errors = [];
  for (const endpoint of LORA_METADATA_ENDPOINTS) {
    for (const candidate of candidates) {
      try {
        const response = await api.fetchApi(endpoint + encodeURIComponent(candidate));
        if (!response || response.status !== 200) {
          errors.push(`${endpoint}${candidate} -> ${response?.status ?? "unknown"}`);
          continue;
        }
        const payload = await response.json();
        if (!payload || typeof payload !== "object") {
          errors.push(`${endpoint}${candidate} -> invalid json`);
          continue;
        }
        if (cacheKey) modelMetadataCache.set(cacheKey, payload);
        return payload;
      } catch (error) {
        errors.push(`${endpoint}${candidate} -> ${String(error?.message || error || "request failed")}`);
      }
    }
  }

  const detail = errors.length > 0 ? ` (${errors.join("; ")})` : "";
  const unavailable = {
    "__umbra_metadata_unavailable": true,
    "__umbra_metadata_error": `metadata request failed${detail}`,
  };
  if (cacheKey) modelMetadataCache.set(cacheKey, unavailable);
  return unavailable;
}

function extractTrainedTags(civitai, metadata) {
  const direct = civitai?.trainedWords;
  if (Array.isArray(direct)) {
    const cleaned = direct.map((tag) => String(tag || "").trim()).filter(Boolean);
    if (cleaned.length > 0) return cleaned;
  }
  if (typeof direct === "string") {
    const parsed = direct.split(",").map((tag) => String(tag || "").trim()).filter(Boolean);
    if (parsed.length > 0) return parsed;
  }

  const metadataTags = metadata?.["modelspec.tags"];
  if (typeof metadataTags === "string") {
    return metadataTags
      .split(",")
      .map((tag) => String(tag || "").trim())
      .filter(Boolean)
      .slice(0, 120);
  }
  return [];
}

function extractCivitaiDescription(civitai) {
  const direct = String(civitai?.description || "").trim();
  if (direct) return direct;
  const model = civitai?.model;
  if (model && typeof model === "object") {
    const modelDescription = String(model.description || "").trim();
    if (modelDescription) return modelDescription;
  }
  return "";
}

function sanitizeRichDescriptionHtml(rawHtml) {
  const source = String(rawHtml || "").trim();
  if (!source) return "";
  if (typeof DOMParser === "undefined" || typeof document === "undefined") return "";

  try {
    const parser = new DOMParser();
    const parsed = parser.parseFromString(source, "text/html");
    const safeDoc = document.implementation.createHTMLDocument("");
    const safeRoot = safeDoc.createElement("div");

    const sanitizeNode = (node, parent) => {
      if (!node) return;
      if (node.nodeType === Node.TEXT_NODE) {
        parent.appendChild(safeDoc.createTextNode(String(node.textContent || "")));
        return;
      }
      if (node.nodeType !== Node.ELEMENT_NODE) return;

      const tag = String(node.tagName || "").toLowerCase();
      const children = Array.from(node.childNodes || []);
      if (!LORA_DESCRIPTION_ALLOWED_TAGS.has(tag)) {
        for (const child of children) sanitizeNode(child, parent);
        return;
      }

      const safeEl = safeDoc.createElement(tag);
      if (tag === "a") {
        const rawHref = String(node.getAttribute("href") || "").trim();
        if (rawHref) {
          try {
            const parsedHref = new URL(rawHref, window.location.origin);
            if (LORA_DESCRIPTION_ALLOWED_PROTOCOLS.has(parsedHref.protocol)) {
              safeEl.setAttribute("href", parsedHref.href);
              safeEl.setAttribute("target", "_blank");
              safeEl.setAttribute("rel", "noopener noreferrer");
            }
          } catch {
            // ignore invalid href
          }
        }
      }

      for (const child of children) sanitizeNode(child, safeEl);
      parent.appendChild(safeEl);
    };

    for (const child of Array.from(parsed.body.childNodes || [])) {
      sanitizeNode(child, safeRoot);
    }

    return String(safeRoot.innerHTML || "").trim();
  } catch {
    return "";
  }
}

function normalizeDescriptionText(rawHtml) {
  const source = String(rawHtml || "").trim();
  if (!source) return "";
  if (typeof DOMParser === "undefined") return source;
  try {
    const parser = new DOMParser();
    const parsed = parser.parseFromString(source, "text/html");
    return String(parsed.body?.textContent || "")
      .replace(/\r/g, "")
      .replace(/[ \t]+\n/g, "\n")
      .replace(/\n{3,}/g, "\n\n")
      .replace(/[ \t]{2,}/g, " ")
      .trim();
  } catch {
    return source;
  }
}

function buildDescriptionPayload(civitai) {
  const rawDescription = extractCivitaiDescription(civitai);
  if (!rawDescription) {
    return { descriptionHtml: "", descriptionText: "" };
  }
  const descriptionHtml = sanitizeRichDescriptionHtml(rawDescription);
  const descriptionText = normalizeDescriptionText(descriptionHtml || rawDescription);
  return { descriptionHtml, descriptionText };
}

function getCivitaiModelId(civitai) {
  const raw = civitai?.modelId ?? civitai?.model?.id ?? "";
  const parsed = Number.parseInt(String(raw || ""), 10);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : 0;
}

function mergeCivitaiModelDetails(civitai, modelDetails) {
  if (!civitai || !modelDetails || typeof modelDetails !== "object") return civitai;
  const existingModel = civitai.model && typeof civitai.model === "object" ? civitai.model : {};
  return {
    ...civitai,
    model: {
      ...modelDetails,
      ...existingModel,
      description: existingModel.description || modelDetails.description || "",
      images: Array.isArray(existingModel.images) ? existingModel.images : (Array.isArray(modelDetails.images) ? modelDetails.images : []),
    },
  };
}

async function fetchCivitaiModelDetails(modelId) {
  const safeId = getCivitaiModelId({ modelId });
  if (!safeId) return null;
  if (civitaiModelCache.has(safeId)) return civitaiModelCache.get(safeId);
  const response = await fetch(`https://civitai.com/api/v1/models/${safeId}`);
  if (response.status !== 200) {
    if (response.status === 404) civitaiModelCache.set(safeId, null);
    return null;
  }
  const payload = await response.json().catch(() => null);
  const value = payload && typeof payload === "object" ? payload : null;
  civitaiModelCache.set(safeId, value);
  return value;
}

async function hydrateCivitaiDescription(civitai) {
  if (!civitai || extractCivitaiDescription(civitai)) return civitai;
  const modelId = getCivitaiModelId(civitai);
  if (!modelId) return civitai;
  try {
    const details = await fetchCivitaiModelDetails(modelId);
    return mergeCivitaiModelDetails(civitai, details);
  } catch {
    return civitai;
  }
}

function slimCivitaiImage(entry) {
  if (!entry || typeof entry !== "object") return null;
  const url = String(entry.url || "").trim();
  if (!url) return null;
  return {
    url,
    type: entry.type || "image",
    nsfw: entry.nsfw,
  };
}

function slimCivitaiPayload(civitai, maxImages = 4) {
  if (!civitai || typeof civitai !== "object") return null;
  const model = civitai.model && typeof civitai.model === "object" ? civitai.model : {};
  const images = Array.isArray(civitai.images)
    ? civitai.images.map(slimCivitaiImage).filter(Boolean).slice(0, maxImages)
    : [];
  const modelImages = images.length < maxImages && Array.isArray(model.images)
    ? model.images.map(slimCivitaiImage).filter(Boolean).slice(0, maxImages - images.length)
    : [];
  return {
    id: civitai.id,
    modelId: civitai.modelId ?? model.id,
    name: civitai.name,
    model: {
      id: model.id ?? civitai.modelId,
      name: model.name,
      type: model.type,
      stats: model.stats,
      images: modelImages,
    },
    images,
    trainedWords: Array.isArray(civitai.trainedWords) ? civitai.trainedWords.slice(0, 120) : civitai.trainedWords,
    stats: civitai.stats,
    url: civitai.url,
  };
}

async function fetchCivitaiByHash(hash) {
  const safeHash = String(hash || "").trim();
  if (!safeHash) return null;
  if (civitaiHashCache.has(safeHash)) {
    return civitaiHashCache.get(safeHash);
  }
  const response = await fetch(`https://civitai.com/api/v1/model-versions/by-hash/${safeHash}`);
  if (response.status === 200) {
    const payload = await response.json();
    civitaiHashCache.set(safeHash, payload);
    return payload;
  }
  if (response.status === 404) {
    civitaiHashCache.set(safeHash, null);
    return null;
  }
  throw new Error(`civitai request failed (${response.status})`);
}

function normalizeCivitaiLoraQuery(rawName) {
  const base = removeLoraFileExtension(getPathBasename(rawName))
    .replace(/[_-]+/g, " ")
    .replace(/[[\]{}()]+/g, " ")
    .replace(/\s+/g, " ")
    .trim();
  return base;
}

async function fetchCivitaiByName(loraName) {
  const query = normalizeCivitaiLoraQuery(loraName);
  if (!query) return null;
  const response = await fetch(`https://civitai.com/api/v1/models?types=LORA&query=${encodeURIComponent(query)}&limit=5`);
  if (response.status !== 200) return null;
  const payload = await response.json().catch(() => null);
  const items = Array.isArray(payload?.items) ? payload.items : [];
  for (const item of items) {
    if (!item || typeof item !== "object") continue;
    const versions = Array.isArray(item.modelVersions) ? item.modelVersions : [];
    if (versions.length === 0) continue;
    const version = versions[0];
    if (!version || typeof version !== "object") continue;
    return {
      ...version,
      model: item,
    };
  }
  return null;
}

async function handleLoraCatalogRequest(message) {
  const requestId = String(message?.requestId || "");
  if (!requestId) return;
  try {
    sendWs({
      type: "lora_catalog_result",
      requestId,
      success: true,
      items: await getMergedLoraCatalogItems(),
    });
  } catch (error) {
    sendWs({
      type: "lora_catalog_result",
      requestId,
      success: false,
      error: String(error?.message || error || "Failed to build LoRA catalog."),
      items: [],
    });
  }
}

async function handleModelCatalogRequest(message) {
  const requestId = String(message?.requestId || "");
  if (!requestId) return;
  try {
    sendWs({
      type: "model_catalog_result",
      requestId,
      success: true,
      items: await getMergedModelCatalogItems(),
    });
  } catch (error) {
    sendWs({
      type: "model_catalog_result",
      requestId,
      success: false,
      error: String(error?.message || error || "Failed to build model catalog."),
      items: [],
    });
  }
}

async function handleLoraInfoRequest(message) {
  const requestId = String(message?.requestId || "");
  const requestedLoraName = String(message?.loraName || "").trim();
  const previewOnly = message?.previewOnly === true;
  if (!requestId) return;
  if (!requestedLoraName) {
    sendWs({
      type: "lora_info_result",
      requestId,
      success: false,
      error: "LoRA name is required.",
    });
    return;
  }
  try {
    const metadata = await fetchLocalLoraMetadata(requestedLoraName);
    const hash = extractLoraHash(metadata);
    let civitai = null;
    if (hash) {
      try {
        civitai = await fetchCivitaiByHash(hash);
      } catch {
        civitai = null;
      }
    }
    if (!civitai) {
      try {
        civitai = await fetchCivitaiByName(requestedLoraName);
      } catch {
        civitai = null;
      }
    }
    if (!previewOnly) {
      civitai = await hydrateCivitaiDescription(civitai);
    }
    const trainedTags = extractTrainedTags(civitai, metadata);
    const { descriptionHtml, descriptionText } = previewOnly
      ? { descriptionHtml: "", descriptionText: "" }
      : buildDescriptionPayload(civitai);
    const responseCivitai = previewOnly ? slimCivitaiPayload(civitai) : civitai;
    sendWs({
      type: "lora_info_result",
      requestId,
      success: true,
      loraName: requestedLoraName,
      metadata: metadata || {},
      civitai: responseCivitai || null,
      trainedTags,
      descriptionHtml,
      descriptionText,
    });
  } catch (error) {
    sendWs({
      type: "lora_info_result",
      requestId,
      success: false,
      loraName: requestedLoraName,
      error: String(error?.message || error || "Failed to load LoRA info."),
    });
  }
}

async function handleModelInfoRequest(message) {
  const requestId = String(message?.requestId || "");
  const requestedModelName = String(message?.modelName || "").trim();
  const previewOnly = message?.previewOnly === true;
  if (!requestId) return;
  if (!requestedModelName) {
    sendWs({
      type: "model_info_result",
      requestId,
      success: false,
      error: "Model name is required.",
    });
    return;
  }
  try {
    const metadata = await fetchLocalModelMetadata(requestedModelName);
    const hash = extractLoraHash(metadata);
    let civitai = null;
    try {
      civitai = await fetchCivitaiByHash(hash);
    } catch {
      civitai = null;
    }
    if (!previewOnly) {
      civitai = await hydrateCivitaiDescription(civitai);
    }
    const trainedTags = extractTrainedTags(civitai, metadata);
    const { descriptionHtml, descriptionText } = previewOnly
      ? { descriptionHtml: "", descriptionText: "" }
      : buildDescriptionPayload(civitai);
    const responseCivitai = previewOnly ? slimCivitaiPayload(civitai) : civitai;
    sendWs({
      type: "model_info_result",
      requestId,
      success: true,
      modelName: requestedModelName,
      metadata: metadata || {},
      civitai: responseCivitai || null,
      trainedTags,
      descriptionHtml,
      descriptionText,
    });
  } catch (error) {
    sendWs({
      type: "model_info_result",
      requestId,
      success: false,
      modelName: requestedModelName,
      error: String(error?.message || error || "Failed to load model info."),
    });
  }
}

async function processQueueRequest(message) {
  await queueManager?.processQueueRequest?.(message);
}

async function pumpQueueRequests() {
  await queueManager?.pumpQueueRequests();
}

function handleQueueRequest(message) {
  if (String(message?.queueTargetType || "").trim() === "api_workflow") {
    queueManager?.handleQueueRequest({
      ...message,
      saveNodeIds: collectApiWorkflowSaveNodeIds(message?.apiWorkflow),
    });
    return;
  }
  queueManager?.handleQueueRequest(message);
}

function handleQueueCancelRequest(message) {
  queueManager?.handleQueueCancelRequest(message);
}

function handleQueuePauseToggle(nextPaused) {
  queueManager?.handleQueuePauseToggle(nextPaused);
}

function handleQueueInterruptActiveRequest(message) {
  queueManager?.handleQueueInterruptActiveRequest(message);
}

function handlePrompterMessage(event) {
  let message = null;
  try {
    message = JSON.parse(String(event?.data || "{}"));
  } catch {
    return;
  }
  if (!message || typeof message !== "object") return;

  if (message.type === "prompter_sync") {
    latestSyncState.prompts = normalizePrompts(message?.state?.prompts || []);
    latestSyncState.activePrompt = cleanPromptText(message?.state?.activePrompt || "");
    latestSyncState.joinedPrompt = String(message?.state?.joinedPrompt || "");
    latestSyncState.file = String(message?.state?.file || "");
    latestSyncState.activeQueueSet = Math.max(1, Math.min(MAX_QUEUE_SETS, Math.floor(Number(message?.state?.activeQueueSet) || 1)));
    latestSyncState.promptSetIds = normalizePromptSetIds(
      message?.state?.promptSetIds,
      latestSyncState.prompts.length,
      latestSyncState.activeQueueSet
    );
    latestSyncState.generation = normalizeGenerationState(message?.state?.generation || latestSyncState.generation);
    latestSyncState.styleSeedMode = String(message?.state?.styleSeedMode || latestSyncState.styleSeedMode || "same").trim().toLowerCase() === "different" ? "different" : "same";
    applySyncToAllNodes(latestSyncState);
    return;
  }

  if (message.type === "queue_request") {
    void handleQueueRequest(message);
    return;
  }

  if (message.type === "queue_cancel") {
    handleQueueCancelRequest(message);
    return;
  }

  if (message.type === "queue_pause") {
    handleQueuePauseToggle(true);
    return;
  }

  if (message.type === "queue_resume") {
    handleQueuePauseToggle(false);
    void pumpQueueRequests();
    return;
  }

  if (message.type === "queue_clear_future") {
    queueManager?.handleQueueClearFutureRequest(message);
    return;
  }

  if (message.type === "queue_delay_update") {
    queueManager?.handleQueueDispatchDelayUpdate(message);
    return;
  }

  if (message.type === "queue_reorder") {
    queueManager?.handleQueueReorder(message);
    return;
  }

  if (message.type === "queue_prompt_remove") {
    queueManager?.handleQueuePromptRemove(message);
    return;
  }

  if (message.type === "queue_interrupt_active") {
    handleQueueInterruptActiveRequest(message);
    return;
  }

  if (message.type === "lora_catalog_request") {
    void handleLoraCatalogRequest(message);
    return;
  }

  if (message.type === "lora_info_request") {
    void handleLoraInfoRequest(message);
    return;
  }

  if (message.type === "model_catalog_request") {
    void handleModelCatalogRequest(message);
    return;
  }

  if (message.type === "model_info_request") {
    void handleModelInfoRequest(message);
  }
}

function connectPrompterWs() {
  if (prompterWs && (prompterWs.readyState === WebSocket.OPEN || prompterWs.readyState === WebSocket.CONNECTING)) {
    return;
  }

  const wsUrl = createPrompterWsUrl();
  const ws = new WebSocket(wsUrl);
  prompterWs = ws;
  prompterWsStatus = "connecting";
  markCanvasDirty();

  ws.onopen = () => {
    prompterWsStatus = "connected";
    markCanvasDirty();
    const validation = validateQueueWorkflow();
    const compatible = validation?.ok === true;
    const missing = Array.isArray(validation?.missing) ? validation.missing.filter(Boolean) : [];
    sendWs({
      type: "register",
      role: WS_ROLE,
      source: "comfyui",
      bridgeId: BRIDGE_ID,
      workflowId: getWorkflowFingerprint(),
      workflowName: getWorkflowName(),
      compatible,
      missing,
    });
    emitBridgeState(true);
    startBridgeStateHeartbeat();
    if (latestSyncState.prompts.length > 0 || latestSyncState.activePrompt) {
      applySyncToAllNodes(latestSyncState);
    }
  };

  ws.onmessage = handlePrompterMessage;

  ws.onclose = () => {
    if (prompterWs === ws) prompterWs = null;
    prompterWsStatus = "disconnected";
    markCanvasDirty();
    clearQueueTrackers();
    stopBridgeStateHeartbeat();
    lastBridgeStateSignature = "";
    if (reconnectTimer) clearTimeout(reconnectTimer);
    reconnectTimer = setTimeout(() => {
      reconnectTimer = null;
      connectPrompterWs();
    }, RECONNECT_MS);
  };

  ws.onerror = () => {
    prompterWsStatus = "disconnected";
    markCanvasDirty();
    clearQueueTrackers();
    stopBridgeStateHeartbeat();
    lastBridgeStateSignature = "";
    try {
      ws.close();
    } catch {
      // no-op
    }
  };
}

function chainCallback(target, property, callback) {
  const original = target?.[property];
  if (typeof original === "function") {
    target[property] = function chainedCallback(...args) {
      const result = original.apply(this, args);
      callback.apply(this, args);
      return result;
    };
    return;
  }
  target[property] = callback;
}

app.registerExtension({
  name: "umbra.powerprompter.reader",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (!nodeType || !nodeData) return;
    const className = String(nodeData?.name || "");
    if (className === UNIFIED_NODE_CLASS) {
      chainCallback(nodeType.prototype, "onNodeCreated", function onUnifiedNodeCreated() {
        applySyncToNewNode(this, latestSyncState, applySyncToUnifiedPowerPrompterNode);
        markCanvasDirty();
      });

      chainCallback(nodeType.prototype, "onConfigure", function onUnifiedConfigure() {
        markCanvasDirty();
      });
      return;
    }

    if (className === NODE_CLASS) {
      chainCallback(nodeType.prototype, "onNodeCreated", function onNodeCreated() {
        applySyncToNewNode(this, latestSyncState, applySyncToNode);
        markCanvasDirty();
      });

      chainCallback(nodeType.prototype, "onConfigure", function onConfigure() {
        markCanvasDirty();
      });
      return;
    }

    if (KSAMPLER_NODE_CLASSES.has(className)) {
      chainCallback(nodeType.prototype, "onNodeCreated", function onKSamplerNodeCreated() {
        applySyncToNewNode(this, latestSyncState, applySyncToKSamplerNode);
        markCanvasDirty();
      });

      chainCallback(nodeType.prototype, "onConfigure", function onKSamplerConfigure() {
        markCanvasDirty();
      });
      return;
    }

    if (SEED_VALUE_NODE_CLASSES.has(className)) {
      chainCallback(nodeType.prototype, "onNodeCreated", function onSeedValueNodeCreated() {
        applySyncToNewNode(this, latestSyncState, applySyncToSeedValueNode);
        markCanvasDirty();
      });

      chainCallback(nodeType.prototype, "onConfigure", function onSeedValueConfigure() {
        markCanvasDirty();
      });
      return;
    }

    if (CHECKPOINT_NODE_CLASSES.has(className)) {
      chainCallback(nodeType.prototype, "onNodeCreated", function onCheckpointNodeCreated() {
        applySyncToNewNode(this, latestSyncState, applySyncToCheckpointNode);
        markCanvasDirty();
      });

      chainCallback(nodeType.prototype, "onConfigure", function onCheckpointNodeConfigure() {
        markCanvasDirty();
      });
      return;
    }

    if (className === LORA_NODE_CLASS) {
      chainCallback(nodeType.prototype, "onNodeCreated", function onLoraNodeCreated() {
        applySyncToNewNode(this, latestSyncState, applySyncToLoraNode);
        markCanvasDirty();
      });

      chainCallback(nodeType.prototype, "onConfigure", function onLoraConfigure() {
        markCanvasDirty();
      });
      return;
    }

    if (SAVE_NODE_TYPES.has(className)) {
      chainCallback(nodeType.prototype, "onNodeCreated", function onSaveNodeCreated() {
        applySyncToNewNode(this, latestSyncState, applySyncToSaveNode);
        markCanvasDirty();
      });

      chainCallback(nodeType.prototype, "onConfigure", function onSaveNodeConfigure() {
        markCanvasDirty();
      });
    }
  },
  setup() {
    ensureExecutionListeners();
    connectPrompterWs();
    window.addEventListener("beforeunload", () => {
      try {
        if (prompterWs) {
          prompterWs.close();
          prompterWs = null;
        }
      } catch {
        // no-op
      }
      clearQueueTrackers();
      if (reconnectTimer) {
        clearTimeout(reconnectTimer);
        reconnectTimer = null;
      }
    });
  },
});
