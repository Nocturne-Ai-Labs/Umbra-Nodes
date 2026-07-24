export function createPowerPrompterQueueManager(deps) {
  const {
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
    saveNodeTypes,
    maxQueueSets,
    idleFallbackDelayMs,
    queueSubmitBatchSize,
    queueSubmitBetweenPromptsMs,
    queueSubmitBetweenBatchesMs,
    promptCompletionTimeoutMs,
    previewFrameThrottleMs,
    previewMaxDataUrlLength,
    extractPromptIdFromExecution,
    extractNodeTypeFromExecution,
    payloadContainsImageOutput,
    extractProgressValue,
    normalizePreviewPayload,
    normalizePreviewPayloadFromEvent,
    blobToDataUrl,
    hasSaveNodes,
    validateQueueWorkflow,
    validateApiWorkflow,
  } = deps;

  let isQueueing = false;
  let queuePumpActive = false;
  let previewFrameInFlight = false;
  let lastPreviewFrameAt = 0;
  let currentExecutingPromptId = "";
  let currentQueueExecution = null;
  let queueCancelAllRequested = false;
  let queuePaused = false;
  let currentQueueDispatchDelayMs = 0;
  let lastQueueDispatchCompletedAt = 0;

  const queueCancelRequestIds = new Set();
  const queueStopAfterCurrentRequestIds = new Set();
  const activeQueueTrackers = new Map();
  const trackerProgressByKey = new Map();
  const trackerProgressSentByKey = new Map();
  const completedPromptWaitKeys = new Set();
  const idleFallbackTimers = new Map();
  const pendingQueueRequests = [];
  const submittedComfyPromptIds = new Set();

  function moveArrayEntry(items, fromIndex, toIndex) {
    if (!Array.isArray(items)) return [];
    if (fromIndex === toIndex) return [...items];
    if (fromIndex < 0 || fromIndex >= items.length) return [...items];
    const next = [...items];
    const moved = next.splice(fromIndex, 1)[0];
    if (moved === undefined) return [...items];
    next.splice(Math.max(0, Math.min(next.length, toIndex)), 0, moved);
    return next;
  }

  function createQueueTracker(requestId, prompts, options = {}) {
    const cleanPrompts = normalizePrompts(prompts, { dedupe: false });
    const queuePlan = {
      prompts: [...cleanPrompts],
      promptSetIds: Array.isArray(options.promptSetIds)
        ? options.promptSetIds.map((entry) => clampQueueSetId(entry, 1))
        : cleanPrompts.map(() => 1),
      promptStyleNames: Array.isArray(options.promptStyleNames)
        ? cleanPrompts.map((_, index) => String(options.promptStyleNames[index] || "").trim())
        : cleanPrompts.map(() => ""),
      generationByPrompt: Array.isArray(options.generationByPrompt)
        ? options.generationByPrompt.map((entry) => normalizeGenerationState(entry))
        : cleanPrompts.map(() => normalizeGenerationState(options.generation)),
      promptOutputSubfolders: Array.isArray(options.promptOutputSubfolders)
        ? cleanPrompts.map((_, index) => String(options.promptOutputSubfolders[index] || "").trim())
        : cleanPrompts.map(() => ""),
      promptSeedGroupIds: Array.isArray(options.promptSeedGroupIds)
        ? cleanPrompts.map((_, index) => String(options.promptSeedGroupIds[index] || `${index}`).trim())
        : cleanPrompts.map((_, index) => `${index}`),
    };
    const tracker = {
      requestId,
      prompts: queuePlan.prompts,
      queuePlan,
      total: cleanPrompts.length,
      completedCount: 0,
      completedBySaveCount: 0,
      completedBySaveFlags: new Array(cleanPrompts.length).fill(false),
      completedFlags: new Array(cleanPrompts.length).fill(false),
      promptIds: new Array(cleanPrompts.length).fill(""),
      promptSeeds: new Array(cleanPrompts.length).fill(0),
      promptIdToIndex: new Map(),
      queueTargetType: String(options.queueTargetType || "").trim() || "live_workflow",
      saveNodeIds: new Set(Array.isArray(options.saveNodeIds) ? options.saveNodeIds.map((entry) => String(entry || "").trim()).filter(Boolean) : []),
      sourceFile: String(options.sourceFile || "").trim(),
      createdAt: Date.now(),
    };
    activeQueueTrackers.set(requestId, tracker);
    return tracker;
  }

  function extractExecutionNodeId(detail) {
    if (!detail || typeof detail !== "object") return "";
    const rawNode = detail.node ?? detail.node_id ?? detail.nodeId ?? detail.id ?? "";
    if (rawNode && typeof rawNode === "object") {
      const nested =
        rawNode.id ??
        rawNode.node ??
        rawNode.node_id ??
        rawNode.nodeId ??
        "";
      return String(nested || "").trim();
    }
    return String(rawNode || "").trim();
  }

  function clearQueueCancelState() {
    queueCancelAllRequested = false;
    queueCancelRequestIds.clear();
    queueStopAfterCurrentRequestIds.clear();
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

  function clearTrackerProgress(requestId) {
    const prefix = `${String(requestId || "")}:`;
    for (const key of Array.from(trackerProgressByKey.keys())) {
      if (key.startsWith(prefix)) trackerProgressByKey.delete(key);
    }
    for (const key of Array.from(trackerProgressSentByKey.keys())) {
      if (key.startsWith(prefix)) trackerProgressSentByKey.delete(key);
    }
  }

  function clearQueueTrackers() {
    clearQueueCancelState();
    pendingQueueRequests.length = 0;
    queuePumpActive = false;
    isQueueing = false;
    queuePaused = false;
    for (const timer of idleFallbackTimers.values()) {
      clearTimeout(timer);
    }
    idleFallbackTimers.clear();
    activeQueueTrackers.clear();
    trackerProgressByKey.clear();
    trackerProgressSentByKey.clear();
    completedPromptWaitKeys.clear();
    submittedComfyPromptIds.clear();
    previewFrameInFlight = false;
    lastPreviewFrameAt = 0;
    currentExecutingPromptId = "";
    currentQueueExecution = null;
  }

  function normalizeQueueDispatchDelayMs(rawValue) {
    const numeric = Number(rawValue);
    if (!Number.isFinite(numeric)) return currentQueueDispatchDelayMs;
    return Math.max(0, Math.floor(numeric));
  }

  async function waitForQueueDispatchDelay(requestId, startedAtInput = Date.now()) {
    const startedAt = Number.isFinite(startedAtInput) ? Number(startedAtInput) : Date.now();
    while (true) {
      if (isQueueCancelRequestedFor(requestId)) {
        return false;
      }
      if (shouldStopAfterCurrentFor(requestId)) {
        return false;
      }
      const canContinue = await waitWhileQueuePaused(requestId);
      if (!canContinue) {
        return false;
      }
      const targetDelay = Math.max(0, Math.floor(Number(currentQueueDispatchDelayMs) || 0));
      const elapsed = Date.now() - startedAt;
      if (elapsed >= targetDelay) {
        return true;
      }
      await sleep(Math.min(150, Math.max(20, targetDelay - elapsed)));
    }
  }

  function findPendingQueueRequestIndex(requestId) {
    const normalizedRequestId = String(requestId || "").trim();
    if (!normalizedRequestId) return -1;
    return pendingQueueRequests.findIndex((entry) => String(entry?.requestId || "").trim() === normalizedRequestId);
  }

  function isQueueRequestAlreadyTracked(requestId) {
    const normalizedRequestId = String(requestId || "").trim();
    if (!normalizedRequestId) return false;
    if (activeQueueTrackers.has(normalizedRequestId)) return true;
    if (currentQueueExecution && String(currentQueueExecution.requestId || "").trim() === normalizedRequestId) return true;
    return findPendingQueueRequestIndex(normalizedRequestId) >= 0;
  }

  function emitQueuePauseState(source = "bridge") {
    sendWs({
      type: "queue_pause_state",
      paused: queuePaused === true,
      source,
      queueing: isQueueing === true || pendingQueueRequests.length > 0,
      activeRequestIds: Array.from(activeQueueTrackers.keys()),
      pendingRequestIds: pendingQueueRequests
        .map((entry) => String(entry?.requestId || "").trim())
        .filter((entry) => entry.length > 0),
      pendingCount: pendingQueueRequests.length,
      dispatchDelayMs: Math.max(0, Math.floor(Number(currentQueueDispatchDelayMs) || 0)),
      updatedAt: Date.now(),
    });
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

  function shouldStopAfterCurrentFor(requestId) {
    const key = String(requestId || "").trim();
    if (!key) return false;
    return queueStopAfterCurrentRequestIds.has(key);
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
    submittedComfyPromptIds.add(promptId);
  }

  function rebuildTrackerPromptIdIndex(tracker) {
    if (!tracker) return;
    tracker.promptIdToIndex = new Map();
    const promptIds = Array.isArray(tracker.promptIds) ? tracker.promptIds : [];
    for (let idx = 0; idx < promptIds.length; idx += 1) {
      const promptId = String(promptIds[idx] || "").trim();
      if (!promptId) continue;
      tracker.promptIdToIndex.set(promptId, idx);
    }
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

  function promptCompletionWaitKey(requestId, promptIndex) {
    return `${String(requestId || "").trim()}:${Math.max(0, Math.floor(Number(promptIndex) || 0))}`;
  }

  function rememberPromptCompletionForWait(requestId, promptIndex) {
    const key = promptCompletionWaitKey(requestId, promptIndex);
    if (!key || key === ":0") return;
    completedPromptWaitKeys.add(key);
    setTimeout(() => completedPromptWaitKeys.delete(key), 60000);
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

  function resolveTrackerForPromptId(promptId, options = {}) {
    if (activeQueueTrackers.size === 0) return null;
    const allowActiveFallback = options?.allowActiveFallback !== false;
    const allowAnyFallback = options?.allowAnyFallback !== false;
    if (promptId) {
      for (const tracker of activeQueueTrackers.values()) {
        const promptIndex = resolvePromptIndexFromTracker(tracker, promptId);
        if (promptIndex >= 0 && tracker.promptIdToIndex.get(promptId) === promptIndex) {
          return { tracker, promptIndex };
        }
      }
    }
    if (allowActiveFallback && currentQueueExecution) {
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
    if (!allowAnyFallback) return null;
    for (const tracker of activeQueueTrackers.values()) {
      const promptIndex = resolvePromptIndexFromTracker(tracker, "");
      if (promptIndex >= 0) return { tracker, promptIndex };
    }
    return null;
  }

  function hasTrackedPromptIds() {
    for (const tracker of activeQueueTrackers.values()) {
      if (!Array.isArray(tracker.promptIds)) continue;
      if (tracker.promptIds.some((entry) => String(entry || "").trim())) return true;
    }
    return false;
  }

  function resolveTrackerForLiveComfyEvent(promptId) {
    const normalizedPromptId = String(promptId || "").trim();
    if (normalizedPromptId) {
      return resolveTrackerForPromptId(normalizedPromptId, {
        allowActiveFallback: false,
        allowAnyFallback: false,
      });
    }
    if (hasTrackedPromptIds()) return null;
    return resolveTrackerForPromptId("", {
      allowActiveFallback: true,
      allowAnyFallback: false,
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
      prompt: String(tracker.prompts[promptIndex] || ""),
      promptId,
      step: progress.step,
      maxStep: progress.maxStep,
      completed: tracker.completedCount,
      total: tracker.total,
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
    if (!tracker) return;
    if (promptIndex < 0 || promptIndex >= tracker.total) return;
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

  function extractSavedOutputEntries(payload, depth = 0) {
    if (!payload || depth > 4) return [];
    if (Array.isArray(payload)) {
      return payload.flatMap((entry) => extractSavedOutputEntries(entry, depth + 1));
    }
    if (typeof payload !== "object") return [];
    const directImages = Array.isArray(payload.images) ? payload.images : [];
    if (directImages.length > 0) {
      return directImages
        .map((entry) => {
          if (!entry || typeof entry !== "object") return null;
          const fullpath = String(entry.fullpath || entry.fullPath || entry.path || "").trim();
          const filename = String(entry.filename || entry.name || "").trim();
          const subfolder = String(entry.subfolder || "").trim();
          const type = String(entry.type || "output").trim() || "output";
          if (!fullpath && !filename) return null;
          return {
            fullpath,
            filename,
            subfolder,
            type,
          };
        })
        .filter(Boolean);
    }
    const nested = [];
    for (const value of Object.values(payload)) {
      nested.push(...extractSavedOutputEntries(value, depth + 1));
    }
    return nested;
  }

  function emitSavedOutputs(tracker, promptIndex, outputs) {
    if (!tracker || promptIndex < 0 || promptIndex >= tracker.total) return;
    const normalizedOutputs = Array.isArray(outputs) ? outputs.filter(Boolean) : [];
    if (normalizedOutputs.length <= 0) return;
    const promptSetId = clampQueueSetId(tracker.queuePlan?.promptSetIds?.[promptIndex], 1);
    const promptSetLabel = `set ${promptSetId}`;
    const promptOutputSubfolder = String(tracker.queuePlan?.promptOutputSubfolders?.[promptIndex] || "").trim();
    const promptStyleName = String(tracker.queuePlan?.promptStyleNames?.[promptIndex] || "").trim();
    const tags = [promptSetLabel];
    const enrichedOutputs = normalizedOutputs.map((output) => {
      if (!output || typeof output !== "object") return output;
      const outputTags = Array.isArray(output.tags) ? output.tags : [];
      return {
        ...output,
        promptSetId,
        promptSetLabel,
        promptOutputSubfolder,
        promptStyleName,
        tags: Array.from(new Set([...outputTags, ...tags].map((entry) => String(entry || "").trim()).filter(Boolean))),
      };
    });
    sendWs({
      type: "queue_saved_outputs",
      requestId: tracker.requestId,
      promptIndex,
      promptSetId,
      promptSetLabel,
      promptOutputSubfolder,
      promptStyleName,
      tags,
      promptId: String(tracker.promptIds[promptIndex] || ""),
      sourceFile: String(tracker.sourceFile || ""),
      outputs: enrichedOutputs,
      updatedAt: Date.now(),
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

  function markPromptCompleted(tracker, promptIndex, source) {
    if (!tracker) return false;
    if (promptIndex < 0 || promptIndex >= tracker.total) return false;
    if (tracker.completedFlags[promptIndex]) return false;
    const existingProgress = getTrackerProgress(tracker, promptIndex);
    if (existingProgress.maxStep > 0) {
      setTrackerProgress(tracker, promptIndex, existingProgress.maxStep, existingProgress.maxStep);
    }
    tracker.completedFlags[promptIndex] = true;
    rememberPromptCompletionForWait(tracker.requestId, promptIndex);
    tracker.completedCount += 1;
    if (source === "save_output") {
      tracker.completedBySaveCount += 1;
      if (Array.isArray(tracker.completedBySaveFlags)) tracker.completedBySaveFlags[promptIndex] = true;
    }
    emitQueueProgress(tracker, promptIndex, source);
    if (tracker.completedCount >= tracker.total) {
      finalizeTracker(tracker, source === "save_output" ? "save_output_complete" : "completed");
    }
    return true;
  }

  function markActivePromptInterrupted() {
    clearAllIdleFallbackTimers();
    const activeRequestId = String(currentQueueExecution?.requestId || "").trim();
    const activePromptIndex = Number.isFinite(currentQueueExecution?.promptIndex)
      ? Math.max(0, Math.floor(currentQueueExecution.promptIndex))
      : -1;

    if (activeRequestId) {
      const tracker = activeQueueTrackers.get(activeRequestId);
      if (tracker && activePromptIndex >= 0 && activePromptIndex < tracker.total) {
        const interrupted = markPromptCompleted(tracker, activePromptIndex, "interrupted");
        if (interrupted) {
          currentQueueExecution = null;
          currentExecutingPromptId = "";
        }
        return interrupted;
      }
    }

    const resolved = resolveTrackerForPromptId(currentExecutingPromptId || "");
    if (!resolved) return false;
    const interrupted = markPromptCompleted(resolved.tracker, resolved.promptIndex, "interrupted");
    if (interrupted) {
      currentQueueExecution = null;
      currentExecutingPromptId = "";
    }
    return interrupted;
  }

  function applyIdleFallbackToTracker(tracker, promptIndexRaw = -1) {
    if (!tracker) return;
    const requestedIndex = Number(promptIndexRaw);
    if (Number.isFinite(requestedIndex)) {
      const promptIndex = Math.max(0, Math.floor(requestedIndex));
      if (promptIndex >= 0 && promptIndex < tracker.total && !tracker.completedFlags[promptIndex]) {
        markPromptCompleted(tracker, promptIndex, "idle_fallback");
      }
      return;
    }
    for (let idx = 0; idx < tracker.total; idx += 1) {
      if (tracker.completedFlags[idx]) continue;
      markPromptCompleted(tracker, idx, "idle_fallback");
      return;
    }
  }

  function scheduleIdleFallback(tracker, promptIndexRaw = -1) {
    if (!tracker) return;
    if (tracker.completedCount >= tracker.total) return;
    const requestedIndex = Number(promptIndexRaw);
    const expectedPromptIndex = Number.isFinite(requestedIndex)
      ? Math.max(0, Math.floor(requestedIndex))
      : -1;
    if (expectedPromptIndex >= 0 && Array.isArray(tracker.completedBySaveFlags) && tracker.completedBySaveFlags[expectedPromptIndex]) return;

    const requestId = String(tracker.requestId || "");
    const expectedCompleted = tracker.completedCount;
    const expectedSaveCompleted = tracker.completedBySaveCount;
    clearIdleFallbackTimer(requestId);

    const timer = setTimeout(() => {
      idleFallbackTimers.delete(requestId);
      const latest = activeQueueTrackers.get(requestId);
      if (!latest) return;
      if (latest.completedCount >= latest.total) return;
      if (expectedPromptIndex >= 0 && Array.isArray(latest.completedBySaveFlags) && latest.completedBySaveFlags[expectedPromptIndex]) return;
      if (
        latest.completedCount !== expectedCompleted ||
        latest.completedBySaveCount !== expectedSaveCompleted
      ) {
        return;
      }
      if (
        expectedPromptIndex >= 0 &&
        expectedPromptIndex < latest.total &&
        latest.completedFlags[expectedPromptIndex]
      ) {
        return;
      }
      applyIdleFallbackToTracker(latest, expectedPromptIndex);
    }, idleFallbackDelayMs);

    idleFallbackTimers.set(requestId, timer);
  }

  function waitForPromptCompletion(requestId, promptIndex, timeoutMs = promptCompletionTimeoutMs) {
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
          resolve(completedPromptWaitKeys.has(promptCompletionWaitKey(id, index)) ? "completed" : "tracker_missing");
          return;
        }
        if (tracker.completedFlags[index]) {
          if (currentQueueExecution && String(currentQueueExecution.requestId || "") === id && Number(currentQueueExecution.promptIndex) === index) {
            currentQueueExecution = null;
          }
          resolve("completed");
          return;
        }
        if (Date.now() - startedAt >= Math.max(5000, Math.floor(Number(timeoutMs) || promptCompletionTimeoutMs))) {
          resolve("timeout");
          return;
        }
        setTimeout(poll, 120);
      };
      poll();
    });
  }

  function readComfyQueuePromptId(entry) {
    if (Array.isArray(entry)) {
      return String(entry[1] ?? entry?.[3]?.prompt_id ?? entry?.[4]?.prompt_id ?? "").trim();
    }
    if (entry && typeof entry === "object") {
      return String(entry.prompt_id ?? entry.promptId ?? entry.id ?? "").trim();
    }
    return "";
  }

  function containsPowerPrompterReader(value, depth = 0) {
    if (!value || depth > 8) return false;
    if (typeof value !== "object") return false;
    const classType = String(value.class_type || value.type || value.node_type || "").trim();
    if (classType === "UmbraPowerPrompterReader" || classType === "UmbraPowerPrompter") return true;
    if (Array.isArray(value)) {
      return value.some((entry) => containsPowerPrompterReader(entry, depth + 1));
    }
    return Object.values(value).some((entry) => containsPowerPrompterReader(entry, depth + 1));
  }

  function comfyQueueEntryContainsPowerPrompter(entry) {
    if (!entry) return false;
    if (containsPowerPrompterReader(entry)) return true;
    if (Array.isArray(entry)) {
      for (const candidate of [entry[2], entry[3], entry[4]]) {
        if (containsPowerPrompterReader(candidate)) return true;
      }
    }
    return false;
  }

  async function fetchComfyQueueSnapshot() {
    const emptySnapshot = { promptIds: new Set(), powerPrompterPromptIds: new Set() };
    if (typeof fetch !== "function") return emptySnapshot;
    const response = await fetch("/queue", { cache: "no-store" });
    if (!response.ok) throw new Error(`Unable to inspect ComfyUI queue (${response.status}).`);
    const payload = await response.json().catch(() => ({}));
    const ids = new Set();
    const powerPrompterIds = new Set();
    for (const section of [payload?.queue_running, payload?.queue_pending]) {
      if (!Array.isArray(section)) continue;
      for (const entry of section) {
        const promptId = readComfyQueuePromptId(entry);
        if (promptId) ids.add(promptId);
        if (promptId && comfyQueueEntryContainsPowerPrompter(entry)) {
          powerPrompterIds.add(promptId);
        }
      }
    }
    return { promptIds: ids, powerPrompterPromptIds: powerPrompterIds };
  }

  async function waitForPowerPrompterComfyQueueDrain(timeoutMs = Math.min(promptCompletionTimeoutMs, 45000)) {
    const startedAt = Date.now();
    while (true) {
      const snapshot = await fetchComfyQueueSnapshot();
      for (const promptId of Array.from(submittedComfyPromptIds)) {
        if (!snapshot.promptIds.has(promptId)) submittedComfyPromptIds.delete(promptId);
      }
      if (submittedComfyPromptIds.size <= 0 && snapshot.powerPrompterPromptIds.size <= 0) return;
      if (Date.now() - startedAt >= Math.max(5000, Math.floor(Number(timeoutMs) || 45000))) {
        throw new Error("Timed out waiting for the previous Power Prompter prompt to leave the ComfyUI queue.");
      }
      await sleep(180);
    }
  }

  async function waitForComfyPromptIdToDrain(promptIdRaw, timeoutMs = Math.min(promptCompletionTimeoutMs, 45000)) {
    const promptId = String(promptIdRaw || "").trim();
    if (!promptId) return;
    const startedAt = Date.now();
    while (true) {
      const snapshot = await fetchComfyQueueSnapshot();
      if (!snapshot.promptIds.has(promptId)) {
        submittedComfyPromptIds.delete(promptId);
        return;
      }
      if (Date.now() - startedAt >= Math.max(5000, Math.floor(Number(timeoutMs) || 45000))) {
        throw new Error(`Timed out waiting for ComfyUI prompt ${promptId} to leave the queue.`);
      }
      await sleep(180);
    }
  }

  function reorderTrackerPrompts(tracker, promptOrder, lockedIndex = -1) {
    if (!tracker || !Array.isArray(promptOrder) || promptOrder.length <= 0) return false;
    const total = Math.max(0, Math.floor(Number(tracker.total) || 0));
    if (total <= 0) return false;
    const uniqueOrder = Array.from(new Set(
      promptOrder
        .map((entry) => Number(entry))
        .filter((entry) => Number.isFinite(entry))
        .map((entry) => Math.max(0, Math.floor(entry)))
    ));
    const allIndices = Array.from({ length: total }, (_, index) => index);
    const immutablePrefix = lockedIndex >= 0 ? allIndices.filter((index) => index < lockedIndex) : [];
    const lockedSegment = lockedIndex >= 0 ? [lockedIndex] : [];
    const movableIndices = allIndices.filter((index) => index > lockedIndex);
    const desiredMovable = uniqueOrder.filter((index) => movableIndices.includes(index));
    const finalOrder = [
      ...immutablePrefix,
      ...lockedSegment,
      ...desiredMovable,
      ...movableIndices.filter((index) => !desiredMovable.includes(index)),
    ];
    if (finalOrder.length !== total) return false;
    const unchanged = finalOrder.every((sourceIndex, targetIndex) => sourceIndex === targetIndex);
    if (unchanged) return false;
    const reorderValues = (values, fallbackFactory) =>
      finalOrder.map((sourceIndex, targetIndex) => {
        if (Array.isArray(values) && sourceIndex < values.length) return values[sourceIndex];
        return fallbackFactory(targetIndex);
      });
    tracker.prompts = reorderValues(tracker.prompts, () => "");
    tracker.queuePlan = tracker.queuePlan || {};
    tracker.queuePlan.prompts = [...tracker.prompts];
    tracker.queuePlan.promptSetIds = reorderValues(tracker.queuePlan.promptSetIds, () => 1);
    tracker.queuePlan.promptStyleNames = reorderValues(tracker.queuePlan.promptStyleNames, () => "");
    tracker.queuePlan.generationByPrompt = reorderValues(tracker.queuePlan.generationByPrompt, () => normalizeGenerationState({}));
    tracker.queuePlan.promptOutputSubfolders = reorderValues(tracker.queuePlan.promptOutputSubfolders, () => "");
    tracker.queuePlan.promptSeedGroupIds = reorderValues(tracker.queuePlan.promptSeedGroupIds, (index) => `${index}`);
    tracker.completedFlags = reorderValues(tracker.completedFlags, () => false);
    tracker.completedBySaveFlags = reorderValues(tracker.completedBySaveFlags, () => false);
    tracker.promptIds = reorderValues(tracker.promptIds, () => "");
    tracker.promptSeeds = reorderValues(tracker.promptSeeds, () => 0);
    rebuildTrackerPromptIdIndex(tracker);
    return true;
  }

  function reorderPendingRequestPrompts(pendingRequest, promptOrder) {
    if (!pendingRequest || !Array.isArray(promptOrder) || promptOrder.length <= 0) return false;
    const prompts = normalizePrompts(pendingRequest?.prompts, { dedupe: false });
    const total = prompts.length;
    if (total <= 0) return false;
    const uniqueOrder = Array.from(new Set(
      promptOrder
        .map((entry) => Number(entry))
        .filter((entry) => Number.isFinite(entry))
        .map((entry) => Math.max(0, Math.floor(entry)))
    ));
    const fullOrder = [
      ...uniqueOrder.filter((index) => index < total),
      ...Array.from({ length: total }, (_, index) => index).filter((index) => !uniqueOrder.includes(index)),
    ];
    if (fullOrder.length !== total || fullOrder.every((value, index) => value === index)) return false;
    const state = pendingRequest.state && typeof pendingRequest.state === "object" ? pendingRequest.state : {};
    const promptSetIds = normalizePromptSetIds(state.promptSetIds, total, state.activeQueueSet || state.activeSetId || 1);
    const promptStyleNames = Array.isArray(state.promptStyleNames)
      ? prompts.map((_, index) => String(state.promptStyleNames[index] || "").trim())
      : prompts.map(() => "");
    const generationByPrompt = Array.isArray(state.generationByPrompt)
      ? state.generationByPrompt.map((entry) => normalizeGenerationState(entry))
      : prompts.map(() => normalizeGenerationState(state.generation));
    const promptOutputSubfolders = Array.isArray(state.promptOutputSubfolders)
      ? prompts.map((_, index) => String(state.promptOutputSubfolders[index] || "").trim())
      : prompts.map(() => "");
    const promptSeedGroupIds = Array.isArray(state.promptSeedGroupIds)
      ? prompts.map((_, index) => String(state.promptSeedGroupIds[index] || `${index}`).trim())
      : prompts.map((_, index) => `${index}`);
    pendingRequest.prompts = fullOrder.map((index) => prompts[index]);
    if (pendingRequest.state && typeof pendingRequest.state === "object") {
      pendingRequest.state.prompts = [...pendingRequest.prompts];
      pendingRequest.state.promptSetIds = fullOrder.map((index) => promptSetIds[index] ?? 1);
      pendingRequest.state.promptStyleNames = fullOrder.map((index) => promptStyleNames[index] ?? "");
      pendingRequest.state.generationByPrompt = fullOrder.map((index) => generationByPrompt[index] ?? normalizeGenerationState(state.generation));
      pendingRequest.state.promptOutputSubfolders = fullOrder.map((index) => promptOutputSubfolders[index] ?? "");
      pendingRequest.state.promptSeedGroupIds = fullOrder.map((index) => promptSeedGroupIds[index] ?? `${index}`);
      pendingRequest.state.activePrompt = String(pendingRequest.prompts[0] || "");
      pendingRequest.state.joinedPrompt = normalizePrompts(pendingRequest.prompts, { dedupe: false }).join(", ");
    }
    return true;
  }

  function removeTrackerPrompts(tracker, promptIndices, lockedIndex = -1) {
    if (!tracker || !Array.isArray(promptIndices) || promptIndices.length <= 0) return { applied: false, removedAll: false };
    const total = Math.max(0, Math.floor(Number(tracker.total) || 0));
    if (total <= 0) return { applied: false, removedAll: false };
    const blockedIndex = lockedIndex >= 0 ? Math.max(0, Math.floor(Number(lockedIndex) || 0)) : -1;
    const removalSet = new Set(
      promptIndices
        .map((entry) => Number(entry))
        .filter((entry) => Number.isFinite(entry))
        .map((entry) => Math.max(0, Math.floor(entry)))
        .filter((entry) => entry < total && (blockedIndex < 0 || entry > blockedIndex))
    );
    if (removalSet.size <= 0) return { applied: false, removedAll: false };
    const keepIndices = Array.from({ length: total }, (_, index) => index).filter((index) => !removalSet.has(index));
    if (keepIndices.length <= 0) {
      return { applied: true, removedAll: true };
    }
    const reorderValues = (values, fallbackFactory) =>
      keepIndices.map((sourceIndex, targetIndex) => {
        if (Array.isArray(values) && sourceIndex < values.length) return values[sourceIndex];
        return fallbackFactory(targetIndex);
      });
    tracker.prompts = reorderValues(tracker.prompts, () => "");
    tracker.queuePlan = tracker.queuePlan || {};
    tracker.queuePlan.prompts = [...tracker.prompts];
    tracker.queuePlan.promptSetIds = reorderValues(tracker.queuePlan.promptSetIds, () => 1);
    tracker.queuePlan.promptStyleNames = reorderValues(tracker.queuePlan.promptStyleNames, () => "");
    tracker.queuePlan.generationByPrompt = reorderValues(tracker.queuePlan.generationByPrompt, () => normalizeGenerationState({}));
    tracker.queuePlan.promptOutputSubfolders = reorderValues(tracker.queuePlan.promptOutputSubfolders, () => "");
    tracker.queuePlan.promptSeedGroupIds = reorderValues(tracker.queuePlan.promptSeedGroupIds, (index) => `${index}`);
    tracker.completedFlags = reorderValues(tracker.completedFlags, () => false);
    tracker.completedBySaveFlags = reorderValues(tracker.completedBySaveFlags, () => false);
    tracker.promptIds = reorderValues(tracker.promptIds, () => "");
    tracker.promptSeeds = reorderValues(tracker.promptSeeds, () => 0);
    tracker.total = keepIndices.length;
    tracker.completedCount = tracker.completedFlags.filter((entry) => entry === true).length;
    tracker.completedBySaveCount = Math.min(tracker.completedBySaveCount, tracker.completedCount);
    rebuildTrackerPromptIdIndex(tracker);
    return { applied: true, removedAll: false };
  }

  function removePendingRequestPrompts(pendingRequest, promptIndices) {
    if (!pendingRequest || !Array.isArray(promptIndices) || promptIndices.length <= 0) return { applied: false, removedAll: false };
    const prompts = normalizePrompts(pendingRequest?.prompts, { dedupe: false });
    const total = prompts.length;
    if (total <= 0) return { applied: false, removedAll: false };
    const removalSet = new Set(
      promptIndices
        .map((entry) => Number(entry))
        .filter((entry) => Number.isFinite(entry))
        .map((entry) => Math.max(0, Math.floor(entry)))
        .filter((entry) => entry < total)
    );
    if (removalSet.size <= 0) return { applied: false, removedAll: false };
    const keepIndices = Array.from({ length: total }, (_, index) => index).filter((index) => !removalSet.has(index));
    if (keepIndices.length <= 0) {
      return { applied: true, removedAll: true };
    }
    const state = pendingRequest.state && typeof pendingRequest.state === "object" ? pendingRequest.state : {};
    const promptSetIds = normalizePromptSetIds(state.promptSetIds, total, state.activeQueueSet || state.activeSetId || 1);
    const promptStyleNames = Array.isArray(state.promptStyleNames)
      ? prompts.map((_, index) => String(state.promptStyleNames[index] || "").trim())
      : prompts.map(() => "");
    const generationByPrompt = Array.isArray(state.generationByPrompt)
      ? state.generationByPrompt.map((entry) => normalizeGenerationState(entry))
      : prompts.map(() => normalizeGenerationState(state.generation));
    const promptOutputSubfolders = Array.isArray(state.promptOutputSubfolders)
      ? prompts.map((_, index) => String(state.promptOutputSubfolders[index] || "").trim())
      : prompts.map(() => "");
    const promptSeedGroupIds = Array.isArray(state.promptSeedGroupIds)
      ? prompts.map((_, index) => String(state.promptSeedGroupIds[index] || `${index}`).trim())
      : prompts.map((_, index) => `${index}`);
    pendingRequest.prompts = keepIndices.map((index) => prompts[index]);
    if (pendingRequest.state && typeof pendingRequest.state === "object") {
      pendingRequest.state.prompts = [...pendingRequest.prompts];
      pendingRequest.state.promptSetIds = keepIndices.map((index) => promptSetIds[index] ?? 1);
      pendingRequest.state.promptStyleNames = keepIndices.map((index) => promptStyleNames[index] ?? "");
      pendingRequest.state.generationByPrompt = keepIndices.map((index) => generationByPrompt[index] ?? normalizeGenerationState(state.generation));
      pendingRequest.state.promptOutputSubfolders = keepIndices.map((index) => promptOutputSubfolders[index] ?? "");
      pendingRequest.state.promptSeedGroupIds = keepIndices.map((index) => promptSeedGroupIds[index] ?? `${index}`);
      pendingRequest.state.activePrompt = String(pendingRequest.prompts[0] || "");
      pendingRequest.state.joinedPrompt = normalizePrompts(pendingRequest.prompts, { dedupe: false }).join(", ");
    }
    return { applied: true, removedAll: false };
  }

  function handleQueueReorder(message) {
    const requestOrder = Array.isArray(message?.requestOrder)
      ? message.requestOrder.map((entry) => String(entry || "").trim()).filter((entry) => entry.length > 0)
      : [];
    const promptOrders = Array.isArray(message?.promptOrders)
      ? message.promptOrders
        .map((entry) => ({
          requestId: String(entry?.requestId || "").trim(),
          promptOrder: Array.isArray(entry?.promptOrder) ? entry.promptOrder : [],
        }))
        .filter((entry) => entry.requestId.length > 0 && entry.promptOrder.length > 0)
      : [];

    let applied = false;

    if (requestOrder.length > 0 && pendingQueueRequests.length > 1) {
      const sortIndexByRequestId = new Map(requestOrder.map((requestId, index) => [requestId, index]));
      const reordered = [...pendingQueueRequests].sort((left, right) => {
        const leftId = String(left?.requestId || "").trim();
        const rightId = String(right?.requestId || "").trim();
        const leftIndex = sortIndexByRequestId.has(leftId) ? sortIndexByRequestId.get(leftId) : Number.MAX_SAFE_INTEGER;
        const rightIndex = sortIndexByRequestId.has(rightId) ? sortIndexByRequestId.get(rightId) : Number.MAX_SAFE_INTEGER;
        if (leftIndex !== rightIndex) return leftIndex - rightIndex;
        return 0;
      });
      for (let idx = 0; idx < reordered.length; idx += 1) {
        pendingQueueRequests[idx] = reordered[idx];
      }
      applied = true;
    }

    for (const entry of promptOrders) {
      const activeRequestId = String(currentQueueExecution?.requestId || "").trim();
      const activePromptIndex = Number.isFinite(currentQueueExecution?.promptIndex)
        ? Math.max(0, Math.floor(currentQueueExecution.promptIndex))
        : -1;
      const tracker = activeQueueTrackers.get(entry.requestId);
      if (tracker) {
        applied = reorderTrackerPrompts(tracker, entry.promptOrder, activeRequestId === entry.requestId ? activePromptIndex : -1) || applied;
        continue;
      }
      const pending = pendingQueueRequests.find((candidate) => String(candidate?.requestId || "").trim() === entry.requestId);
      if (pending) {
        applied = reorderPendingRequestPrompts(pending, entry.promptOrder) || applied;
      }
    }

    sendWs({
      type: "queue_reorder_result",
      requestId: String(message?.requestId || "").trim(),
      success: true,
      applied,
    });
    emitQueuePauseState("queue_reorder");
  }

  function handleQueuePromptRemove(message) {
    const removals = Array.isArray(message?.promptRemovals)
      ? message.promptRemovals
        .map((entry) => ({
          requestId: String(entry?.requestId || "").trim(),
          promptIndices: Array.isArray(entry?.promptIndices) ? entry.promptIndices : [],
        }))
        .filter((entry) => entry.requestId.length > 0 && entry.promptIndices.length > 0)
      : [];

    let applied = false;
    const removedRequestIds = [];
    for (const entry of removals) {
      const activeRequestId = String(currentQueueExecution?.requestId || "").trim();
      const activePromptIndex = Number.isFinite(currentQueueExecution?.promptIndex)
        ? Math.max(0, Math.floor(currentQueueExecution.promptIndex))
        : -1;
      const tracker = activeQueueTrackers.get(entry.requestId);
      if (tracker) {
        const result = removeTrackerPrompts(tracker, entry.promptIndices, activeRequestId === entry.requestId ? activePromptIndex : -1);
        applied = result.applied || applied;
        if (result.removedAll) {
          cancelTrackerByRequestId(entry.requestId, "queue_canceled");
          removedRequestIds.push(entry.requestId);
        }
        continue;
      }
      const pendingIndex = findPendingQueueRequestIndex(entry.requestId);
      if (pendingIndex >= 0) {
        const pending = pendingQueueRequests[pendingIndex];
        const result = removePendingRequestPrompts(pending, entry.promptIndices);
        applied = result.applied || applied;
        if (result.removedAll) {
          pendingQueueRequests.splice(pendingIndex, 1);
          removedRequestIds.push(entry.requestId);
        }
      }
    }

    sendWs({
      type: "queue_prompt_remove_result",
      requestId: String(message?.requestId || "").trim(),
      success: true,
      applied,
      removedRequestIds,
    });
    emitQueuePauseState("queue_prompt_remove");
  }

  function handleQueueDispatchDelayUpdate(message) {
    currentQueueDispatchDelayMs = normalizeQueueDispatchDelayMs(message?.dispatchDelayMs);
    sendWs({
      type: "queue_delay_result",
      requestId: String(message?.requestId || "").trim(),
      success: true,
      dispatchDelayMs: currentQueueDispatchDelayMs,
    });
    emitQueuePauseState("queue_delay");
  }

  async function processQueueRequest(message) {
    const requestId = String(message?.requestId || "");
    if (!requestId) return;

    const queueTargetType = String(message?.queueTargetType || "").trim();
    const requestUsesApiWorkflow = queueTargetType === "api_workflow";
    const validation = requestUsesApiWorkflow
      ? (typeof validateApiWorkflow === "function" ? validateApiWorkflow(message?.apiWorkflow) : { ok: true, missing: [] })
      : (typeof validateQueueWorkflow === "function" ? validateQueueWorkflow() : { ok: true, missing: [] });

    if (validation) {
      if (validation && validation.ok === false) {
        const details = Array.isArray(validation.missing) ? validation.missing.filter(Boolean) : [];
        const suffix = details.length > 0 ? ` Missing: ${details.join(", ")}` : "";
        sendWs({
          type: "queue_result",
          requestId,
          success: false,
          error: `Workflow is missing required Umbra nodes.${suffix}`,
        });
        return;
      }
    }

    const prompts = normalizePrompts(message?.prompts, { dedupe: false });
    if (prompts.length === 0) {
      sendWs({
        type: "queue_result",
        requestId,
        success: false,
        error: "No prompts received for queue request.",
      });
      return;
    }

    clearQueueCancelState();
    const updatedState = {
      ...latestSyncState,
      prompts: normalizePrompts(message?.state?.prompts || latestSyncState.prompts || prompts, { dedupe: false }),
      activePrompt: cleanPromptText(message?.state?.activePrompt || prompts[0] || latestSyncState.activePrompt || ""),
      activeQueueSet: Math.max(1, Math.min(maxQueueSets, Math.floor(Number(message?.state?.activeQueueSet) || Number(latestSyncState.activeQueueSet) || 1))),
      generation: normalizeGenerationState(message?.state?.generation || latestSyncState.generation),
      styleSeedMode: String(message?.state?.styleSeedMode || latestSyncState.styleSeedMode || "same").trim().toLowerCase() === "different" ? "different" : "same",
    };
    latestSyncState.prompts = updatedState.prompts;
    latestSyncState.joinedPrompt = String(message?.state?.joinedPrompt || latestSyncState.joinedPrompt || "");
    latestSyncState.file = String(message?.state?.file || latestSyncState.file || "");
    latestSyncState.activeQueueSet = updatedState.activeQueueSet;
    latestSyncState.generation = { ...updatedState.generation };
    latestSyncState.styleSeedMode = updatedState.styleSeedMode;
    const generationByPrompt = Array.isArray(message?.state?.generationByPrompt)
      ? message.state.generationByPrompt.map((entry) => normalizeGenerationState(entry))
      : [];
    const promptSetIds = normalizePromptSetIds(
      message?.state?.promptSetIds,
      prompts.length,
      updatedState.activeQueueSet
    );
    const promptStyleNames = Array.isArray(message?.state?.promptStyleNames)
      ? prompts.map((_, index) => String(message.state.promptStyleNames[index] || "").trim())
      : prompts.map(() => "");
    const promptOutputSubfolders = Array.isArray(message?.state?.promptOutputSubfolders)
      ? prompts.map((_, index) => String(message.state.promptOutputSubfolders[index] || "").trim())
      : prompts.map(() => "");
    const promptSeedGroupIds = Array.isArray(message?.state?.promptSeedGroupIds)
      ? prompts.map((_, index) => String(message.state.promptSeedGroupIds[index] || `${promptSetIds[index]}:${index}`).trim())
      : prompts.map((_, index) => `${promptSetIds[index]}:${index}`);
    latestSyncState.promptSetIds = [...promptSetIds];

    isQueueing = true;
    let queuedCount = 0;
    let canceledByUser = false;
    let stoppedAfterCurrent = false;
    let queueAckSent = false;
    const tracker = createQueueTracker(requestId, prompts, {
      queueTargetType,
      saveNodeIds: Array.isArray(message?.saveNodeIds) ? message.saveNodeIds : [],
      sourceFile: String(message?.state?.sourceFile || message?.state?.file || "").trim(),
      promptSetIds,
      promptStyleNames,
      generation: updatedState.generation,
      generationByPrompt,
      promptOutputSubfolders,
      promptSeedGroupIds,
    });
    const seedGroupIndexById = new Map();
    const resolvePromptSeedGroupIndex = (promptIndex) => {
      const groupId = String(tracker.queuePlan.promptSeedGroupIds?.[promptIndex] || `${promptIndex}`).trim();
      if (!seedGroupIndexById.has(groupId)) {
        seedGroupIndexById.set(groupId, seedGroupIndexById.size);
      }
      return seedGroupIndexById.get(groupId) ?? promptIndex;
    };
    for (let promptIndex = 0; promptIndex < prompts.length; promptIndex += 1) {
      const generationForPrompt = tracker.queuePlan.generationByPrompt[promptIndex] || updatedState.generation;
      const effectiveSeed = resolveSeedForQueueRun(generationForPrompt, resolvePromptSeedGroupIndex(promptIndex));
      setTrackerPromptSeed(tracker, promptIndex, effectiveSeed);
    }
    try {
      for (let batchStart = 0; batchStart < tracker.total; batchStart += queueSubmitBatchSize) {
        const batchEnd = Math.min(tracker.total, batchStart + queueSubmitBatchSize);
        for (let promptIndex = batchStart; promptIndex < batchEnd; promptIndex += 1) {
          const canContinue = await waitWhileQueuePaused(requestId);
          if (!canContinue) {
            canceledByUser = true;
            throw new Error("Queue canceled by user.");
          }
          if (isQueueCancelRequestedFor(requestId)) {
            canceledByUser = true;
            throw new Error("Queue canceled by user.");
          }
          const prompt = String(tracker.prompts[promptIndex] || "");
          const generationForPrompt = tracker.queuePlan.generationByPrompt[promptIndex] || updatedState.generation;
          const promptSetId = clampQueueSetId(tracker.queuePlan.promptSetIds[promptIndex], updatedState.activeQueueSet);
          const promptOutputSubfolder = String(tracker.queuePlan.promptOutputSubfolders?.[promptIndex] || "").trim();
          const effectiveSeed = Math.max(0, Math.floor(Number(tracker.promptSeeds[promptIndex]) || 0));
          setTrackerPromptSeed(tracker, promptIndex, effectiveSeed);
          const generationForRun = {
            ...generationForPrompt,
            seed: effectiveSeed,
            controlAfterGenerate: "fixed",
          };
          if (promptIndex > 0) {
            const dispatchReady = await waitForQueueDispatchDelay(requestId, Date.now());
            if (!dispatchReady) {
              canceledByUser = true;
              throw new Error("Queue canceled by user.");
            }
          }
          updatedState.activePrompt = prompt;
          latestSyncState.activePrompt = prompt;
          if (!requestUsesApiWorkflow) {
            applySyncToAllNodes({
              ...updatedState,
              generation: generationForRun,
            }, { promptSetId, outputSubfolder: promptOutputSubfolder });
          }
          await waitForPowerPrompterComfyQueueDrain();
          currentQueueExecution = {
            requestId,
            promptIndex,
            prompt,
            submitted: false,
          };
          const queueResponse = await queueCurrentWorkflow({
            queueTargetType,
            apiWorkflow: requestUsesApiWorkflow ? message?.apiWorkflow : null,
            prompt,
            promptIndex,
            promptSetId,
            outputSubfolder: promptOutputSubfolder,
            state: {
              ...updatedState,
              generation: generationForRun,
            },
          });
          queuedCount += 1;
          const queuedPromptId = parsePromptIdFromQueueResult(queueResponse);
          setTrackerPromptId(tracker, promptIndex, queuedPromptId);
          if (currentQueueExecution && currentQueueExecution.requestId === requestId && currentQueueExecution.promptIndex === promptIndex) {
            currentQueueExecution = {
              ...currentQueueExecution,
              promptId: queuedPromptId,
              submitted: true,
            };
          }
          if (!queueAckSent) {
            sendWs({
              type: "queue_result",
              requestId,
              success: true,
              queued: tracker.total,
              total: tracker.total,
              promptIds: tracker.promptIds.map((entry) => String(entry || "")),
              promptSeeds: tracker.promptSeeds.map((entry) => Math.max(0, Math.floor(Number(entry) || 0))),
            });
            queueAckSent = true;
          }
          const completionStatus = await waitForPromptCompletion(requestId, promptIndex);
          if (completionStatus === "canceled") {
            canceledByUser = true;
            throw new Error("Queue canceled by user.");
          }
          if (completionStatus === "timeout") {
            throw new Error("Prompt completion timed out. Check workflow validation/save-node outputs.");
          }
          if (completionStatus !== "completed") {
            throw new Error(`Prompt completion tracker ended unexpectedly (${completionStatus}).`);
          }
          await waitForComfyPromptIdToDrain(queuedPromptId);
          lastQueueDispatchCompletedAt = Date.now();
          if (shouldStopAfterCurrentFor(requestId)) {
            stoppedAfterCurrent = true;
            break;
          }
          await sleep(queueSubmitBetweenPromptsMs);
        }
        if (stoppedAfterCurrent) break;
        if (batchEnd < tracker.total) {
          const canContinue = await waitWhileQueuePaused(requestId);
          if (!canContinue) {
            canceledByUser = true;
            throw new Error("Queue canceled by user.");
          }
          if (isQueueCancelRequestedFor(requestId)) {
            canceledByUser = true;
            throw new Error("Queue canceled by user.");
          }
          await sleep(queueSubmitBetweenBatchesMs);
        }
      }
    if (!queueAckSent) {
        sendWs({
          type: "queue_result",
          requestId,
          success: true,
          queued: tracker.total,
          total: tracker.total,
          promptIds: tracker.promptIds.map((entry) => String(entry || "")),
          promptSeeds: tracker.promptSeeds.map((entry) => Math.max(0, Math.floor(Number(entry) || 0))),
        });
        queueAckSent = true;
      }
      if (stoppedAfterCurrent) {
        queueStopAfterCurrentRequestIds.delete(requestId);
        if (activeQueueTrackers.has(requestId)) {
          finalizeTracker(activeQueueTrackers.get(requestId), "queue_cleared");
        }
      }
    } catch (error) {
      sendWs({
        type: "queue_result",
        requestId,
        success: false,
        canceled: canceledByUser,
        queued: queuedCount,
        total: tracker.total,
        runtime: queueAckSent,
        error: String(error?.message || error || "Unknown queue failure"),
      });
      cancelTrackerByRequestId(requestId, canceledByUser ? "queue_canceled" : "queue_failed");
    } finally {
      isQueueing = false;
      queueStopAfterCurrentRequestIds.delete(requestId);
      clearQueueCancelState();
      emitQueuePauseState("queue_finished");
    }
  }

  async function pumpQueueRequests() {
    if (queuePumpActive) return;
    queuePumpActive = true;
    try {
      let hasProcessedRequest = false;
      while (pendingQueueRequests.length > 0) {
        const nextPreview = pendingQueueRequests[0];
        const nextRequestId = String(nextPreview?.requestId || "").trim();
        if (hasProcessedRequest && nextRequestId && lastQueueDispatchCompletedAt > 0) {
          const dispatchReady = await waitForQueueDispatchDelay(nextRequestId, lastQueueDispatchCompletedAt);
          if (!dispatchReady) {
            const pendingIndex = findPendingQueueRequestIndex(nextRequestId);
            if (pendingIndex >= 0) {
              pendingQueueRequests.splice(pendingIndex, 1);
            }
            continue;
          }
        }
        const next = pendingQueueRequests.shift();
        if (!next) continue;
        await processQueueRequest(next);
        hasProcessedRequest = true;
      }
    } finally {
      queuePumpActive = false;
      if (pendingQueueRequests.length > 0) {
        setTimeout(() => {
          void pumpQueueRequests();
        }, 0);
      }
    }
  }

  function handleQueueRequest(message) {
    const requestId = String(message?.requestId || "").trim();
    if (requestId && isQueueRequestAlreadyTracked(requestId)) {
      sendWs({
        type: "queue_forwarded",
        requestId,
        success: true,
        duplicate: true,
        targetRole: "comfy_bridge",
      });
      emitQueuePauseState("queue_duplicate_ignored");
      return;
    }
    pendingQueueRequests.push(message);
    emitQueuePauseState("queue_request");
    void pumpQueueRequests();
  }

  function handleQueueCancelRequest(message) {
    const idsFromArray = Array.isArray(message?.requestIds)
      ? message.requestIds
      : [];
    const ids = idsFromArray
      .map((entry) => String(entry || "").trim())
      .filter((entry) => entry.length > 0);
    const singleId = String(message?.requestId || "").trim();
    const uniqueIds = Array.from(new Set(ids));

    const canceledPendingRequestIds = [];
    if (uniqueIds.length > 0) {
      for (let idx = pendingQueueRequests.length - 1; idx >= 0; idx -= 1) {
        const pending = pendingQueueRequests[idx];
        const pendingRequestId = String(pending?.requestId || "").trim();
        if (!pendingRequestId || !uniqueIds.includes(pendingRequestId)) continue;
        pendingQueueRequests.splice(idx, 1);
        canceledPendingRequestIds.push(pendingRequestId);
      }
    } else if (pendingQueueRequests.length > 0) {
      while (pendingQueueRequests.length > 0) {
        const pending = pendingQueueRequests.pop();
        const pendingRequestId = String(pending?.requestId || "").trim();
        if (pendingRequestId) canceledPendingRequestIds.push(pendingRequestId);
      }
    }

    for (const canceledId of canceledPendingRequestIds) {
      sendWs({
        type: "queue_result",
        requestId: canceledId,
        success: false,
        canceled: true,
        queued: 0,
        total: 0,
        error: "Queue canceled by user.",
      });
    }

    markQueueCancelRequested(uniqueIds);
    if (uniqueIds.length > 0) {
      for (const requestId of uniqueIds) {
        cancelTrackerByRequestId(requestId, "queue_canceled");
      }
    } else if (activeQueueTrackers.size > 0) {
      for (const tracker of Array.from(activeQueueTrackers.values())) {
        cancelTrackerByRequestId(tracker.requestId, "queue_canceled");
      }
    }

    sendWs({
      type: "queue_cancel_result",
      requestId: singleId || "",
      requestIds: uniqueIds,
      success: true,
    });
    emitQueuePauseState("queue_cancel");
  }

  function handleQueuePauseToggle(nextPaused) {
    queuePaused = nextPaused === true;
    emitQueuePauseState(queuePaused ? "paused" : "resumed");
  }

  function handleQueueClearFutureRequest(message) {
    const activeRequestId = String(message?.activeRequestId || "").trim();
    const idsFromArray = Array.isArray(message?.requestIds) ? message.requestIds : [];
    const requestIds = Array.from(new Set(
      idsFromArray.map((entry) => String(entry || "").trim()).filter((entry) => entry.length > 0)
    ));

    const clearedPendingRequestIds = [];
    for (let idx = pendingQueueRequests.length - 1; idx >= 0; idx -= 1) {
      const pending = pendingQueueRequests[idx];
      const pendingRequestId = String(pending?.requestId || "").trim();
      if (!pendingRequestId) continue;
      if (activeRequestId && pendingRequestId === activeRequestId) continue;
      if (requestIds.length > 0 && !requestIds.includes(pendingRequestId)) continue;
      pendingQueueRequests.splice(idx, 1);
      clearedPendingRequestIds.push(pendingRequestId);
    }

    for (const clearedId of clearedPendingRequestIds) {
      sendWs({
        type: "queue_result",
        requestId: clearedId,
        success: false,
        canceled: true,
        queued: 0,
        total: 0,
        error: "Queue cleared by user.",
      });
    }

    for (const requestId of requestIds) {
      if (!requestId || requestId === activeRequestId) continue;
      cancelTrackerByRequestId(requestId, "queue_cleared");
    }

    if (activeRequestId) {
      queueStopAfterCurrentRequestIds.add(activeRequestId);
    }

    sendWs({
      type: "queue_clear_future_result",
      requestId: String(message?.requestId || "").trim(),
      activeRequestId,
      success: true,
      clearedRequestIds: clearedPendingRequestIds,
    });
    emitQueuePauseState("queue_clear_future");
  }

  function handleQueueInterruptActiveRequest(message) {
    const requestId = String(message?.requestId || "").trim();
    const skipped = markActivePromptInterrupted();
    sendWs({
      type: "queue_interrupt_result",
      requestId,
      success: skipped,
    });
    emitQueuePauseState("queue_interrupt");
  }

  function handleComfyExecuted(event) {
    if (activeQueueTrackers.size === 0) return;
    clearAllIdleFallbackTimers();
    const detail = event?.detail;
    const promptId = extractPromptIdFromExecution(detail);
    if (!promptId) return;
    const resolved = resolveTrackerForPromptId(promptId, { allowActiveFallback: false, allowAnyFallback: false });
    if (!resolved) return;
    const nodeType = extractNodeTypeFromExecution(detail);
    const nodeId = extractExecutionNodeId(detail);
    const isSaveOutputNode = saveNodeTypes.has(nodeType)
      || (resolved.tracker.queueTargetType === "api_workflow" && resolved.tracker.saveNodeIds.has(nodeId));
    if (!isSaveOutputNode) return;
    const output = detail?.output ?? detail?.outputs ?? detail?.result;
    if (!payloadContainsImageOutput(output)) return;
    emitSavedOutputs(resolved.tracker, resolved.promptIndex, extractSavedOutputEntries(output));
    markPromptCompleted(resolved.tracker, resolved.promptIndex, "save_output");
  }

  function handleComfyExecutionSuccess(event) {
    if (activeQueueTrackers.size === 0) return;
    clearAllIdleFallbackTimers();
    const detail = event?.detail ?? event;
    const promptId = extractPromptIdFromExecution(detail);
    if (!promptId) return;
    const resolved = resolveTrackerForPromptId(promptId, { allowActiveFallback: false, allowAnyFallback: false });
    if (!resolved) return;
    if (resolved.tracker.completedFlags[resolved.promptIndex]) return;
    const expectsSaveOutput = resolved.tracker.queueTargetType === "api_workflow"
      ? resolved.tracker.saveNodeIds.size > 0
      : hasSaveNodes();
    if (expectsSaveOutput) return;
    markPromptCompleted(resolved.tracker, resolved.promptIndex, "execution_success");
  }

  function handleComfyProgress(event) {
    if (activeQueueTrackers.size === 0) return;
    clearAllIdleFallbackTimers();
    const detail = event?.detail;
    if (!detail || typeof detail !== "object") return;
    const promptId = extractPromptIdFromExecution(detail);
    if (promptId) {
      currentExecutingPromptId = promptId;
    }
    const resolved = resolveTrackerForLiveComfyEvent(promptId);
    if (!resolved) return;
    const progress = extractProgressValue(detail);
    if (progress.hasProgress) {
      setTrackerProgress(resolved.tracker, resolved.promptIndex, progress.step, progress.maxStep);
      emitJobProgress(resolved.tracker, resolved.promptIndex);
    }

    const previewPayload = normalizePreviewPayload(detail);
    if (!previewPayload) return;
    const now = Date.now();
    if (now - lastPreviewFrameAt < previewFrameThrottleMs) return;
    if (previewFrameInFlight) return;

    previewFrameInFlight = true;
    Promise.resolve()
      .then(async () => {
        const dataUrl = previewPayload.type === "data_url"
          ? String(previewPayload.value || "")
          : await blobToDataUrl(previewPayload.value);
        if (!dataUrl || dataUrl.length > previewMaxDataUrlLength) return;
        if (!activeQueueTrackers.has(resolved.tracker.requestId)) return;
        lastPreviewFrameAt = Date.now();
        emitGenerationPreview(resolved.tracker, resolved.promptIndex, dataUrl);
      })
      .catch(() => {
        // best effort
      })
      .finally(() => {
        previewFrameInFlight = false;
      });
  }

  async function handleComfyPreview(event) {
    if (activeQueueTrackers.size === 0) return;
    clearAllIdleFallbackTimers();
    const detail = event?.detail ?? event;
    const promptId = extractPromptIdFromExecution(detail);
    if (promptId) {
      currentExecutingPromptId = promptId;
    }
    const now = Date.now();
    if (now - lastPreviewFrameAt < previewFrameThrottleMs) return;
    if (previewFrameInFlight) return;

    const { previewId, payload: previewPayload } = await normalizePreviewPayloadFromEvent(detail, event?.type || "");
    if (!previewPayload) return;

    const resolved = resolveTrackerForLiveComfyEvent(promptId || previewId || currentExecutingPromptId || "");
    if (!resolved) return;
    if (resolved.tracker.completedFlags[resolved.promptIndex]) return;

    previewFrameInFlight = true;
    try {
      const dataUrl = previewPayload.type === "data_url"
        ? String(previewPayload.value || "")
        : await blobToDataUrl(previewPayload.value);
      if (!dataUrl || dataUrl.length > previewMaxDataUrlLength) return;
      if (!activeQueueTrackers.has(resolved.tracker.requestId)) return;
      lastPreviewFrameAt = Date.now();
      emitGenerationPreview(resolved.tracker, resolved.promptIndex, dataUrl);
    } catch {
      // best effort
    } finally {
      previewFrameInFlight = false;
    }
  }

  function handleComfyIdle(event) {
    if (activeQueueTrackers.size === 0) return;
    if (event?.detail !== null) {
      const promptId = extractPromptIdFromExecution(event?.detail);
      if (promptId) {
        currentExecutingPromptId = promptId;
      }
      clearAllIdleFallbackTimers();
      return;
    }
    currentExecutingPromptId = "";
    if (currentQueueExecution && currentQueueExecution.submitted !== true) {
      return;
    }
    const activeTracker = currentQueueExecution
      ? activeQueueTrackers.get(String(currentQueueExecution.requestId || "").trim())
      : null;
    const requestHasSaveNodes = activeTracker
      ? (
        activeTracker.queueTargetType === "api_workflow"
          ? activeTracker.saveNodeIds.size > 0
          : hasSaveNodes()
      )
      : hasSaveNodes();
    if (requestHasSaveNodes) {
      const activePromptIndex = Number.isFinite(currentQueueExecution?.promptIndex)
        ? Math.max(0, Math.floor(currentQueueExecution.promptIndex))
        : -1;
      if (activeTracker && activePromptIndex >= 0 && activePromptIndex < activeTracker.total) {
        scheduleIdleFallback(activeTracker, activePromptIndex);
      }
      return;
    }

    for (const tracker of Array.from(activeQueueTrackers.values())) {
      if (tracker.completedBySaveCount > 0) continue;
      scheduleIdleFallback(tracker);
    }
  }

  return {
    clearQueueTrackers,
    emitQueuePauseState,
    processQueueRequest,
    handleComfyExecuted,
    handleComfyExecutionSuccess,
    handleComfyProgress,
    handleComfyPreview,
    handleComfyIdle,
    handleQueueRequest,
    handleQueueCancelRequest,
    handleQueuePauseToggle,
    handleQueueClearFutureRequest,
    handleQueueInterruptActiveRequest,
    handleQueueReorder,
    handleQueuePromptRemove,
    handleQueueDispatchDelayUpdate,
    pumpQueueRequests,
  };
}

function normalizePromptSetIds(rawValue, count, fallbackSetId) {
  const total = Math.max(0, Math.floor(Number(count) || 0));
  const fallback = Number.isFinite(fallbackSetId) ? Math.max(1, Math.floor(fallbackSetId)) : 1;
  const values = Array.isArray(rawValue) ? rawValue : [];
  const result = new Array(total);
  for (let idx = 0; idx < total; idx += 1) {
    const nextValue = Number(values[idx]);
    result[idx] = Number.isFinite(nextValue) ? Math.max(1, Math.floor(nextValue)) : fallback;
  }
  return result;
}
