import { app } from "../../../scripts/app.js";

const REQUEST_GET_NODES = "UMBRA_COMFY_GET_IMAGE_NODES";
const RESPONSE_GET_NODES = "UMBRA_COMFY_IMAGE_NODES";
const REQUEST_ASSIGN_IMAGE = "UMBRA_COMFY_ASSIGN_IMAGE";
const RESPONSE_ASSIGN_IMAGE = "UMBRA_COMFY_ASSIGN_RESULT";
const REQUEST_HANDOFF_IMAGES = "UMBRA_COMFY_HANDOFF_IMAGES";
const RESPONSE_HANDOFF_IMAGES = "UMBRA_COMFY_HANDOFF_RESULT";
const REQUEST_LOAD_WORKFLOW = "UMBRA_COMFY_LOAD_WORKFLOW";
const RESPONSE_LOAD_WORKFLOW = "UMBRA_COMFY_LOAD_WORKFLOW_RESULT";

function normalizeNodeId(value) {
  const id = Number(value);
  return Number.isFinite(id) ? id : null;
}

function getCandidateImageWidget(node) {
  if (!node || !Array.isArray(node.widgets)) return null;
  const preferredNames = new Set(["image", "image_path", "filename", "file", "upload"]);

  for (const widget of node.widgets) {
    const name = String(widget?.name || "").toLowerCase();
    if (!name) continue;

    const hasSelectableValues = Array.isArray(widget?.options?.values);
    const hasStringValue = typeof widget?.value === "string" || widget?.value == null;
    if (!hasSelectableValues && !hasStringValue) continue;

    if (preferredNames.has(name) || name.includes("image")) {
      return widget;
    }
  }
  return null;
}

function isImageLoaderNode(node) {
  const type = String(node?.type || "").toLowerCase();
  const title = String(node?.title || "").toLowerCase();
  const hasImageWidget = Boolean(getCandidateImageWidget(node));
  if (hasImageWidget) return true;
  return type.includes("loadimage") || type.includes("load_image") || title.includes("load image");
}

function getSelectedNodeIds() {
  const selectedNodes = app?.canvas?.selected_nodes;
  if (!selectedNodes || typeof selectedNodes !== "object") return new Set();
  const ids = Object.keys(selectedNodes)
    .map((value) => normalizeNodeId(value))
    .filter((value) => value !== null);
  return new Set(ids);
}

function listImageNodes() {
  const graphNodes = app?.graph?._nodes;
  if (!Array.isArray(graphNodes)) return [];

  const selectedIds = getSelectedNodeIds();
  const rows = [];
  for (const node of graphNodes) {
    const nodeId = normalizeNodeId(node?.id);
    if (nodeId === null) continue;
    if (!isImageLoaderNode(node)) continue;

    const widget = getCandidateImageWidget(node);
    rows.push({
      id: nodeId,
      title: String(node?.title || node?.type || `Node ${nodeId}`),
      type: String(node?.type || "Unknown"),
      selected: selectedIds.has(nodeId),
      widgetName: widget?.name ? String(widget.name) : null,
      widgetValue: widget?.value != null ? String(widget.value) : "",
    });
  }

  rows.sort((a, b) => {
    if (a.selected !== b.selected) return a.selected ? -1 : 1;
    return a.id - b.id;
  });
  return rows;
}

function assignImageToNode(nodeIdRaw, filenameRaw) {
  const nodeId = normalizeNodeId(nodeIdRaw);
  const filename = String(filenameRaw || "").trim();
  if (nodeId === null) return { ok: false, error: "Invalid nodeId." };
  if (!filename) return { ok: false, error: "Missing filename." };

  const graph = app?.graph;
  if (!graph || !graph._nodes_by_id) return { ok: false, error: "Comfy graph is not available." };

  const node = graph._nodes_by_id[nodeId];
  if (!node) return { ok: false, error: `Node #${nodeId} was not found.` };

  const widget = getCandidateImageWidget(node);
  if (!widget) return { ok: false, error: `Node #${nodeId} has no image widget.` };

  widget.value = filename;
  if (typeof widget.callback === "function") {
    try {
      widget.callback(filename, app, node);
    } catch {
      // keep assignment alive even if widget callback throws
    }
  }

  if (typeof node.onWidgetChanged === "function") {
    try {
      node.onWidgetChanged(widget.name, filename, null, widget);
    } catch {
      // best effort
    }
  }

  if (typeof graph.setDirtyCanvas === "function") {
    graph.setDirtyCanvas(true, true);
  }
  if (typeof app?.canvas?.setDirty === "function") {
    app.canvas.setDirty(true, true);
  }

  return {
    ok: true,
    nodeId,
    filename,
    widgetName: widget?.name ? String(widget.name) : null,
  };
}

function createLoadImageNode() {
  const graph = app?.canvas?.graph || app?.graph;
  const LiteGraphApi = window.LiteGraph || globalThis.LiteGraph;
  if (!graph || !LiteGraphApi || typeof LiteGraphApi.createNode !== "function") {
    return { ok: false, error: "Comfy graph node creation is not available." };
  }

  const node = LiteGraphApi.createNode("LoadImage");
  if (!node) return { ok: false, error: "Could not create a Load Image node." };

  if (app?.canvas?.graph_mouse && Array.isArray(app.canvas.graph_mouse)) {
    node.pos = [app.canvas.graph_mouse[0], app.canvas.graph_mouse[1]];
  } else {
    const nodes = Array.isArray(graph._nodes) ? graph._nodes : [];
    const maxX = nodes.reduce((value, item) => Math.max(value, Number(item?.pos?.[0] || 0)), 0);
    node.pos = [maxX + 80, 80];
  }

  graph.add(node);
  if (typeof graph.setDirtyCanvas === "function") {
    graph.setDirtyCanvas(true, true);
  }
  return { ok: true, node };
}

function handoffImages(filenamesRaw) {
  const filenames = Array.isArray(filenamesRaw)
    ? filenamesRaw.map((value) => String(value || "").trim()).filter(Boolean)
    : [];
  if (filenames.length === 0) return { ok: false, error: "No filenames provided." };

  let nodes = listImageNodes();
  let targetNode = nodes.find((node) => node.selected) || nodes[0] || null;
  let createdNode = false;

  if (!targetNode) {
    const created = createLoadImageNode();
    if (!created.ok) return created;
    createdNode = true;
    nodes = listImageNodes();
    const createdNodeId = normalizeNodeId(created.node?.id);
    targetNode = nodes.find((node) => node.id === createdNodeId) || nodes[0] || null;
  }

  if (!targetNode) return { ok: false, error: "No Load Image-style nodes found." };

  const assigned = assignImageToNode(targetNode.id, filenames[0]);
  if (!assigned.ok) return assigned;

  return {
    ok: true,
    createdNode,
    assigned: {
      nodeId: assigned.nodeId,
      filename: assigned.filename,
      widgetName: assigned.widgetName,
    },
    skipped: filenames.slice(1),
    nodes: listImageNodes(),
  };
}

function extractPromptGraph(workflowRaw) {
  const workflow = workflowRaw && typeof workflowRaw === "object" ? workflowRaw : null;
  if (!workflow || Array.isArray(workflow)) return null;
  const directEntries = Object.values(workflow);
  if (
    directEntries.length > 0
    && directEntries.every((entry) => entry && typeof entry === "object" && !Array.isArray(entry) && String(entry.class_type || "").trim())
  ) {
    return workflow;
  }
  for (const key of ["prompt", "output"]) {
    const candidate = workflow[key];
    if (!candidate || typeof candidate !== "object" || Array.isArray(candidate)) continue;
    const entries = Object.values(candidate);
    if (
      entries.length > 0
      && entries.every((entry) => entry && typeof entry === "object" && !Array.isArray(entry) && String(entry.class_type || "").trim())
    ) {
      return candidate;
    }
  }
  return null;
}

function setNodeWidgetValue(node, name, value) {
  const widgets = Array.isArray(node?.widgets) ? node.widgets : [];
  const widget = widgets.find((entry) => String(entry?.name || "") === String(name || ""));
  if (!widget) {
    node.properties = node.properties || {};
    node.properties[name] = value;
    return;
  }
  widget.value = value;
  if (typeof widget.callback === "function") {
    try {
      widget.callback(value, app, node, undefined, widget);
    } catch {
      // Keep workflow loading alive if a widget callback rejects API values.
    }
  }
  if (typeof node.onWidgetChanged === "function") {
    try {
      node.onWidgetChanged(widget.name, value, null, widget);
    } catch {
      // Best effort.
    }
  }
}

function findNodeInputIndex(node, inputName) {
  const inputs = Array.isArray(node?.inputs) ? node.inputs : [];
  const exactIndex = inputs.findIndex((entry) => String(entry?.name || "") === String(inputName || ""));
  if (exactIndex >= 0) return exactIndex;
  const normalizedInputName = String(inputName || "").toLowerCase();
  const looseIndex = inputs.findIndex((entry) => String(entry?.name || "").toLowerCase() === normalizedInputName);
  return looseIndex >= 0 ? looseIndex : -1;
}

function buildGraphFromApiPrompt(promptGraph, workflowNameRaw) {
  const graph = app?.canvas?.graph || app?.graph;
  const LiteGraphApi = window.LiteGraph || globalThis.LiteGraph;
  if (!graph || !LiteGraphApi || typeof LiteGraphApi.createNode !== "function") {
    return { ok: false, error: "Comfy graph node creation is not available." };
  }

  if (typeof graph.clear === "function") graph.clear();

  const rawEntries = Object.entries(promptGraph || {});
  const createdByRawId = new Map();
  const pendingLinks = [];
  let row = 0;
  let col = 0;

  for (const [rawId, spec] of rawEntries) {
    const classType = String(spec?.class_type || "").trim();
    if (!classType) continue;
    const node = LiteGraphApi.createNode(classType);
    if (!node) continue;
    node.title = String(spec?._meta?.title || node.title || classType);
    node.pos = [80 + col * 360, 80 + row * 220];
    col += 1;
    if (col >= 4) {
      col = 0;
      row += 1;
    }
    graph.add(node);
    createdByRawId.set(String(rawId), node);
  }

  for (const [rawId, spec] of rawEntries) {
    const node = createdByRawId.get(String(rawId));
    if (!node) continue;
    const inputs = spec?.inputs && typeof spec.inputs === "object" && !Array.isArray(spec.inputs)
      ? spec.inputs
      : {};
    for (const [inputName, value] of Object.entries(inputs)) {
      if (Array.isArray(value) && value.length >= 2) {
        pendingLinks.push({
          sourceId: String(value[0]),
          sourceSlot: Math.max(0, Math.floor(Number(value[1]) || 0)),
          target: node,
          inputName,
        });
        continue;
      }
      setNodeWidgetValue(node, inputName, value);
    }
  }

  let linked = 0;
  for (const link of pendingLinks) {
    const source = createdByRawId.get(link.sourceId);
    if (!source || !link.target || typeof source.connect !== "function") continue;
    const inputIndex = findNodeInputIndex(link.target, link.inputName);
    if (inputIndex < 0) continue;
    try {
      source.connect(link.sourceSlot, link.target, inputIndex);
      linked += 1;
    } catch {
      // Skip links whose node definitions are not ready or have changed.
    }
  }

  if (typeof graph.setDirtyCanvas === "function") graph.setDirtyCanvas(true, true);
  if (typeof app?.canvas?.setDirty === "function") app.canvas.setDirty(true, true);
  if (typeof app?.canvas?.zoomToFit === "function") {
    try {
      app.canvas.zoomToFit();
    } catch {
      // Optional view nicety.
    }
  }

  return {
    ok: true,
    mode: "api_prompt_graph",
    workflowName: String(workflowNameRaw || ""),
    nodes: createdByRawId.size,
    links: linked,
  };
}

async function loadWorkflowIntoComfy(workflowRaw, workflowNameRaw) {
  const workflow = workflowRaw && typeof workflowRaw === "object" ? workflowRaw : null;
  if (!workflow) return { ok: false, error: "Missing workflow document." };

  const graphWorkflow = Array.isArray(workflow?.nodes)
    ? workflow
    : (workflow.workflow && typeof workflow.workflow === "object" && Array.isArray(workflow.workflow.nodes) ? workflow.workflow : null);
  if (graphWorkflow && typeof app?.loadGraphData === "function") {
    await app.loadGraphData(graphWorkflow);
    return {
      ok: true,
      mode: "comfy_workflow",
      workflowName: String(workflowNameRaw || ""),
      nodes: Array.isArray(graphWorkflow.nodes) ? graphWorkflow.nodes.length : 0,
    };
  }

  const promptGraph = extractPromptGraph(workflow);
  if (!promptGraph) return { ok: false, error: "Selected file is not a ComfyUI workflow or API prompt graph." };
  return buildGraphFromApiPrompt(promptGraph, workflowNameRaw);
}

function postBridgeResponse(origin, payload) {
  const target = window.parent || window;
  const targetOrigin = origin && origin !== "null" ? origin : "*";
  target.postMessage(payload, targetOrigin);
}

function handleParentMessage(event) {
  if (event.source !== window.parent) return;
  const payload = event.data;
  if (!payload || typeof payload !== "object") return;

  const type = payload.type;
  const requestId = payload.requestId;
  if (!type || !requestId) return;

  if (type === REQUEST_GET_NODES) {
    postBridgeResponse(event.origin, {
      type: RESPONSE_GET_NODES,
      requestId,
      nodes: listImageNodes(),
    });
    return;
  }

  if (type === REQUEST_ASSIGN_IMAGE) {
    const result = assignImageToNode(payload.nodeId, payload.filename);
    postBridgeResponse(event.origin, {
      type: RESPONSE_ASSIGN_IMAGE,
      requestId,
      ...result,
    });
    return;
  }

  if (type === REQUEST_HANDOFF_IMAGES) {
    const result = handoffImages(payload.filenames);
    postBridgeResponse(event.origin, {
      type: RESPONSE_HANDOFF_IMAGES,
      requestId,
      ...result,
    });
    return;
  }

  if (type === REQUEST_LOAD_WORKFLOW) {
    Promise.resolve(loadWorkflowIntoComfy(payload.workflow, payload.workflowName))
      .then((result) => {
        postBridgeResponse(event.origin, {
          type: RESPONSE_LOAD_WORKFLOW,
          requestId,
          ...result,
        });
      })
      .catch((error) => {
        postBridgeResponse(event.origin, {
          type: RESPONSE_LOAD_WORKFLOW,
          requestId,
          ok: false,
          error: String(error?.message || error || "Failed to load workflow."),
        });
      });
  }
}

app.registerExtension({
  name: "umbra.comfy.bridge",
  setup() {
    window.addEventListener("message", handleParentMessage);
    console.log("[UmbraBridge] Comfy bridge extension ready");
  },
});
