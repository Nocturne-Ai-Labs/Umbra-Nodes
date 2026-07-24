import { app } from "../../../scripts/app.js";

const TARGET_NODE_TYPES = new Set([
  "UmbraLabSaveImage",
  "UmbraLabSaveImageSimple",
]);

const STRING_WIDGET_NAMES = new Set([
  "filename_prefix",
  "positive_prompt",
  "negative_prompt",
  "output_folder",
  "set_subfolder",
  "model_name",
  "sampler_name",
  "sampler_name_text",
  "scheduler",
]);

function sanitizeWidget(widget) {
  if (!widget) return false;
  const name = String(widget?.name || "");
  if (!STRING_WIDGET_NAMES.has(name)) return false;
  if (widget.value == null) {
    widget.value = "";
    return true;
  }
  if (typeof widget.value !== "string") {
    widget.value = String(widget.value);
    return true;
  }
  return false;
}

function sanitizeNodeWidgets(node) {
  if (!node || !Array.isArray(node.widgets)) return false;
  let changed = false;
  for (const widget of node.widgets) {
    changed = sanitizeWidget(widget) || changed;
  }
  return changed;
}

function sanitizeConfigureData(node, info) {
  if (!info || !Array.isArray(info.widgets_values)) return;
  const widgets = Array.isArray(node?.widgets) ? node.widgets : [];
  for (let idx = 0; idx < info.widgets_values.length; idx += 1) {
    const current = info.widgets_values[idx];
    if (current !== null && current !== undefined) continue;
    const widgetName = String(widgets[idx]?.name || "");
    if (!widgetName || STRING_WIDGET_NAMES.has(widgetName)) {
      info.widgets_values[idx] = "";
    }
  }
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
    // best effort only
  }
}

function wrapConfigure(nodeType) {
  const original = nodeType.prototype?.onConfigure;
  nodeType.prototype.onConfigure = function onConfigurePatched(...args) {
    sanitizeConfigureData(this, args[0]);
    const result = typeof original === "function" ? original.apply(this, args) : undefined;
    const changed = sanitizeNodeWidgets(this);
    if (changed) markCanvasDirty();
    return result;
  };
}

function wrapNodeCreated(nodeType) {
  const original = nodeType.prototype?.onNodeCreated;
  nodeType.prototype.onNodeCreated = function onNodeCreatedPatched(...args) {
    const result = typeof original === "function" ? original.apply(this, args) : undefined;
    const changed = sanitizeNodeWidgets(this);
    if (changed) markCanvasDirty();
    return result;
  };
}

app.registerExtension({
  name: "umbra.saveimage.sanitize",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (!nodeType || !nodeData) return;
    const className = String(nodeData?.name || "").trim();
    if (!TARGET_NODE_TYPES.has(className)) return;
    wrapConfigure(nodeType);
    wrapNodeCreated(nodeType);
  },
  setup() {
    const scanAndSanitize = () => {
      const graphNodes = app?.graph?._nodes;
      if (!Array.isArray(graphNodes)) return;
      let changed = false;
      for (const node of graphNodes) {
        const type = String(node?.type || "");
        if (!TARGET_NODE_TYPES.has(type)) continue;
        changed = sanitizeNodeWidgets(node) || changed;
      }
      if (changed) markCanvasDirty();
    };

    scanAndSanitize();
    setTimeout(scanAndSanitize, 250);
  },
});
