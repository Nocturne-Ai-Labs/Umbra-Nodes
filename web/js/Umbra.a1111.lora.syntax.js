import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";
import { $el, ComfyDialog } from "../../../scripts/ui.js";

const NODE_NAME = "UmbraA1111LoraSyntax";
const NONE_OPTION = "[None]";
const LORA_METADATA_ENDPOINTS = ["/easyuse/metadata/", "/pysssss/metadata/", "/umbra/metadata/"];
const LORA_DESCRIPTION_ALLOWED_TAGS = new Set([
  "p", "br",
  "strong", "b", "em", "i", "u", "s",
  "ul", "ol", "li",
  "h1", "h2", "h3", "h4", "h5", "h6",
  "blockquote", "code", "pre",
  "a",
]);
const LORA_DESCRIPTION_ALLOWED_PROTOCOLS = new Set(["http:", "https:"]);

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

function findWidget(node, name) {
  if (!node || !Array.isArray(node.widgets)) return null;
  return node.widgets.find((widget) => String(widget?.name || "") === name) || null;
}

function normalizeLoraForTag(selectedValue) {
  const value = String(selectedValue || "").trim();
  if (!value || value === NONE_OPTION) return "";
  const normalized = value.replace(/\\/g, "/");
  return normalized.replace(/\.[^/.]+$/, "");
}

function numberToTagValue(input, fallback) {
  const numeric = Number(input);
  if (!Number.isFinite(numeric)) return String(fallback);
  return String(numeric).replace(/\.0+$/, "").replace(/(\.\d*?)0+$/, "$1");
}

function buildTag(node, selectedValue) {
  const tagName = normalizeLoraForTag(selectedValue);
  if (!tagName) return "";
  const modelStrength = findWidget(node, "lora_strength_model")?.value;
  const clipStrength = findWidget(node, "lora_strength_clip")?.value;
  const modelValue = numberToTagValue(modelStrength, 1);
  const clipValue = numberToTagValue(clipStrength, modelValue);
  return `<lora:${tagName}:${modelValue}:${clipValue}>`;
}

function getSelectedLoraValue(node) {
  const loraWidget = findWidget(node, "lora_name") || findWidget(node, "selected_lora");
  return String(loraWidget?.value || "").trim();
}

function makeInfoRow(label, value) {
  if (value === null || value === undefined || String(value).trim() === "") return null;
  return $el("p", [
    $el("label", { textContent: `${label}: ` }),
    $el("span", { textContent: String(value) }),
  ]);
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

function normalizeHttpUrl(rawValue) {
  const raw = String(rawValue || "").trim();
  if (!raw) return "";
  try {
    const parsed = new URL(raw, window.location.origin);
    if (!LORA_DESCRIPTION_ALLOWED_PROTOCOLS.has(parsed.protocol)) return "";
    return parsed.href;
  } catch {
    return "";
  }
}

class UmbraLoraInfoDialog extends ComfyDialog {
  constructor(name) {
    super();
    this.name = String(name || "");
    this.element.classList.add("umbra-lora-info");
  }

  normalizeLoraMetadataName(rawName) {
    return String(rawName || "").trim().replace(/\\/g, "/").replace(/^\/+/, "");
  }

  removeLoraFileExtension(rawName) {
    return String(rawName || "").replace(/\.(safetensors|ckpt|pt|pth|bin)$/i, "");
  }

  buildLoraMetadataCandidates(rawName) {
    const normalized = this.normalizeLoraMetadataName(rawName);
    if (!normalized) return [];
    const parts = normalized.split("/").filter(Boolean);
    const basename = parts.length > 0 ? parts[parts.length - 1] : normalized;
    const noExtNormalized = this.removeLoraFileExtension(normalized);
    const noExtBasename = this.removeLoraFileExtension(basename);
    const candidates = [normalized, basename, noExtNormalized, noExtBasename];
    const unique = [];
    const seen = new Set();
    for (const candidate of candidates) {
      const clean = String(candidate || "").trim();
      if (!clean) continue;
      const key = clean.toLowerCase();
      if (seen.has(key)) continue;
      seen.add(key);
      unique.push(clean);
    }
    return unique;
  }

  extractLoraHash(metadata) {
    return String(
      metadata?.["pysssss.sha256"] ||
      metadata?.["easyuse.sha256"] ||
      metadata?.sha256 ||
      metadata?.["sshs_model_hash"] ||
      metadata?.["ss_sd_model_hash"] ||
      metadata?.["modelspec.hash_sha256"] ||
      "",
    ).trim();
  }

  async fetchLocalMetadata(name) {
    const candidates = this.buildLoraMetadataCandidates(name);
    if (candidates.length === 0) {
      throw new Error("LoRA name is required.");
    }

    const errors = [];
    for (const endpoint of LORA_METADATA_ENDPOINTS) {
      for (const candidate of candidates) {
        const rel = `loras/${candidate}`;
        try {
          const response = await api.fetchApi(endpoint + encodeURIComponent(rel));
          if (!response || response.status !== 200) {
            errors.push(`${endpoint}${rel} -> ${response?.status ?? "unknown"}`);
            continue;
          }
          const payload = await response.json();
          if (!payload || typeof payload !== "object") {
            errors.push(`${endpoint}${rel} -> invalid json`);
            continue;
          }
          return payload;
        } catch (error) {
          errors.push(`${endpoint}${rel} -> ${String(error?.message || error || "request failed")}`);
        }
      }
    }

    const detail = errors.length > 0 ? ` (${errors.join("; ")})` : "";
    throw new Error(`metadata request failed${detail}`);
  }

  extractTrainedWords(civitai, metadata) {
    const trained = civitai?.trainedWords;
    if (Array.isArray(trained)) {
      const cleaned = trained.map((word) => String(word || "").trim()).filter(Boolean);
      if (cleaned.length > 0) return cleaned;
    }
    if (typeof trained === "string") {
      const parsed = trained.split(",").map((word) => String(word || "").trim()).filter(Boolean);
      if (parsed.length > 0) return parsed;
    }
    const metadataTags = metadata?.["modelspec.tags"];
    if (typeof metadataTags === "string") {
      return metadataTags
        .split(",")
        .map((word) => String(word || "").trim())
        .filter(Boolean)
        .slice(0, 120);
    }
    return [];
  }

  async fetchCivitaiInfo(hash) {
    const response = await fetch(`https://civitai.com/api/v1/model-versions/by-hash/${hash}`);
    if (response.status === 200) {
      return await response.json();
    }
    if (response.status === 404) {
      return null;
    }
    throw new Error(`civitai request failed (${response.status})`);
  }

  createLoadingContent(name) {
    return $el("div", { style: { minWidth: "460px", maxWidth: "840px" } }, [
      $el("h2", { textContent: String(name || "") }),
      $el("p", { textContent: "Loading LoRA info..." }),
    ]);
  }

  createErrorContent(name, message) {
    return $el("div", { style: { minWidth: "460px", maxWidth: "840px" } }, [
      $el("h2", { textContent: String(name || "") }),
      $el("p", { textContent: `Failed to load LoRA info: ${message}` }),
    ]);
  }

  createInfoContent(name, metadata, civitai) {
    const body = [];
    const hash = this.extractLoraHash(metadata);
    const descriptionRaw = extractCivitaiDescription(civitai);
    const descriptionHtml = sanitizeRichDescriptionHtml(descriptionRaw);
    const descriptionText = normalizeDescriptionText(descriptionHtml || descriptionRaw);
    const shouldShowDescriptionToggle = descriptionText.length > 420;
    const baseModel =
      metadata?.ss_base_model_version ||
      metadata?.["modelspec.base_model"] ||
      metadata?.["modelspec.architecture"] ||
      "";

    body.push(makeInfoRow("LoRA", name));
    body.push(makeInfoRow("SHA256", hash));
    body.push(makeInfoRow("Base Model", baseModel));

    if (civitai) {
      const modelName = civitai?.model?.name || "";
      const versionName = civitai?.name || "";
      const downloads = civitai?.stats?.downloadCount ?? "";
      const triggerWords = this.extractTrainedWords(civitai, metadata).join(", ");
      const pageUrl =
        civitai?.modelId && civitai?.id
          ? `https://civitai.com/models/${civitai.modelId}?modelVersionId=${civitai.id}`
          : "";

      body.push($el("hr"));
      body.push(makeInfoRow("Civitai Model", modelName));
      body.push(makeInfoRow("Version", versionName));
      body.push(makeInfoRow("Downloads", downloads));
      body.push(makeInfoRow("Trained Words", triggerWords));
      if (pageUrl) {
        body.push($el("p", [
          $el("label", { textContent: "Civitai Link: " }),
          $el("a", {
            href: pageUrl,
            textContent: pageUrl,
            target: "_blank",
            rel: "noopener noreferrer",
          }),
        ]));
      }

      const imageEntries = Array.isArray(civitai?.images) ? civitai.images : [];
      const thumbnailUrls = imageEntries
        .map((entry) => (entry && typeof entry === "object" ? entry : null))
        .filter(Boolean)
        .filter((entry) => {
          const type = String(entry.type || "").trim().toLowerCase();
          return type === "" || type === "image";
        })
        .map((entry) => normalizeHttpUrl(entry.url))
        .filter(Boolean)
        .slice(0, 8);
      if (thumbnailUrls.length > 0) {
        const primaryUrl = thumbnailUrls[0];
        body.push($el("a", {
          href: primaryUrl,
          target: "_blank",
          rel: "noopener noreferrer",
          style: {
            display: "block",
            marginTop: "10px",
          },
        }, [
          $el("img", {
            src: primaryUrl,
            style: {
              maxWidth: "100%",
              maxHeight: "420px",
              borderRadius: "8px",
              display: "block",
            },
          }),
        ]));
        if (thumbnailUrls.length > 1) {
          body.push($el("div", {
            style: {
              marginTop: "8px",
              display: "grid",
              gridTemplateColumns: "repeat(4, minmax(0, 1fr))",
              gap: "6px",
            },
          }, thumbnailUrls.map((url) => $el("a", {
            href: url,
            target: "_blank",
            rel: "noopener noreferrer",
            style: {
              display: "block",
              borderRadius: "6px",
              overflow: "hidden",
              border: "1px solid rgba(255,255,255,0.16)",
            },
          }, [
            $el("img", {
              src: url,
              style: {
                width: "100%",
                height: "72px",
                objectFit: "cover",
                display: "block",
              },
            }),
          ]))));
        }
      }
    }

    if (descriptionHtml) {
      let expanded = false;
      const descriptionBody = $el("div", {
        innerHTML: descriptionHtml,
        style: {
          marginTop: "8px",
          maxHeight: "220px",
          overflowY: "auto",
          paddingRight: "4px",
          lineHeight: "1.45",
          fontSize: "12px",
        },
      });
      const toggleBtn = shouldShowDescriptionToggle
        ? $el("button", {
          type: "button",
          textContent: "Show More",
          style: {
            marginTop: "8px",
            fontSize: "11px",
            textTransform: "uppercase",
            letterSpacing: "0.08em",
            borderRadius: "6px",
            border: "1px solid rgba(255,255,255,0.2)",
            background: "rgba(255,255,255,0.04)",
            color: "#ddd",
            padding: "4px 8px",
            cursor: "pointer",
          },
          onclick: () => {
            expanded = !expanded;
            descriptionBody.style.maxHeight = expanded ? "none" : "220px";
            descriptionBody.style.overflowY = expanded ? "visible" : "auto";
            toggleBtn.textContent = expanded ? "Show Less" : "Show More";
          },
        })
        : null;
      body.push($el("hr"));
      body.push($el("h3", { textContent: "Description" }));
      body.push(descriptionBody);
      if (toggleBtn) body.push(toggleBtn);
    } else {
      body.push($el("hr"));
      body.push(makeInfoRow("Description", "No description available for this LoRA."));
    }

    const fallbackRows = Object.keys(metadata || {})
      .sort((a, b) => a.localeCompare(b))
      .slice(0, 16)
      .map((key) => makeInfoRow(key, metadata[key]))
      .filter(Boolean);

    if (!civitai && fallbackRows.length > 0) {
      body.push($el("hr"));
      body.push($el("h3", { textContent: "Local Metadata" }));
      for (const row of fallbackRows) {
        body.push(row);
      }
    }

    return $el("div", { style: { minWidth: "460px", maxWidth: "840px" } }, [
      $el("h2", { textContent: String(name || "") }),
      ...body.filter(Boolean),
    ]);
  }

  async showFor(name) {
    const loraName = String(name || "").trim();
    if (!loraName) return;
    this.name = loraName;
    super.show(this.createLoadingContent(loraName));

    try {
      const metadata = await this.fetchLocalMetadata(loraName);
      const hash = this.extractLoraHash(metadata);
      let civitai = null;
      if (hash) {
        try {
          civitai = await this.fetchCivitaiInfo(hash);
        } catch {
          civitai = null;
        }
      }
      this.content = this.createInfoContent(loraName, metadata, civitai);
      super.show(this.content);
    } catch (error) {
      const message = String(error?.message || error || "unknown error");
      this.content = this.createErrorContent(loraName, message);
      super.show(this.content);
    }
  }
}

function appendSelectedLoraToSyntaxField(node, selectedValue) {
  const syntaxWidget = findWidget(node, "lora_syntax_text");
  if (!syntaxWidget) return;

  const tag = buildTag(node, selectedValue);
  if (!tag) return;

  const current = String(syntaxWidget.value ?? "").trim();
  if (current.includes(tag)) return;
  syntaxWidget.value = current ? `${current}\n${tag}` : tag;
  if (typeof syntaxWidget.callback === "function") {
    try {
      syntaxWidget.callback(syntaxWidget.value, app, node);
    } catch {
      // Keep UX responsive even if a callback throws.
    }
  }
  if (typeof app?.graph?.setDirtyCanvas === "function") {
    app.graph.setDirtyCanvas(true, true);
  }
  if (typeof app?.canvas?.setDirty === "function") {
    app.canvas.setDirty(true, true);
  }
}

async function openLoraInfo(node) {
  const selectedValue = getSelectedLoraValue(node);
  if (!selectedValue || selectedValue === NONE_OPTION) return;
  const dialog = new UmbraLoraInfoDialog(selectedValue);
  await dialog.showFor(selectedValue);
}

function wireLoraDropdownCallback(node) {
  const loraWidget = findWidget(node, "lora_name") || findWidget(node, "selected_lora");
  if (!loraWidget || loraWidget.umbraLoraSyntaxPatched) return;
  loraWidget.umbraLoraSyntaxPatched = true;
  chainCallback(loraWidget, "callback", function onLoraSelected(value) {
    appendSelectedLoraToSyntaxField(node, value);
  });
}

function ensureControls(node) {
  if (node.__umbraLoraControlsReady) return;
  node.__umbraLoraControlsReady = true;
  const infoButton = node.addWidget("button", "View Lora Info", null, () => {
    openLoraInfo(node);
  });
  infoButton.umbraLoraControl = true;
}

app.registerExtension({
  name: "umbra.a1111.lora.syntax",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (!nodeType || !nodeData) return;
    const className = String(nodeData?.name || "");
    if (className !== NODE_NAME) return;

    chainCallback(nodeType.prototype, "onWidgetChanged", function onWidgetChanged(name, value) {
      const widgetName = String(name || "");
      if (widgetName !== "lora_name" && widgetName !== "selected_lora") return;
      appendSelectedLoraToSyntaxField(this, value);
    });

    chainCallback(nodeType.prototype, "onNodeCreated", function onNodeCreated() {
      wireLoraDropdownCallback(this);
      ensureControls(this);
    });

    chainCallback(nodeType.prototype, "onConfigure", function onConfigure() {
      wireLoraDropdownCallback(this);
      ensureControls(this);
    });
  },
});
