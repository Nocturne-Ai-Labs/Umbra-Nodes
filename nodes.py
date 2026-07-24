"""
Umbra Lab Save Image Node
Saves images with standard metadata formats:
- ComfyUI standard (prompt/workflow)  
- A1111 compatible parameters

This is fully compatible with Umbra Lab's metadata scanner.
"""

import os
import sys
import re
import json
import time
import random
import shutil
import functools
import inspect
import hashlib
import threading
import numpy as np
from datetime import datetime
from PIL import Image
from PIL.PngImagePlugin import PngInfo
import folder_paths
import torch
import comfy.sd
import comfy.utils as comfy_utils
import comfy.model_management as model_management

from .detailer_sampling import (
    DETAILER_SAMPLING_PROVIDER_TYPE,
    UmbraFlux2DetailerProviderNode,
    UmbraHiDreamO1DetailerProviderNode,
    UmbraIdeogram4DetailerProviderNode,
    UmbraOmniGen2DetailerProviderNode,
    dispatch_detailer_stage,
    run_native_detailer,
)


def _install_hidream_o1_gqa_compat():
    """Forward HiDream-O1's GQA flag until ComfyUI's two-pass wrapper does so itself."""
    try:
        import comfy.ops
        import comfy.ldm.hidream_o1.attention as attention_module
        import comfy.ldm.hidream_o1.model as model_module
        from comfy.ldm.modules.attention import optimized_attention
    except Exception:
        return

    original_factory = attention_module.make_two_pass_attention
    try:
        if "enable_gqa" in inspect.getsource(original_factory):
            return
    except Exception:
        pass

    def make_two_pass_attention(ar_len, transformer_options=None):
        def two_pass_attention(q, k, v, heads, **kwargs):
            batch, head_count, token_count, head_dim = q.shape
            gqa_kwargs = {"enable_gqa": True} if kwargs.get("enable_gqa") else {}

            if token_count < k.shape[2]:
                out = optimized_attention(
                    q, k, v, heads,
                    mask=None,
                    skip_reshape=True,
                    skip_output_reshape=True,
                    transformer_options=transformer_options,
                    **gqa_kwargs,
                )
            elif ar_len >= token_count:
                out = comfy.ops.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=None,
                    dropout_p=0.0,
                    is_causal=True,
                    **gqa_kwargs,
                )
            elif ar_len <= 0:
                out = optimized_attention(
                    q, k, v, heads,
                    mask=None,
                    skip_reshape=True,
                    skip_output_reshape=True,
                    transformer_options=transformer_options,
                    **gqa_kwargs,
                )
            else:
                out_ar = comfy.ops.scaled_dot_product_attention(
                    q[:, :, :ar_len], k[:, :, :ar_len], v[:, :, :ar_len],
                    attn_mask=None,
                    dropout_p=0.0,
                    is_causal=True,
                    **gqa_kwargs,
                )
                out_gen = optimized_attention(
                    q[:, :, ar_len:], k, v, heads,
                    mask=None,
                    skip_reshape=True,
                    skip_output_reshape=True,
                    transformer_options=transformer_options,
                    **gqa_kwargs,
                )
                out = torch.cat([out_ar, out_gen], dim=2)

            return out.transpose(1, 2).reshape(batch, token_count, head_count * head_dim)

        return two_pass_attention

    make_two_pass_attention._umbra_gqa_compat = True
    attention_module.make_two_pass_attention = make_two_pass_attention
    model_module.make_two_pass_attention = make_two_pass_attention


_install_hidream_o1_gqa_compat()


def _resolve_output_base_dir(output_folder=None):
    override = str(output_folder or "").strip()
    if not override:
        override = str(os.getenv("UMBRA_EXTERNAL_OUTPUT_DIR") or "").strip()

    if not override:
        return folder_paths.get_output_directory()

    override = os.path.expanduser(os.path.expandvars(override))
    if not os.path.isabs(override):
        override = os.path.join(folder_paths.get_output_directory(), override)
    return os.path.normpath(override)


def _normalize_output_subfolder(raw_subfolder):
    text = str(raw_subfolder or "").strip().replace("\\", "/")
    if not text:
        return ""

    safe_parts = []
    for part in text.split("/"):
        part = str(part or "").strip()
        if not part or part in (".", ".."):
            continue
        part = re.sub(r'[<>:"|?*]', "", part).rstrip(". ")
        if part:
            safe_parts.append(part)

    if not safe_parts:
        return ""
    return os.path.join(*safe_parts)

def get_umbra_output_dir(
    output_folder=None,
    save_to_date_folder=False,
    save_to_set_subfolder=False,
    set_subfolder="",
):
    """
    Get Umbra image output directory.
    Saves to: <base>/[yyyy-mm-dd]/[set-subfolder]
    Base resolution order:
      1) node input output_folder
      2) UMBRA_EXTERNAL_OUTPUT_DIR environment variable
      3) ComfyUI output directory
    """
    comfy_output = _resolve_output_base_dir(output_folder)
    output_path = comfy_output
    if _to_bool(save_to_date_folder, default=False):
        output_path = os.path.join(output_path, datetime.now().strftime("%Y-%m-%d"))
    if _to_bool(save_to_set_subfolder, default=False):
        normalized_set_subfolder = _normalize_output_subfolder(set_subfolder)
        if normalized_set_subfolder:
            output_path = os.path.join(output_path, normalized_set_subfolder)
    os.makedirs(output_path, exist_ok=True)
    return output_path


def _resolve_filename_prefix(filename_prefix, default_prefix):
    def apply_tokens(value):
        text = str(value or "")
        replacements = {
            "%date%": now.strftime("%Y-%m-%d"),
            "%time%": now.strftime("%H-%M-%S"),
            "%datetime%": now.strftime("%Y-%m-%d_%H-%M-%S"),
        }
        for token, token_value in replacements.items():
            text = text.replace(token, token_value)
        return text

    now = datetime.now()
    fallback = apply_tokens(default_prefix or "UmbraLab_%date%").strip() or "UmbraLab"
    raw_prefix = str(filename_prefix or "").strip()
    prefix = apply_tokens(raw_prefix) if raw_prefix else fallback

    # ComfyUI's get_save_image_path treats a leading slash as absolute path and blocks it.
    # Keep user subfolder support, but force prefixes to remain inside output_dir.
    prefix = prefix.replace("\\", "/")
    prefix = re.sub(r"^[a-zA-Z]:", "", prefix)  # Drop Windows drive prefix.
    prefix = prefix.lstrip("/")
    safe_parts = []
    for part in prefix.split("/"):
        part = part.strip()
        if not part or part in (".", ".."):
            continue
        safe_parts.append(part)

    sanitized = "/".join(safe_parts).strip()
    return sanitized or fallback


def create_temp_preview_copy(source_path, prefix="umbra_preview"):
    """
    Copy a saved media file to ComfyUI temp dir and return preview tuple
    (filename, subfolder, type) for UI previews.
    """
    source_path = str(source_path or "")
    base_name = os.path.basename(source_path) if source_path else ""
    if not source_path or not os.path.exists(source_path):
        return base_name, "", "output"

    temp_dir = folder_paths.get_temp_directory()
    os.makedirs(temp_dir, exist_ok=True)

    stem, ext = os.path.splitext(base_name)
    safe_stem = re.sub(r"[^a-zA-Z0-9._-]", "_", stem).strip("._") or prefix
    candidate = f"{safe_stem}_{prefix}{ext}"
    preview_path = os.path.join(temp_dir, candidate)
    index = 1
    while os.path.exists(preview_path):
        candidate = f"{safe_stem}_{prefix}_{index}{ext}"
        preview_path = os.path.join(temp_dir, candidate)
        index += 1

    try:
        shutil.copy2(source_path, preview_path)
        return os.path.basename(preview_path), "", "temp"
    except Exception:
        # Return a direct output preview path fallback rather than a null-ish payload.
        return base_name, "", "output"


BIGMAX = (2**53 - 1)
SEED_MAX = 0xffffffffffffffff
FRONTEND_SAFE_SEED_MAX = 9007199254740991

_POWER_PROMPTER_ASPECT_RATIO_PRESETS = {
    "SD1.5 - 1:1 square 512x512": (512, 512),
    "SD1.5 - 2:3 portrait 512x768": (512, 768),
    "SD1.5 - 3:4 portrait 512x682": (512, 682),
    "SD1.5 - 3:2 landscape 768x512": (768, 512),
    "SD1.5 - 4:3 landscape 682x512": (682, 512),
    "SD1.5 - 16:9 cinema 910x512": (910, 512),
    "SD1.5 - 1.85:1 cinema 952x512": (952, 512),
    "SD1.5 - 2:1 cinema 1024x512": (1024, 512),
    "SD1.5 - 2.39:1 anamorphic 1224x512": (1224, 512),
    "SDXL - 1:1 square 1024x1024": (1024, 1024),
    "SDXL - 3:4 portrait 896x1152": (896, 1152),
    "SDXL - 5:8 portrait 832x1216": (832, 1216),
    "SDXL - 9:16 portrait 768x1344": (768, 1344),
    "SDXL - 9:21 portrait 640x1536": (640, 1536),
    "SDXL - 4:3 landscape 1152x896": (1152, 896),
    "SDXL - 3:2 landscape 1216x832": (1216, 832),
    "SDXL - 16:9 landscape 1344x768": (1344, 768),
    "SDXL - 21:9 landscape 1536x640": (1536, 640),
    "1536 - 1:1 square 1536x1536": (1536, 1536),
    "1536 - 2:3 portrait 1024x1536": (1024, 1536),
    "1536 - 3:4 portrait 1152x1536": (1152, 1536),
    "1536 - 5:8 portrait 960x1536": (960, 1536),
    "1536 - 9:16 portrait 864x1536": (864, 1536),
    "1536 - 9:21 portrait 656x1536": (656, 1536),
    "1536 - 3:2 landscape 1536x1024": (1536, 1024),
    "1536 - 4:3 landscape 1536x1152": (1536, 1152),
    "1536 - 8:5 landscape 1536x960": (1536, 960),
    "1536 - 16:9 landscape 1536x864": (1536, 864),
    "1536 - 1.85:1 cinema 1536x832": (1536, 832),
    "1536 - 2:1 cinema 1536x768": (1536, 768),
    "1536 - 2.39:1 anamorphic 1536x640": (1536, 640),
    "1536 - 21:9 landscape 1536x656": (1536, 656),
}
_POWER_PROMPTER_ASPECT_RATIO_OPTIONS = ["custom", *_POWER_PROMPTER_ASPECT_RATIO_PRESETS.keys()]
_POWER_PROMPTER_SEED_COUNTERS = {}
_POWER_PROMPTER_SEED_COUNTERS_LIMIT = 1024


def _normalize_seed(seed):
    try:
        seed_value = int(seed)
    except Exception:
        seed_value = 0
    return max(0, min(SEED_MAX, seed_value))


def _value_for_index(value, index, default=None):
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return default
        if index < len(value):
            return value[index]
        return value[-1]
    if value is None:
        return default
    return value


def _to_int(value, default=0, minimum=None, maximum=None):
    try:
        result = int(value)
    except Exception:
        result = int(default)
    if minimum is not None:
        result = max(int(minimum), result)
    if maximum is not None:
        result = min(int(maximum), result)
    return result


def _to_float(value, default=0.0, minimum=None, maximum=None):
    try:
        result = float(value)
    except Exception:
        result = float(default)
    if minimum is not None:
        result = max(float(minimum), result)
    if maximum is not None:
        result = min(float(maximum), result)
    return result


def _to_bool(value, default=False):
    if isinstance(value, bool):
        return value
    if value is None:
        return bool(default)
    if isinstance(value, (int, float)):
        return value != 0
    text = str(value).strip().lower()
    if text in ("true", "1", "yes", "y", "on"):
        return True
    if text in ("false", "0", "no", "n", "off", ""):
        return False
    return bool(default)


_A1111_LORA_TAG_RE = re.compile(r"<lora:([^:>]+)(?::([^:>]*))?(?::([^:>]*))?>", re.IGNORECASE)
_LORA_STATE_CACHE = {}
_FILE_SHA256_CACHE = {}
_LORA_NONE_OPTION = "[None]"


def _parse_strength(value, default):
    if value is None:
        return float(default)
    text = str(value).strip()
    if text == "":
        return float(default)
    try:
        return float(text)
    except Exception:
        return float(default)


def _parse_a1111_lora_tags(text):
    parsed = []
    source = str(text or "")
    for match in _A1111_LORA_TAG_RE.finditer(source):
        name = str(match.group(1) or "").strip()
        if not name:
            continue
        strength_model = _parse_strength(match.group(2), 1.0)
        strength_clip = _parse_strength(match.group(3), strength_model)
        parsed.append(
            {
                "raw": match.group(0),
                "name": name,
                "strength_model": strength_model,
                "strength_clip": strength_clip,
            }
        )
    return parsed


def _strip_a1111_lora_tags(text):
    source = str(text or "")
    stripped = _A1111_LORA_TAG_RE.sub(" ", source)
    stripped = re.sub(r"\s+,", ",", stripped)
    stripped = re.sub(r",\s+", ", ", stripped)
    stripped = re.sub(r"\s{2,}", " ", stripped)
    return stripped.strip(" ,")


def _display_resource_name(filename):
    normalized = str(filename or "").replace("\\", "/").strip()
    if not normalized:
        return ""
    return os.path.splitext(os.path.basename(normalized))[0].strip()


def _resolve_model_resource_path(model_name):
    name = str(model_name or "").strip().replace("\\", "/")
    if not name:
        return None, ""
    for folder_name in ("checkpoints", "diffusion_models", "unet"):
        try:
            path = folder_paths.get_full_path(folder_name, name)
        except Exception:
            path = None
        if path:
            return path, folder_name
    return None, ""


def _sha256_for_file(path):
    if not path:
        return ""
    try:
        stat = os.stat(path)
        cache_key = (os.path.abspath(path), int(stat.st_mtime), int(stat.st_size))
    except Exception:
        return ""
    cached = _FILE_SHA256_CACHE.get(cache_key)
    if cached:
        return cached
    digest = hashlib.sha256()
    try:
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                if not chunk:
                    break
                digest.update(chunk)
    except Exception:
        return ""
    value = digest.hexdigest()
    _FILE_SHA256_CACHE[cache_key] = value
    return value


def _build_civitai_resource_entry(resource_type, name, filename="", strength=None, path=None):
    display_name = _display_resource_name(name or filename)
    if not display_name:
        return None
    entry = {
        "type": str(resource_type or "").strip() or "model",
        "name": display_name,
        "modelName": display_name,
    }
    version_name = _display_resource_name(filename)
    if version_name:
        entry["modelVersionName"] = version_name
        entry["filename"] = str(filename or "").replace("\\", "/")
    if strength is not None:
        entry["weight"] = _to_float(strength, default=1.0)
    file_hash = _sha256_for_file(path)
    if file_hash:
        entry["hash"] = file_hash
        entry["hashes"] = {"SHA256": file_hash}
    return entry


def _build_civitai_metadata_payload(positive_prompt, model_name):
    resources = []
    seen = set()

    model_value = str(model_name or "").strip()
    if model_value:
        model_path, model_folder = _resolve_model_resource_path(model_value)
        model_type = "checkpoint" if model_folder == "checkpoints" else "model"
        model_entry = _build_civitai_resource_entry(model_type, model_value, model_value, path=model_path)
        if model_entry:
            resources.append(model_entry)
            seen.add((model_entry.get("type", ""), str(model_entry.get("name", "")).lower()))

    for tag in _parse_a1111_lora_tags(positive_prompt):
        requested_name = str(tag.get("name") or "").strip()
        if not requested_name:
            continue
        resolved_filename, lora_path = _resolve_lora_path(requested_name)
        lora_name = resolved_filename or requested_name
        entry = _build_civitai_resource_entry(
            "lora",
            requested_name,
            lora_name,
            strength=tag.get("strength_model"),
            path=lora_path,
        )
        if not entry:
            continue
        key = (entry.get("type", ""), str(entry.get("name", "")).lower())
        if key in seen:
            continue
        seen.add(key)
        resources.append(entry)

    return {
        "resources": resources,
    }


def _append_civitai_parameters(a1111_params, civitai_payload):
    resources = civitai_payload.get("resources") if isinstance(civitai_payload, dict) else None
    if not resources:
        return a1111_params
    return f"{a1111_params}, Civitai resources: {_safe_json_dumps(resources)}"


def _normalize_lora_name_for_tag(lora_filename):
    normalized = str(lora_filename or "").replace("\\", "/").strip()
    if not normalized:
        return ""
    no_ext = os.path.splitext(normalized)[0]
    return no_ext


def _format_lora_tag(name, strength_model=1.0, strength_clip=None):
    normalized_name = str(name or "").strip()
    if not normalized_name:
        return ""
    if strength_clip is None:
        return f"<lora:{normalized_name}:{float(strength_model):g}>"
    return f"<lora:{normalized_name}:{float(strength_model):g}:{float(strength_clip):g}>"


def _get_lora_dropdown_options():
    try:
        loras = list(folder_paths.get_filename_list("loras") or [])
    except Exception:
        loras = []
    normalized = sorted({str(item).replace("\\", "/") for item in loras}, key=lambda value: value.lower())
    return [_LORA_NONE_OPTION] + normalized


def _build_all_lora_syntax_catalog():
    lines = []
    for lora_filename in _get_lora_dropdown_options()[1:]:
        tag_name = _normalize_lora_name_for_tag(lora_filename)
        if not tag_name:
            continue
        lines.append(_format_lora_tag(tag_name, 1.0, 1.0))
    return "\n".join(lines)


def _resolve_lora_filename(requested_name):
    requested = str(requested_name or "").strip().replace("\\", "/")
    if not requested:
        return None

    available = list(folder_paths.get_filename_list("loras") or [])
    if not available:
        return None

    by_lower = {str(item).replace("\\", "/").lower(): item for item in available}
    requested_lower = requested.lower()
    requested_base = os.path.splitext(os.path.basename(requested_lower))[0]

    candidates = [requested_lower]
    if "." not in os.path.basename(requested_lower):
        for ext in (".safetensors", ".ckpt", ".pt", ".bin", ".pth"):
            candidates.append(f"{requested_lower}{ext}")

    for candidate in candidates:
        exact = by_lower.get(candidate)
        if exact is not None:
            return exact

    for original in available:
        normalized = str(original).replace("\\", "/")
        normalized_base = os.path.splitext(os.path.basename(normalized.lower()))[0]
        if normalized_base == requested_base:
            return original

    return None


def _resolve_lora_path(requested_name):
    filename = _resolve_lora_filename(requested_name)
    if not filename:
        return None, None
    path = folder_paths.get_full_path("loras", filename)
    if not path:
        return filename, None
    return filename, path


def _load_lora_state(lora_filename):
    lora_path = folder_paths.get_full_path("loras", lora_filename)
    if not lora_path:
        raise Exception(f"Could not resolve LoRA path for '{lora_filename}'.")

    try:
        mtime = os.path.getmtime(lora_path)
    except Exception:
        mtime = 0.0

    cache_key = (lora_path, mtime)
    cached = _LORA_STATE_CACHE.get(cache_key)
    if cached is not None:
        return cached

    lora_state = comfy_utils.load_torch_file(lora_path, safe_load=True)
    _LORA_STATE_CACHE[cache_key] = lora_state
    if len(_LORA_STATE_CACHE) > 128:
        # Keep memory bounded across long ComfyUI sessions.
        _LORA_STATE_CACHE.clear()
        _LORA_STATE_CACHE[cache_key] = lora_state
    return lora_state


def _find_existing_lora_preview_path(lora_path):
    if not lora_path:
        return ""
    stem, _ = os.path.splitext(lora_path)
    for ext in (".png", ".jpg", ".jpeg", ".webp"):
        candidate = f"{stem}{ext}"
        if os.path.exists(candidate):
            return candidate
    return ""


def _save_lora_preview_image(lora_path, preview_image, overwrite=True):
    if not lora_path or preview_image is None:
        return ""

    tensor = preview_image
    if isinstance(tensor, (list, tuple)):
        tensor = tensor[0] if tensor else None
    if tensor is None:
        return ""

    if isinstance(tensor, torch.Tensor) and tensor.ndim == 4:
        tensor = tensor[0]
    if not isinstance(tensor, torch.Tensor) or tensor.ndim != 3:
        return ""

    stem, _ = os.path.splitext(lora_path)
    preview_path = f"{stem}.png"
    if os.path.exists(preview_path) and not _to_bool(overwrite, default=True):
        return preview_path

    image_data = 255.0 * tensor.cpu().numpy()
    img = Image.fromarray(np.clip(image_data, 0, 255).astype(np.uint8))
    os.makedirs(os.path.dirname(preview_path), exist_ok=True)
    img.save(preview_path, compress_level=4)
    return preview_path


_PROMPT_FIELD_NAME_RE = re.compile(r"^field_(\d+)$", re.IGNORECASE)


def _normalize_prompt_segment(
    text,
    normalize_whitespace=True,
    remove_all_whitespace=False,
    underscore_mode="keep",
):
    value = str(text or "")
    if _to_bool(normalize_whitespace, default=True):
        value = re.sub(r"\s+", " ", value).strip()
    if _to_bool(remove_all_whitespace, default=False):
        value = re.sub(r"\s+", "", value)

    mode = str(underscore_mode or "keep").strip().lower()
    if mode == "spaces_to_underscores":
        value = value.replace(" ", "_")
    elif mode == "underscores_to_spaces":
        value = value.replace("_", " ")
        if _to_bool(normalize_whitespace, default=True):
            value = re.sub(r"\s+", " ", value).strip()
    elif mode == "remove_underscores":
        value = value.replace("_", "")
        if _to_bool(normalize_whitespace, default=True):
            value = re.sub(r"\s+", " ", value).strip()
    return value


def _discover_umbra_project_root(start_file):
    current = os.path.dirname(os.path.abspath(start_file))
    visited = []
    for _ in range(12):
        visited.append(current)
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent

    # Prefer roots that already contain the expected prompts directory.
    for base in visited:
        prompts_dir = os.path.join(base, "User", "PowerPrompter", "Prompts")
        if os.path.isdir(prompts_dir):
            return base

    # Fallback to any ancestor that has a User directory.
    for base in visited:
        if os.path.isdir(os.path.join(base, "User")):
            return base

    # Last resort: keep previous behavior.
    return os.path.dirname(os.path.dirname(os.path.abspath(start_file)))


def _get_powerprompter_prompts_dir():
    base_path = _discover_umbra_project_root(__file__)
    return os.path.join(base_path, "User", "PowerPrompter", "Prompts")


def _sanitize_relative_folder_path(value):
    raw = str(value or "").replace("\\", "/").strip()
    safe_parts = []
    for part in raw.split("/"):
        part = part.strip()
        if not part or part in (".", ".."):
            continue
        safe_parts.append(part)
    return "/".join(safe_parts)


def _sanitize_powerprompter_filename(filename):
    name = os.path.basename(str(filename or "").strip())
    if not name:
        return ""
    if not name.lower().endswith(".txt"):
        name += ".txt"
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name).strip(" .")
    return name


def _parse_selected_fields(value, valid_fields):
    valid_set = set(valid_fields or [])
    parsed = []
    text = str(value or "").strip()
    if not text:
        return []

    try:
        loaded = json.loads(text)
        if isinstance(loaded, list):
            for item in loaded:
                name = str(item or "").strip()
                if name in valid_set and name not in parsed:
                    parsed.append(name)
            return parsed
    except Exception:
        pass

    for part in text.split(","):
        name = str(part or "").strip()
        if name in valid_set and name not in parsed:
            parsed.append(name)
    return parsed


def _extract_model_name_from_model(model):
    if model is None:
        return ""

    # Handle common Comfy model wrappers first.
    for attr_path in (
        ("model_options", "ckpt_name"),
        ("model", "model_config", "name"),
        ("model_config", "name"),
    ):
        current = model
        ok = True
        for attr in attr_path:
            if isinstance(current, dict):
                current = current.get(attr)
            else:
                current = getattr(current, attr, None)
            if current is None:
                ok = False
                break
        if ok and isinstance(current, str) and current.strip():
            return current.strip()

    # Avoid leaking internal wrapper class names (e.g. "ModelPatcher") as model IDs.
    return ""


def _get_core_common_ksampler():
    try:
        import nodes as comfy_nodes
        return getattr(comfy_nodes, "common_ksampler", None)
    except Exception:
        return None


def _get_sampler_names():
    fallback = [
        "euler",
        "euler_ancestral",
        "heun",
        "dpm_2",
        "dpm_2_ancestral",
        "lms",
        "dpm_fast",
        "dpm_adaptive",
        "dpmpp_2s_ancestral",
        "dpmpp_sde",
        "dpmpp_2m",
        "dpmpp_2m_sde",
        "ddim",
        "uni_pc",
    ]
    try:
        import comfy.samplers

        samplers = list(getattr(comfy.samplers.KSampler, "SAMPLERS", []) or [])
        if samplers:
            return samplers
    except Exception:
        pass
    return fallback


def _get_scheduler_names():
    fallback = ["normal", "karras", "exponential", "sgm_uniform", "simple", "ddim_uniform"]
    try:
        import comfy.samplers

        schedulers = list(getattr(comfy.samplers.KSampler, "SCHEDULERS", []) or [])
        if schedulers:
            return schedulers
    except Exception:
        pass
    return fallback


_REPEAT_COUNTERS = {
    "UmbraSeedValue": {},
    "UmbraKSampler": {},
}


def _stable_fingerprint(value):
    try:
        return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
    except Exception:
        return repr(value)


def _build_repeat_key(namespace, unique_id, prompt_dict, config):
    node_inputs = {}
    node_data = _get_prompt_node(prompt_dict, unique_id)
    if isinstance(node_data, dict):
        inputs = node_data.get("inputs", {})
        if isinstance(inputs, dict):
            node_inputs = inputs
    return "|".join(
        [
            str(namespace or ""),
            str(unique_id or ""),
            _stable_fingerprint(node_inputs),
            _stable_fingerprint(config),
        ]
    )


def _next_repeat_index(namespace, repeat_key):
    bucket = _REPEAT_COUNTERS.setdefault(namespace, {})
    index = int(bucket.get(repeat_key, 0))
    bucket[repeat_key] = index + 1
    if len(bucket) > 4096:
        # Keep memory bounded on long-running Comfy sessions.
        bucket.clear()
        bucket[repeat_key] = index + 1
    return index


def _get_connected_input_info(prompt_dict, unique_id, input_name):
    node_data = _get_prompt_node(prompt_dict, unique_id)
    if not isinstance(node_data, dict):
        return None, None, None
    inputs = node_data.get("inputs", {})
    if not isinstance(inputs, dict):
        return None, None, None
    input_ref = inputs.get(input_name)
    if not isinstance(input_ref, (list, tuple)) or len(input_ref) < 2:
        return None, None, None
    source_id = input_ref[0]
    source_output_index = _to_int(input_ref[1], default=0, minimum=0)
    source_node = _get_prompt_node(prompt_dict, source_id)
    source_class = source_node.get("class_type") if isinstance(source_node, dict) else None
    return source_class, source_output_index, source_id


class _UmbraLogger:
    def info(self, message):
        print(f"[Umbra-Nodes] {message}")

    def warn(self, message):
        print(f"[Umbra-Nodes][WARN] {message}")

    def error(self, message):
        print(f"[Umbra-Nodes][ERROR] {message}", file=sys.stderr)


logger = _UmbraLogger()


class ContainsAll(dict):
    def __contains__(self, _):
        return True

    def __getitem__(self, key):
        return super().get(key, (None, {}))


def _extract_prompts_from_workflow(prompt_dict):
    positive = ""
    negative = ""
    if not prompt_dict:
        return positive, negative

    negative_keywords = [
        "negative", "bad", "worst", "blurry", "deformed", "ugly", "lowres",
        "low quality", "artifact", "jpeg artifacts", "watermark",
    ]
    clip_texts = []

    for _, node_data in prompt_dict.items():
        class_type = node_data.get("class_type", "")
        inputs = node_data.get("inputs", {})
        if class_type != "CLIPTextEncode":
            continue
        text = inputs.get("text", "")
        if not isinstance(text, str) or len(text.strip()) == 0:
            continue
        text = text.strip()
        clip_texts.append(text)
        lowered = text.lower()
        is_negative = any(kw in lowered for kw in negative_keywords)
        if is_negative and not negative:
            negative = text
        elif not is_negative and not positive:
            positive = text

    if not positive and clip_texts:
        positive = clip_texts[0]
    if not negative and len(clip_texts) > 1:
        for text in clip_texts:
            if text != positive:
                negative = text
                break

    return positive, negative


def _get_prompt_node(prompt_dict, node_id):
    if not prompt_dict:
        return None
    if node_id in prompt_dict:
        return prompt_dict[node_id]
    node_id_str = str(node_id)
    if node_id_str in prompt_dict:
        return prompt_dict[node_id_str]
    return None


def _extract_direct_model_name_from_inputs(inputs):
    if not isinstance(inputs, dict):
        return ""
    for key in (
        "ckpt_name",
        "checkpoint_name",
        "model_name",
        "unet_name",
        "diffusion_model",
        "filename",
        "file",
    ):
        value = inputs.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _extract_model_name_from_prompt_reference(prompt_dict, node_ref, visited=None, depth=0):
    if not prompt_dict or not isinstance(node_ref, (list, tuple)) or len(node_ref) == 0:
        return ""
    if depth > 50:
        return ""

    node_id = str(node_ref[0])
    if visited is None:
        visited = set()
    if node_id in visited:
        return ""
    visited.add(node_id)

    node_data = _get_prompt_node(prompt_dict, node_id)
    if not isinstance(node_data, dict):
        return ""

    inputs = node_data.get("inputs", {})
    direct_name = _extract_direct_model_name_from_inputs(inputs)
    if direct_name:
        return direct_name

    preferred_links = (
        "model",
        "base_model",
        "model_a",
        "model_b",
        "model1",
        "model2",
        "unet",
        "ckpt",
        "checkpoint",
        "diffusion_model",
    )
    for key in preferred_links:
        value = inputs.get(key)
        if isinstance(value, (list, tuple)) and len(value) > 0 and _get_prompt_node(prompt_dict, value[0]) is not None:
            resolved = _extract_model_name_from_prompt_reference(prompt_dict, value, visited, depth + 1)
            if resolved:
                return resolved

    for value in inputs.values():
        if isinstance(value, (list, tuple)) and len(value) > 0 and _get_prompt_node(prompt_dict, value[0]) is not None:
            resolved = _extract_model_name_from_prompt_reference(prompt_dict, value, visited, depth + 1)
            if resolved:
                return resolved

    return ""


def _extract_model_name_from_node_connection(prompt_dict, unique_id, input_name="model"):
    if not prompt_dict or unique_id is None:
        return ""
    node_data = _get_prompt_node(prompt_dict, unique_id)
    if not isinstance(node_data, dict):
        return ""
    node_inputs = node_data.get("inputs", {})
    ref = node_inputs.get(input_name)
    if not isinstance(ref, (list, tuple)) or len(ref) == 0:
        return ""
    return _extract_model_name_from_prompt_reference(prompt_dict, ref)


def _extract_first_model_name_from_workflow(prompt_dict):
    if not prompt_dict:
        return ""
    for _, node_data in prompt_dict.items():
        if not isinstance(node_data, dict):
            continue
        name = _extract_direct_model_name_from_inputs(node_data.get("inputs", {}))
        if name:
            return name
    return ""


def _extract_text_from_prompt_reference(prompt_dict, node_ref, visited=None, depth=0):
    if not prompt_dict or not isinstance(node_ref, (list, tuple)) or len(node_ref) == 0:
        return ""
    if depth > 40:
        return ""

    node_id = str(node_ref[0])
    if visited is None:
        visited = set()
    if node_id in visited:
        return ""
    visited.add(node_id)

    node_data = _get_prompt_node(prompt_dict, node_id)
    if not isinstance(node_data, dict):
        return ""

    class_type = node_data.get("class_type", "")
    inputs = node_data.get("inputs", {})
    if class_type == "CLIPTextEncode":
        text = inputs.get("text", "")
        if isinstance(text, str):
            return text.strip()
        return ""

    for value in inputs.values():
        if isinstance(value, (list, tuple)) and len(value) > 0 and _get_prompt_node(prompt_dict, value[0]) is not None:
            text = _extract_text_from_prompt_reference(prompt_dict, value, visited, depth + 1)
            if text:
                return text
    return ""


def _extract_prompt_from_node_connection(prompt_dict, unique_id, input_name):
    if not prompt_dict or unique_id is None:
        return ""
    node_data = _get_prompt_node(prompt_dict, unique_id)
    if not isinstance(node_data, dict):
        return ""
    node_inputs = node_data.get("inputs", {})
    ref = node_inputs.get(input_name)
    if not isinstance(ref, (list, tuple)) or len(ref) == 0:
        return ""
    return _extract_text_from_prompt_reference(prompt_dict, ref)


def _resolve_prompt_text(explicit_text, conditioning_input, prompt_dict, prompt_type, unique_id=None):
    if isinstance(explicit_text, str) and explicit_text.strip():
        return explicit_text.strip()
    if isinstance(conditioning_input, str) and conditioning_input.strip():
        return conditioning_input.strip()

    if conditioning_input is not None and prompt_dict:
        connected_text = _extract_prompt_from_node_connection(prompt_dict, unique_id, prompt_type)
        if connected_text:
            return connected_text

    wf_positive, wf_negative = _extract_prompts_from_workflow(prompt_dict)
    return wf_positive if prompt_type == "positive" else wf_negative


def _safe_json_dumps(value):
    try:
        return json.dumps(value, ensure_ascii=False)
    except TypeError:
        return json.dumps(value, ensure_ascii=False, default=str)


def _normalize_png_json_payload(payload):
    if isinstance(payload, str):
        raw = payload.strip()
        if raw:
            try:
                return json.loads(raw)
            except Exception:
                return payload
    return payload


def _looks_like_comfy_api_prompt_graph(payload):
    if not isinstance(payload, dict) or not payload:
        return False
    inspected = 0
    matching = 0
    for value in payload.values():
        if not isinstance(value, dict):
            continue
        inspected += 1
        if "class_type" in value and isinstance(value.get("inputs"), dict):
            matching += 1
        if inspected >= 8:
            break
    return inspected > 0 and matching == inspected


def _looks_like_comfy_ui_workflow(payload):
    if not isinstance(payload, dict):
        return False
    nodes = payload.get("nodes")
    if isinstance(nodes, list):
        return True
    return isinstance(payload.get("links"), list) or isinstance(payload.get("last_node_id"), int)


def _resolve_prompt_metadata_payload(prompt_dict=None, extra_pnginfo=None):
    if prompt_dict is not None:
        return _normalize_png_json_payload(prompt_dict)
    if isinstance(extra_pnginfo, dict):
        prompt_blob = extra_pnginfo.get("prompt")
        if prompt_blob is not None:
            return _normalize_png_json_payload(prompt_blob)
    return None


def _resolve_workflow_metadata_payload(prompt_dict, extra_pnginfo=None):
    if isinstance(extra_pnginfo, dict):
        workflow = extra_pnginfo.get("workflow")
        if workflow is not None:
            workflow_payload = _normalize_png_json_payload(workflow)
            if _looks_like_comfy_ui_workflow(workflow_payload):
                return workflow_payload
            if not _looks_like_comfy_api_prompt_graph(workflow_payload):
                return workflow_payload
    return None


def _resolve_source_file_metadata_value(extra_pnginfo=None):
    if not isinstance(extra_pnginfo, dict):
        return ""
    direct_value = extra_pnginfo.get("source_file") or extra_pnginfo.get("sourceFile")
    if isinstance(direct_value, str) and direct_value.strip():
        return direct_value.strip().replace("\\", "/")
    umbra_meta = extra_pnginfo.get("umbra_metadata")
    if isinstance(umbra_meta, dict):
        nested_value = umbra_meta.get("source_file") or umbra_meta.get("sourceFile")
        if isinstance(nested_value, str) and nested_value.strip():
            return nested_value.strip().replace("\\", "/")
    return ""


def _extend_generation_metadata_with_source_file(generation_meta, extra_pnginfo=None):
    if not isinstance(generation_meta, dict):
        generation_meta = {}
    source_file = _resolve_source_file_metadata_value(extra_pnginfo)
    if source_file:
        generation_meta["source_file"] = source_file
    return generation_meta


def _write_png_json_metadata(metadata, prompt_dict=None, extra_pnginfo=None, umbra_metadata=None):
    written_keys = set()

    prompt_payload = _resolve_prompt_metadata_payload(prompt_dict, extra_pnginfo)
    if prompt_payload is not None:
        metadata.add_text("prompt", _safe_json_dumps(prompt_payload))
        written_keys.add("prompt")

    workflow_payload = _resolve_workflow_metadata_payload(prompt_dict, extra_pnginfo)
    if workflow_payload is not None:
        metadata.add_text("workflow", _safe_json_dumps(workflow_payload))
        written_keys.add("workflow")

    if isinstance(extra_pnginfo, dict):
        for key, value in extra_pnginfo.items():
            if key in written_keys:
                continue
            if key == "workflow":
                workflow_payload = _normalize_png_json_payload(value)
                if _looks_like_comfy_api_prompt_graph(workflow_payload):
                    # ComfyUI treats the reserved "workflow" PNG chunk as an editor graph.
                    # API prompt graphs belong in "prompt" or app-specific metadata instead;
                    # writing them as "workflow" makes drag-to-canvas load an empty graph.
                    continue
            metadata.add_text(key, _safe_json_dumps(value))
            written_keys.add(key)

    if umbra_metadata is not None:
        metadata.add_text("umbra_metadata", _safe_json_dumps(umbra_metadata))


class UmbraLabSaveImage:
    """
    Save Image node that embeds ComfyUI and A1111 compatible metadata.
    Saves to: ComfyUI/output/ (or configured external base)
    """
    
    def __init__(self):
        # Keep for compatibility, but resolve per save call so date folders roll over at midnight.
        self.output_dir = get_umbra_output_dir()
        self.type = "output"
        self.prefix_append = ""
        self.compress_level = 4

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "filename_prefix": ("STRING", {"default": "UmbraLab_%date%"}),
                "positive_prompt": ("STRING", {"default": "", "multiline": True}),
                "negative_prompt": ("STRING", {"default": "", "multiline": True}),
            },
            "optional": {
                # Backwards/forward compatibility with workflows that pass CONDITIONING.
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "output_folder": ("STRING", {"default": ""}),
                "save_to_yyyy_mm_dd_folder": ("BOOLEAN", {"default": False}),
                "save_to_set_subfolder": ("BOOLEAN", {"default": False}),
                "set_subfolder": ("STRING", {"default": ""}),
                "save_set_to_style_subfolder": ("STRING", {"default": ""}),
                "model_name": ("STRING", {"default": ""}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": False}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "sampler_name": ("STRING", {"default": "euler"}),
                "scheduler": ("STRING", {"default": "normal"}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "save_images"
    OUTPUT_NODE = True
    CATEGORY = "Umbra Lab"

    def save_images(self, images, filename_prefix, positive_prompt="", negative_prompt="",
                    positive=None, negative=None, output_folder="", save_to_yyyy_mm_dd_folder=False,
                    save_to_set_subfolder=False, set_subfolder="",
                    save_set_to_style_subfolder="",
                    model_name="", seed=0, steps=20, cfg=7.0,
                    sampler_name="euler", scheduler="normal", prompt=None, unique_id=None,
                    extra_pnginfo=None, **kwargs):
        
        # Accept both text-style prompts and CONDITIONING-style inputs for compatibility.
        positive_prompt = self._resolve_prompt_text(positive_prompt, positive, prompt, "positive", unique_id)
        negative_prompt = self._resolve_prompt_text(negative_prompt, negative, prompt, "negative", unique_id)
        
        filename_prefix = _resolve_filename_prefix(filename_prefix, "UmbraLab_%date%") + self.prefix_append
        style_subfolder = _normalize_output_subfolder(_value_for_index(save_set_to_style_subfolder, 0, ""))
        save_to_date_folder = bool(style_subfolder) or _to_bool(_value_for_index(save_to_yyyy_mm_dd_folder, 0, False), default=False)
        normalized_set_subfolder = _normalize_output_subfolder(_value_for_index(set_subfolder, 0, ""))
        if style_subfolder and normalized_set_subfolder:
            resolved_set_subfolder = os.path.join(normalized_set_subfolder, style_subfolder)
        elif style_subfolder:
            resolved_set_subfolder = style_subfolder
        else:
            resolved_set_subfolder = normalized_set_subfolder
        save_to_set_folder = bool(style_subfolder) or _to_bool(_value_for_index(save_to_set_subfolder, 0, False), default=False)
        output_dir = get_umbra_output_dir(output_folder, save_to_date_folder, save_to_set_folder, resolved_set_subfolder)
        full_output_folder, filename, counter, subfolder, filename_prefix = \
            folder_paths.get_save_image_path(filename_prefix, output_dir, 
                                             images[0].shape[1], images[0].shape[0])
        
        results = []
        
        for batch_index, image in enumerate(images):
            image_seed = _normalize_seed(_value_for_index(seed, batch_index, 0))
            image_steps = _to_int(_value_for_index(steps, batch_index, 20), default=20, minimum=1)
            image_cfg = _to_float(_value_for_index(cfg, batch_index, 7.0), default=7.0)
            image_sampler = str(_value_for_index(sampler_name, batch_index, "euler") or "euler")
            image_scheduler = str(_value_for_index(scheduler, batch_index, "normal") or "normal")
            image_model_name = str(_value_for_index(model_name, batch_index, "") or "")

            # Convert tensor to PIL Image
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            
            # Create PNG metadata
            metadata = PngInfo()
            generation_meta = {
                "positive_prompt": positive_prompt,
                "negative_prompt": negative_prompt,
                "model": image_model_name,
                "seed": image_seed,
                "steps": image_steps,
                "cfg": image_cfg,
                "sampler": image_sampler,
                "scheduler": image_scheduler,
                "width": img.width,
                "height": img.height,
                "app_version": "UmbraLab/ComfyUI",
            }
            generation_meta = _extend_generation_metadata_with_source_file(generation_meta, extra_pnginfo)
            civitai_payload = _build_civitai_metadata_payload(positive_prompt, image_model_name)
            if civitai_payload.get("resources"):
                generation_meta["civitai"] = civitai_payload
            
            # === ComfyUI Standard Metadata ===
            _write_png_json_metadata(
                metadata,
                prompt_dict=prompt,
                extra_pnginfo=extra_pnginfo,
                umbra_metadata=generation_meta,
            )
            
            # === A1111 Compatible Format (for maximum compatibility) ===
            a1111_params = f"{positive_prompt}\n"
            if negative_prompt:
                a1111_params += f"Negative prompt: {negative_prompt}\n"
            a1111_params += (
                f"Steps: {image_steps}, Sampler: {image_sampler}, CFG scale: {image_cfg}, Seed: {image_seed}"
            )
            if image_scheduler:
                a1111_params += f", Schedule type: {image_scheduler}"
            if image_model_name:
                a1111_params += f", Model: {image_model_name}"
            a1111_params += f", Size: {img.width}x{img.height}"
            a1111_params = _append_civitai_parameters(a1111_params, civitai_payload)
            
            metadata.add_text("parameters", a1111_params)
            metadata.add_text("civitai_metadata", _safe_json_dumps(civitai_payload))
            
            # Save image
            file = f"{filename}_{counter:05}_.png"
            saved_path = os.path.join(full_output_folder, file)
            img.save(saved_path, pnginfo=metadata, compress_level=self.compress_level)

            preview_file, preview_subfolder, preview_type = create_temp_preview_copy(saved_path, "umbra_img")
            results.append({
                "filename": str(preview_file or file),
                "subfolder": str(preview_subfolder or ""),
                "type": str(preview_type or "output"),
                "fullpath": saved_path,
            })
            counter += 1

        return {"ui": {"images": results}}

    def _resolve_prompt_text(self, explicit_text, conditioning_input, prompt_dict, prompt_type, unique_id=None):
        return _resolve_prompt_text(explicit_text, conditioning_input, prompt_dict, prompt_type, unique_id)

    def _extract_prompt_from_workflow(self, prompt_dict, prompt_type):
        positive, negative = _extract_prompts_from_workflow(prompt_dict)
        return positive if prompt_type == "positive" else negative


class UmbraLabSaveImageSimple:
    """
    Simplified version that auto-extracts prompts from connected CLIP nodes.
    Saves to: ComfyUI/output/ (or configured external base)
    """
    
    def __init__(self):
        # Keep for compatibility, but resolve per save call so date folders roll over at midnight.
        self.output_dir = get_umbra_output_dir()
        self.type = "output"
        self.compress_level = 4

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "filename_prefix": ("STRING", {"default": "UmbraLab_%date%"}),
            },
            "optional": {
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "output_folder": ("STRING", {"default": ""}),
                "save_to_yyyy_mm_dd_folder": ("BOOLEAN", {"default": False}),
                "save_to_set_subfolder": ("BOOLEAN", {"default": False}),
                "set_subfolder": ("STRING", {"default": ""}),
                "save_set_to_style_subfolder": ("STRING", {"default": ""}),
                # Alias fields for workflows that send text prompts directly.
                "positive_prompt": ("STRING", {"default": "", "multiline": True}),
                "negative_prompt": ("STRING", {"default": "", "multiline": True}),
                "model_info": ("MODEL",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": False}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "sampler_name": (["euler", "euler_ancestral", "heun", "dpm_2", "dpm_2_ancestral",
                                  "lms", "dpm_fast", "dpm_adaptive", "dpmpp_2s_ancestral", 
                                  "dpmpp_sde", "dpmpp_2m", "dpmpp_2m_sde", "ddim", "uni_pc"],),
                "sampler_name_text": ("STRING", {"default": ""}),
                "scheduler": ("STRING", {"default": "normal"}),
                "model_name": ("STRING", {"default": ""}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "save_images"
    OUTPUT_NODE = True
    CATEGORY = "Umbra Lab"

    def save_images(self, images, filename_prefix, positive=None, negative=None,
                    output_folder="", save_to_yyyy_mm_dd_folder=False,
                    save_to_set_subfolder=False, set_subfolder="",
                    save_set_to_style_subfolder="",
                    positive_prompt="", negative_prompt="", model_info=None, seed=0,
                    steps=20, cfg=7.0, sampler_name="euler", sampler_name_text="",
                    scheduler="normal", model_name="", prompt=None, unique_id=None,
                    extra_pnginfo=None, **kwargs):
        positive_text = _resolve_prompt_text(positive_prompt, positive, prompt, "positive", unique_id)
        negative_text = _resolve_prompt_text(negative_prompt, negative, prompt, "negative", unique_id)
        
        # Get model name from workflow
        resolved_model_name = str(model_name or "").strip()
        if not resolved_model_name and model_info is not None:
            resolved_model_name = _extract_model_name_from_model(model_info)
        if not resolved_model_name:
            resolved_model_name = self._extract_model_from_workflow(prompt) if prompt else ""
        
        filename_prefix = _resolve_filename_prefix(filename_prefix, "UmbraLab_%date%")
        style_subfolder = _normalize_output_subfolder(_value_for_index(save_set_to_style_subfolder, 0, ""))
        save_to_date_folder = bool(style_subfolder) or _to_bool(_value_for_index(save_to_yyyy_mm_dd_folder, 0, False), default=False)
        normalized_set_subfolder = _normalize_output_subfolder(_value_for_index(set_subfolder, 0, ""))
        if style_subfolder and normalized_set_subfolder:
            resolved_set_subfolder = os.path.join(normalized_set_subfolder, style_subfolder)
        elif style_subfolder:
            resolved_set_subfolder = style_subfolder
        else:
            resolved_set_subfolder = normalized_set_subfolder
        save_to_set_folder = bool(style_subfolder) or _to_bool(_value_for_index(save_to_set_subfolder, 0, False), default=False)
        output_dir = get_umbra_output_dir(output_folder, save_to_date_folder, save_to_set_folder, resolved_set_subfolder)
        full_output_folder, filename, counter, subfolder, filename_prefix = \
            folder_paths.get_save_image_path(filename_prefix, output_dir, 
                                             images[0].shape[1], images[0].shape[0])
        
        results = []
        
        for batch_index, image in enumerate(images):
            image_seed = _normalize_seed(_value_for_index(seed, batch_index, 0))
            image_steps = _to_int(_value_for_index(steps, batch_index, 20), default=20, minimum=1)
            image_cfg = _to_float(_value_for_index(cfg, batch_index, 7.0), default=7.0)
            widget_sampler = _value_for_index(sampler_name, batch_index, "euler")
            wired_sampler = _value_for_index(sampler_name_text, batch_index, "")
            image_sampler = str(wired_sampler or widget_sampler or "euler")
            image_scheduler = str(_value_for_index(scheduler, batch_index, "normal") or "normal")
            image_model_name = str(_value_for_index(resolved_model_name, batch_index, "") or "")

            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            
            metadata = PngInfo()
            generation_meta = {
                "positive_prompt": positive_text,
                "negative_prompt": negative_text,
                "model": resolved_model_name,
                "seed": image_seed,
                "steps": image_steps,
                "cfg": image_cfg,
                "sampler": image_sampler,
                "scheduler": image_scheduler,
                "width": img.width,
                "height": img.height,
                "app_version": "UmbraLab/ComfyUI",
            }
            generation_meta = _extend_generation_metadata_with_source_file(generation_meta, extra_pnginfo)
            
            # ComfyUI metadata
            _write_png_json_metadata(
                metadata,
                prompt_dict=prompt,
                extra_pnginfo=extra_pnginfo,
                umbra_metadata=generation_meta,
            )
            
            # A1111 format
            a1111_params = f"{positive_text}\n"
            if negative_text:
                a1111_params += f"Negative prompt: {negative_text}\n"
            a1111_params += (
                f"Steps: {image_steps}, Sampler: {image_sampler}, CFG scale: {image_cfg}, Seed: {image_seed}"
            )
            if image_scheduler:
                a1111_params += f", Schedule type: {image_scheduler}"
            if image_model_name:
                a1111_params += f", Model: {image_model_name}"
            a1111_params += f", Size: {img.width}x{img.height}"
            
            metadata.add_text("parameters", a1111_params)
            
            file = f"{filename}_{counter:05}_.png"
            saved_path = os.path.join(full_output_folder, file)
            img.save(saved_path, pnginfo=metadata, compress_level=self.compress_level)

            preview_file, preview_subfolder, preview_type = create_temp_preview_copy(saved_path, "umbra_img")
            results.append({
                "filename": str(preview_file or file),
                "subfolder": str(preview_subfolder or ""),
                "type": str(preview_type or "output"),
                "fullpath": saved_path,
            })
            counter += 1

        return {"ui": {"images": results}}

    def _extract_prompt_from_workflow(self, prompt_dict, prompt_type):
        """Extract prompt text from the workflow prompt dictionary."""
        if not prompt_dict:
            return ""
        
        for node_id, node_data in prompt_dict.items():
            class_type = node_data.get("class_type", "")
            inputs = node_data.get("inputs", {})
            
            if class_type == "CLIPTextEncode":
                text = inputs.get("text", "")
                if isinstance(text, str) and len(text) > 0:
                    # Heuristic: negative prompts often contain these keywords
                    is_negative = any(kw in text.lower() for kw in 
                                      ["ugly", "bad", "worst", "blurry", "deformed"])
                    
                    if prompt_type == "negative" and is_negative:
                        return text
                    elif prompt_type == "positive" and not is_negative:
                        return text
        
        return ""

    def _extract_model_from_workflow(self, prompt_dict):
        """Extract model name from the workflow."""
        return _extract_first_model_name_from_workflow(prompt_dict)


class UmbraCFGValue:
    """
    Utility node for supplying CFG values to downstream nodes.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 100.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("cfg",)
    FUNCTION = "output_value"
    CATEGORY = "Umbra Lab"

    def output_value(self, cfg):
        return (float(cfg),)


class UmbraInfinitePromptBuilder:
    """
    Simple multiline prompt composer with fixed fields.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "field_1": ("STRING", {"default": "", "multiline": True}),
                "field_2": ("STRING", {"default": "", "multiline": True}),
            },
            "optional": ContainsAll({
                "field_3": ("STRING", {"default": "", "multiline": True}),
                "field_4": ("STRING", {"default": "", "multiline": True}),
                "field_5": ("STRING", {"default": "", "multiline": True}),
                "field_6": ("STRING", {"default": "", "multiline": True}),
                "field_7": ("STRING", {"default": "", "multiline": True}),
                "field_8": ("STRING", {"default": "", "multiline": True}),
                "separator": ("STRING", {"default": ", "}),
                "normalize_whitespace": ("BOOLEAN", {"default": True}),
                "remove_all_whitespace": ("BOOLEAN", {"default": False}),
                "underscore_mode": (
                    [
                        "keep",
                        "spaces_to_underscores",
                        "underscores_to_spaces",
                        "remove_underscores",
                    ],
                    {"default": "keep"},
                ),
                "drop_empty_fields": ("BOOLEAN", {"default": True}),
            }),
            "hidden": ContainsAll({
                "prompt": "PROMPT",
                "unique_id": "UNIQUE_ID",
            }),
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "compose"
    CATEGORY = "Umbra Lab"

    def compose(
        self,
        field_1="",
        field_2="",
        field_3="",
        field_4="",
        field_5="",
        field_6="",
        field_7="",
        field_8="",
        separator=", ",
        normalize_whitespace=True,
        remove_all_whitespace=False,
        underscore_mode="keep",
        drop_empty_fields=True,
        prompt=None,
        unique_id=None,
    ):
        ordered_names = [
            "field_1",
            "field_2",
            "field_3",
            "field_4",
            "field_5",
            "field_6",
            "field_7",
            "field_8",
        ]
        values_by_name = {
            "field_1": field_1,
            "field_2": field_2,
            "field_3": field_3,
            "field_4": field_4,
            "field_5": field_5,
            "field_6": field_6,
            "field_7": field_7,
            "field_8": field_8,
        }
        segments = []
        for name in ordered_names:
            raw_value = values_by_name.get(name, "")
            if isinstance(raw_value, (list, tuple)) and len(raw_value) > 0:
                extracted = _extract_text_from_prompt_reference(prompt, raw_value)
                if extracted:
                    raw_value = extracted
                else:
                    raw_value = ""
            elif isinstance(raw_value, dict):
                raw_value = ""
            text = _normalize_prompt_segment(
                raw_value,
                normalize_whitespace=normalize_whitespace,
                remove_all_whitespace=remove_all_whitespace,
                underscore_mode=underscore_mode,
            )
            if _to_bool(drop_empty_fields, default=True) and not text:
                continue
            segments.append(text)

        joiner = str(separator if separator is not None else ", ")
        result = joiner.join([segment for segment in segments if segment is not None])
        if _to_bool(normalize_whitespace, default=True) and joiner.strip():
            # Clean accidental repeated separators and edge separators.
            sep_escaped = re.escape(joiner)
            result = re.sub(rf"(?:{sep_escaped}){{2,}}", joiner, result)
            result = result.strip()
            result = re.sub(rf"^(?:{sep_escaped})+|(?:{sep_escaped})+$", "", result).strip()
        return (result,)


class UmbraA1111LoraSyntax:
    """
    Parses A1111 LoRA syntax from a prompt string and applies LoRAs to MODEL/CLIP.
    Example tags:
      <lora:my_lora:1>
      <lora:my_lora:0.8:0.6>
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "prompt_text": ("STRING", {"default": "", "multiline": True}),
            },
            "optional": {
                "positive_in": ("CONDITIONING",),
                "lora_syntax_text": ("STRING", {"default": "", "multiline": True}),
                # Keep this widget name aligned with stock ComfyUI so model-info tooling behaves the same.
                "lora_name": (_get_lora_dropdown_options(), {"default": _LORA_NONE_OPTION}),
                "lora_strength_model": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.05}),
                "lora_strength_clip": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.05}),
                "strict_mode": ("BOOLEAN", {"default": False}),
                "strip_lora_tags_from_text": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("MODEL", "CLIP", "STRING", "CONDITIONING")
    RETURN_NAMES = (
        "model",
        "clip",
        "prompt_text",
        "positive",
    )
    FUNCTION = "apply_lora_syntax"
    CATEGORY = "Umbra Lab"

    def apply_lora_syntax(
        self,
        model,
        clip,
        prompt_text,
        positive_in=None,
        lora_syntax_text="",
        lora_name=_LORA_NONE_OPTION,
        lora_strength_model=1.0,
        lora_strength_clip=1.0,
        strict_mode=False,
        strip_lora_tags_from_text=True,
        **kwargs,
    ):
        # Power Prompter source-of-truth: LoRA application should follow only the
        # explicit <lora:...> tags present in prompt_text to avoid stale widget state.
        _ = lora_syntax_text
        _ = lora_name
        _ = lora_strength_model
        _ = lora_strength_clip
        _ = kwargs.get("selected_lora", _LORA_NONE_OPTION)

        syntax_text_value = ""
        parsed_tags = []
        parsed_tags.extend(_parse_a1111_lora_tags(prompt_text))

        merged_tags = []
        seen = set()
        for tag in parsed_tags:
            name = str(tag.get("name") or "").strip()
            if not name:
                continue
            sm = _to_float(tag.get("strength_model"), default=1.0)
            sc = _to_float(tag.get("strength_clip"), default=sm)
            key = (name.lower(), float(sm), float(sc))
            if key in seen:
                continue
            seen.add(key)
            merged_tags.append(
                {
                    "name": name,
                    "strength_model": sm,
                    "strength_clip": sc,
                }
            )

        syntax_text_value = "\n".join(
            [
                _format_lora_tag(tag["name"], tag["strength_model"], tag["strength_clip"])
                for tag in merged_tags
                if str(tag.get("name") or "").strip()
            ]
        ).strip()

        updated_model = model
        updated_clip = clip

        missing = []
        for tag in merged_tags:
            requested_name = tag["name"]
            resolved_filename, _ = _resolve_lora_path(requested_name)
            if not resolved_filename:
                missing.append(requested_name)
                continue

            try:
                lora_state = _load_lora_state(resolved_filename)
                updated_model, updated_clip = comfy.sd.load_lora_for_models(
                    updated_model,
                    updated_clip,
                    lora_state,
                    float(tag["strength_model"]),
                    float(tag["strength_clip"]),
                )
            except Exception as error:
                if _to_bool(strict_mode, default=False):
                    raise
                logger.warn(f"Failed to apply LoRA '{requested_name}': {error}")

        if missing:
            message = f"Missing LoRA(s): {', '.join(missing)}"
            if _to_bool(strict_mode, default=False):
                raise Exception(message)
            logger.warn(message)

        output_text = (
            _strip_a1111_lora_tags(prompt_text)
            if _to_bool(strip_lora_tags_from_text, default=True)
            else str(prompt_text or "")
        )

        # Preserve upstream conditioning when the graph provides it. SDXL-style
        # conditioning carries pooled metadata there, and replacing it with a
        # fresh encode can leave clip_pooled as None in the sampler.
        positive = positive_in if positive_in is not None else None
        if positive is None and updated_clip is not None:
            try:
                tokens = updated_clip.tokenize(output_text)
                # Match ComfyUI's native CLIPTextEncode path so special model
                # families keep their full conditioning metadata after LoRA parsing.
                positive = updated_clip.encode_from_tokens_scheduled(tokens)
            except Exception as error:
                logger.warn(f"Failed to encode LoRA positive conditioning: {error}")

        return (
            updated_model,
            updated_clip,
            output_text,
            positive if positive is not None else [],
        )


class UmbraKSampler:
    """
    Umbra KSampler wrapper around the core ComfyUI common_ksampler.
    Adds metadata passthrough outputs and optional per-image seed progression.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "seed": ("INT", {"default": 0, "min": 0, "max": SEED_MAX}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "sampler_name": (_get_sampler_names(),),
                "scheduler": (_get_scheduler_names(),),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "seed_mode": (["fixed", "increment_per_image", "random_per_image"], {"default": "increment_per_image"}),
                "seed_step": ("INT", {"default": 1, "min": 1, "max": SEED_MAX}),
                "repeat_behavior": (["inherit", "increment_per_repeat", "random_per_repeat", "none"], {"default": "inherit"}),
                "repeat_step": ("INT", {"default": 1, "min": 1, "max": SEED_MAX}),
                "style_seed_behavior": (["normal", "same_seed_style_cycle"], {"default": "normal"}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("LATENT", "STRING", "INT", "INT", "FLOAT", "STRING", "STRING")
    RETURN_NAMES = ("samples", "model_name", "seed", "steps", "cfg", "sampler_name", "scheduler")
    FUNCTION = "sample"
    CATEGORY = "Umbra Lab"

    def sample(
        self,
        model,
        positive,
        negative,
        latent_image,
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        denoise,
        seed_mode="increment_per_image",
        seed_step=1,
        repeat_behavior="inherit",
        repeat_step=1,
        style_seed_behavior="normal",
        prompt=None,
        unique_id=None,
    ):
        core_ksampler = _get_core_common_ksampler()
        if core_ksampler is None:
            raise Exception("Core ComfyUI common_ksampler is unavailable; cannot execute UmbraKSampler.")

        rng = random.SystemRandom()
        effective_seed = _normalize_seed(seed)
        effective_steps = _to_int(steps, default=20, minimum=1)
        effective_cfg = _to_float(cfg, default=7.0)
        effective_sampler = str(sampler_name or "euler")
        effective_scheduler = str(scheduler or "normal")
        effective_denoise = _to_float(denoise, default=1.0, minimum=0.0, maximum=1.0)
        effective_seed_step = _to_int(seed_step, default=1, minimum=1, maximum=SEED_MAX)
        effective_repeat_step = _to_int(repeat_step, default=1, minimum=1, maximum=SEED_MAX)
        model_name = _extract_model_name_from_node_connection(prompt, unique_id, "model")
        if not model_name:
            model_name = _extract_first_model_name_from_workflow(prompt)
        if not model_name:
            model_name = _extract_model_name_from_model(model)

        style_seed_mode = str(style_seed_behavior or "normal")
        is_same_seed_style_cycle = style_seed_mode == "same_seed_style_cycle"
        repeat_mode = "none" if is_same_seed_style_cycle else repeat_behavior
        seed_source_class, seed_source_output, _ = _get_connected_input_info(prompt, unique_id, "seed")
        if repeat_mode == "inherit":
            if seed_source_output == 1:
                # Seed list output already varies upstream.
                repeat_mode = "none"
            elif seed_source_class == "UmbraSeedValue":
                # UmbraSeedValue handles repeat progression itself.
                repeat_mode = "none"
            else:
                # Static/manual/non-Umbra sources get automatic repeat progression.
                repeat_mode = "increment_per_repeat"

        if repeat_mode == "random_per_repeat":
            effective_seed = rng.randint(0, SEED_MAX)
        elif repeat_mode == "increment_per_repeat":
            repeat_key = _build_repeat_key(
                "UmbraKSampler",
                unique_id,
                prompt,
                {
                    "seed": effective_seed,
                    "seed_mode": seed_mode,
                    "seed_step": effective_seed_step,
                    "steps": effective_steps,
                    "cfg": effective_cfg,
                    "sampler_name": effective_sampler,
                    "scheduler": effective_scheduler,
                    "repeat_step": effective_repeat_step,
                },
            )
            repeat_index = _next_repeat_index("UmbraKSampler", repeat_key)
            effective_seed = _normalize_seed(effective_seed + (repeat_index * effective_repeat_step))

        latent_samples = latent_image.get("samples") if isinstance(latent_image, dict) else None
        batch_size = 1
        if isinstance(latent_samples, torch.Tensor) and latent_samples.ndim >= 1:
            batch_size = int(latent_samples.shape[0])

        batch_seed_mode = seed_mode
        if is_same_seed_style_cycle and seed_mode == "fixed":
            batch_seed_mode = "increment_per_image"

        if batch_size <= 1 or batch_seed_mode == "fixed":
            sampled = core_ksampler(
                model,
                effective_seed,
                effective_steps,
                effective_cfg,
                effective_sampler,
                effective_scheduler,
                positive,
                negative,
                latent_image,
                denoise=effective_denoise,
            )[0]
            return (
                sampled,
                model_name,
                effective_seed,
                effective_steps,
                effective_cfg,
                effective_sampler,
                effective_scheduler,
            )

        sampled_batches = []
        for i in range(batch_size):
            if batch_seed_mode == "random_per_image":
                item_seed = rng.randint(0, SEED_MAX)
            else:
                item_seed = _normalize_seed(effective_seed + (i * effective_seed_step))

            latent_item = dict(latent_image)
            latent_item["samples"] = latent_samples[i:i + 1]

            noise_mask = latent_image.get("noise_mask") if isinstance(latent_image, dict) else None
            if isinstance(noise_mask, torch.Tensor) and noise_mask.ndim >= 1 and noise_mask.shape[0] == batch_size:
                latent_item["noise_mask"] = noise_mask[i:i + 1]

            batch_index = latent_image.get("batch_index") if isinstance(latent_image, dict) else None
            if isinstance(batch_index, list) and len(batch_index) == batch_size:
                latent_item["batch_index"] = [batch_index[i]]
            elif isinstance(batch_index, torch.Tensor) and batch_index.ndim >= 1 and batch_index.shape[0] == batch_size:
                latent_item["batch_index"] = batch_index[i:i + 1]

            sampled_item = core_ksampler(
                model,
                item_seed,
                effective_steps,
                effective_cfg,
                effective_sampler,
                effective_scheduler,
                positive,
                negative,
                latent_item,
                denoise=effective_denoise,
            )[0]
            sampled_batches.append(sampled_item["samples"])

        merged = dict(latent_image)
        merged["samples"] = torch.cat(sampled_batches, dim=0)
        return (
            merged,
            model_name,
            effective_seed,
            effective_steps,
            effective_cfg,
            effective_sampler,
            effective_scheduler,
        )

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        repeat_behavior = kwargs.get("repeat_behavior", "inherit")
        seed_mode = kwargs.get("seed_mode", "increment_per_image")
        style_seed_behavior = kwargs.get("style_seed_behavior", "normal")
        if style_seed_behavior == "same_seed_style_cycle":
            return float("nan")
        if repeat_behavior != "none":
            return float("nan")
        if seed_mode == "random_per_image":
            return float("nan")
        return "umbra_ksampler_stable"


_LATENT_UPSCALE_METHODS = ["nearest-exact", "bilinear", "area", "bicubic", "bislerp"]
_IMAGE_UPSCALE_METHODS = ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"]
_ALL_UPSCALE_METHODS = list(dict.fromkeys(_LATENT_UPSCALE_METHODS + _IMAGE_UPSCALE_METHODS))


class UmbraKSamplerNormal(UmbraKSampler):
    """
    Explicit normal-generation alias for UmbraKSampler.
    """

    CATEGORY = "Umbra Lab"


class UmbraKSamplerHiResFix:
    """
    Forge-style two-pass sampler: base generation, selectable upscale, and refinement.
    """

    _LATENT_PREFIX = "Latent"
    _PIXEL_UPSCALERS = {
        "Nearest": "nearest-exact",
        "Bilinear": "bilinear",
        "Area": "area",
        "Bicubic": "bicubic",
        "Lanczos": "lanczos",
    }

    @classmethod
    def _upscaler_choices(cls):
        latent = [
            "Latent",
            "Latent (nearest-exact)",
            "Latent (bilinear)",
            "Latent (area)",
            "Latent (bicubic)",
            "Latent (bislerp)",
        ]
        return latent + list(cls._PIXEL_UPSCALERS.keys()) + _preferred_upscale_model_choices()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "vae": ("VAE",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "seed": ("INT", {"default": 0, "min": 0, "max": SEED_MAX, "control_after_generate": True}),
                "steps": ("INT", {"default": 24, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "sampler_name": (_get_sampler_names(),),
                "scheduler": (_get_scheduler_names(),),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "enabled": ("BOOLEAN", {"default": False}),
                "upscaler": (cls._upscaler_choices(), {"default": "Latent"}),
                "resize_mode": (["upscale by", "resize to"], {"default": "upscale by"}),
                "scale_by": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 8.0, "step": 0.05}),
                "resize_width": ("INT", {"default": 0, "min": 0, "max": 16384, "step": 8}),
                "resize_height": ("INT", {"default": 0, "min": 0, "max": 16384, "step": 8}),
                "hires_steps": ("INT", {"default": 0, "min": 0, "max": 10000}),
                "hires_cfg": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "hires_sampler_name": (["Use same"] + _get_sampler_names(), {"default": "Use same"}),
                "hires_scheduler": (["Use same"] + _get_scheduler_names(), {"default": "Use same"}),
                "hires_denoise": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("LATENT", "STRING", "INT", "INT", "FLOAT", "STRING", "STRING", "INT")
    RETURN_NAMES = ("samples", "model_name", "seed", "steps", "cfg", "sampler_name", "scheduler", "hires_seed")
    FUNCTION = "sample"
    CATEGORY = "Umbra Lab"
    DESCRIPTION = "A Forge-style Hires Fix pass with latent, pixel, or installed model upscalers and exact output sizing."

    @classmethod
    def _round_dimension(cls, value):
        return min(16384, max(8, int(round(float(value) / 8.0) * 8)))

    @classmethod
    def _target_pixel_size(cls, latent, resize_mode, scale_by, resize_width, resize_height):
        latent_samples = latent.get("samples") if isinstance(latent, dict) else None
        if not isinstance(latent_samples, torch.Tensor):
            raise RuntimeError("Umbra Hires Fix received an invalid latent image.")

        base_width = max(8, int(latent_samples.shape[-1]) * 8)
        base_height = max(8, int(latent_samples.shape[-2]) * 8)
        safe_scale = _to_float(scale_by, default=2.0, minimum=1.0, maximum=8.0)
        requested_width = _to_int(resize_width, default=0, minimum=0)
        requested_height = _to_int(resize_height, default=0, minimum=0)

        if str(resize_mode or "").strip().lower() == "resize to" and (requested_width > 0 or requested_height > 0):
            if requested_width <= 0:
                requested_width = round(requested_height * base_width / float(base_height))
            if requested_height <= 0:
                requested_height = round(requested_width * base_height / float(base_width))
            return cls._round_dimension(requested_width), cls._round_dimension(requested_height)

        return (
            cls._round_dimension(base_width * safe_scale),
            cls._round_dimension(base_height * safe_scale),
        )

    @classmethod
    def _latent_method(cls, upscaler):
        normalized = str(upscaler or "Latent").strip()
        if normalized == "Latent":
            return "bislerp"
        if normalized.startswith("Latent (") and normalized.endswith(")"):
            requested = normalized[len("Latent ("):-1]
            return requested if requested in _LATENT_UPSCALE_METHODS else "bislerp"
        return None

    @classmethod
    def _resize_pixels(cls, pixels, target_width, target_height, method):
        channels_first = pixels.movedim(-1, 1)
        resized = comfy_utils.common_upscale(
            channels_first,
            target_width,
            target_height,
            method,
            "disabled",
        )
        return resized.movedim(1, -1)

    @classmethod
    def _upscale_latent_for_hires(cls, vae, latent, upscaler, target_width, target_height):
        latent_samples = latent.get("samples") if isinstance(latent, dict) else None
        if not isinstance(latent_samples, torch.Tensor):
            raise RuntimeError("Umbra Hires Fix received an invalid latent image.")

        latent_method = cls._latent_method(upscaler)
        if latent_method:
            out = dict(latent)
            out["samples"] = comfy_utils.common_upscale(
                latent_samples,
                max(1, target_width // 8),
                max(1, target_height // 8),
                latent_method,
                "disabled",
            )
            return out

        decoded = vae.decode(latent_samples)
        if not isinstance(decoded, torch.Tensor):
            raise RuntimeError("Umbra Hires Fix could not decode the base latent.")
        if len(decoded.shape) == 5:
            decoded = decoded.reshape(-1, decoded.shape[-3], decoded.shape[-2], decoded.shape[-1])

        pixel_method = cls._PIXEL_UPSCALERS.get(str(upscaler or "").strip())
        if pixel_method:
            upscaled_pixels = cls._resize_pixels(decoded, target_width, target_height, pixel_method)
        else:
            try:
                from comfy_extras.nodes_upscale_model import ImageUpscaleWithModel
                loaded_model = _load_umbra_upscale_model(upscaler)
                upscaled_pixels = ImageUpscaleWithModel.upscale(loaded_model, decoded)[0]
                if int(upscaled_pixels.shape[2]) != target_width or int(upscaled_pixels.shape[1]) != target_height:
                    upscaled_pixels = cls._resize_pixels(upscaled_pixels, target_width, target_height, "lanczos")
            except Exception as exc:
                if isinstance(exc, RuntimeError) and str(exc).startswith("Umbra UI"):
                    raise
                raise RuntimeError(f"Umbra Hires Fix failed to apply upscaler '{upscaler}'.") from exc

        encoded = vae.encode(upscaled_pixels)
        out = dict(latent)
        out["samples"] = encoded
        return out

    def sample(
        self,
        model,
        vae,
        positive,
        negative,
        latent_image,
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        denoise,
        enabled,
        upscaler,
        resize_mode,
        scale_by,
        resize_width,
        resize_height,
        hires_steps,
        hires_cfg,
        hires_sampler_name,
        hires_scheduler,
        hires_denoise,
    ):
        core_ksampler = _get_core_common_ksampler()
        if core_ksampler is None:
            raise Exception("Core ComfyUI common_ksampler is unavailable; cannot execute UmbraKSamplerHiResFix.")

        effective_seed = _normalize_seed(seed)
        effective_steps = _to_int(steps, default=24, minimum=1)
        effective_cfg = _to_float(cfg, default=7.0)
        effective_sampler = str(sampler_name or "euler")
        effective_scheduler = str(scheduler or "normal")
        effective_denoise = _to_float(denoise, default=1.0, minimum=0.0, maximum=1.0)
        requested_hires_steps = _to_int(hires_steps, default=0, minimum=0)
        requested_hires_cfg = _to_float(hires_cfg, default=0.0, minimum=0.0, maximum=100.0)
        effective_hires_steps = requested_hires_steps or effective_steps
        effective_hires_cfg = requested_hires_cfg or effective_cfg
        effective_hires_sampler = str(hires_sampler_name or "Use same")
        if effective_hires_sampler == "Use same":
            effective_hires_sampler = effective_sampler
        effective_hires_scheduler = str(hires_scheduler or "Use same")
        if effective_hires_scheduler == "Use same":
            effective_hires_scheduler = effective_scheduler
        effective_hires_denoise = _to_float(hires_denoise, default=0.35, minimum=0.0, maximum=1.0)
        model_name = _extract_model_name_from_model(model)

        base_sample = core_ksampler(
            model,
            effective_seed,
            effective_steps,
            effective_cfg,
            effective_sampler,
            effective_scheduler,
            positive,
            negative,
            latent_image,
            denoise=effective_denoise,
        )[0]

        if not bool(enabled):
            return (
                base_sample,
                model_name,
                effective_seed,
                effective_steps,
                effective_cfg,
                effective_sampler,
                effective_scheduler,
                effective_seed,
            )

        target_width, target_height = self._target_pixel_size(
            base_sample,
            resize_mode,
            scale_by,
            resize_width,
            resize_height,
        )

        hires_input_latent = self._upscale_latent_for_hires(
            vae=vae,
            latent=base_sample,
            upscaler=upscaler,
            target_width=target_width,
            target_height=target_height,
        )

        hires_seed = effective_seed
        refined = core_ksampler(
            model,
            hires_seed,
            effective_hires_steps,
            effective_hires_cfg,
            effective_hires_sampler,
            effective_hires_scheduler,
            positive,
            negative,
            hires_input_latent,
            denoise=effective_hires_denoise,
        )[0]

        return (
            refined,
            model_name,
            effective_seed,
            effective_steps,
            effective_cfg,
            effective_sampler,
            effective_scheduler,
            hires_seed,
        )

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        scale_by = _to_float(kwargs.get("scale_by", 2.0), default=2.0, minimum=1.0, maximum=8.0)
        upscaler = str(kwargs.get("upscaler", "Latent"))
        resize_mode = str(kwargs.get("resize_mode", "upscale by"))
        resize_width = _to_int(kwargs.get("resize_width", 0), default=0, minimum=0)
        resize_height = _to_int(kwargs.get("resize_height", 0), default=0, minimum=0)
        enabled = bool(kwargs.get("enabled", False))
        stable_seed = _normalize_seed(kwargs.get("seed", 0))
        return f"umbra_ksampler_hiresfix:{stable_seed}:{enabled}:{upscaler}:{resize_mode}:{scale_by:.4f}:{resize_width}x{resize_height}"


class UmbraStepsValue:
    """
    Utility node for supplying step count values to downstream nodes.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
            }
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("steps",)
    FUNCTION = "output_value"
    CATEGORY = "Umbra Lab"

    def output_value(self, steps):
        return (_to_int(steps, default=20, minimum=1),)


class UmbraSeedValue:
    """
    Utility node for supplying seed values to downstream nodes.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "mode": (["fixed", "randomize"], {"default": "fixed"}),
            },
            "optional": {
                "batch_mode": (["single", "increment_per_item", "random_per_item"], {"default": "single"}),
                "batch_count": ("INT", {"default": 1, "min": 1, "max": BIGMAX, "step": 1}),
                "repeat_behavior": (["increment_per_repeat", "random_per_repeat", "none"], {"default": "increment_per_repeat"}),
                "repeat_step": ("INT", {"default": 1, "min": 1, "max": SEED_MAX}),
                "style_seed_behavior": (["normal", "same_seed_style_cycle"], {"default": "normal"}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("seed", "seed_list")
    OUTPUT_IS_LIST = (False, True)
    FUNCTION = "output_seed"
    CATEGORY = "Umbra Lab"

    def output_seed(
        self,
        seed,
        mode,
        batch_mode="single",
        batch_count=1,
        repeat_behavior="increment_per_repeat",
        repeat_step=1,
        style_seed_behavior="normal",
        prompt=None,
        unique_id=None,
        increment_step=1,
    ):
        seed_count = _to_int(batch_count, default=1, minimum=1)
        step = _to_int(increment_step, default=1, minimum=1, maximum=SEED_MAX)
        effective_repeat_step = _to_int(repeat_step, default=1, minimum=1, maximum=SEED_MAX)
        rng = random.SystemRandom()

        base_seed = _normalize_seed(seed)
        if mode == "randomize":
            base_seed = rng.randint(0, SEED_MAX)

        style_seed_mode = str(style_seed_behavior or "normal")
        is_same_seed_style_cycle = style_seed_mode == "same_seed_style_cycle"
        effective_repeat_behavior = "none" if is_same_seed_style_cycle else repeat_behavior

        if effective_repeat_behavior == "random_per_repeat":
            base_seed = rng.randint(0, SEED_MAX)
        elif effective_repeat_behavior == "increment_per_repeat":
            repeat_key = _build_repeat_key(
                "UmbraSeedValue",
                unique_id,
                prompt,
                {
                    "seed": _normalize_seed(seed),
                    "mode": mode,
                    "batch_mode": batch_mode,
                    "batch_count": seed_count,
                    "increment_step": step,
                    "repeat_step": effective_repeat_step,
                },
            )
            repeat_index = _next_repeat_index("UmbraSeedValue", repeat_key)
            base_seed = _normalize_seed(base_seed + (repeat_index * effective_repeat_step))

        effective_batch_mode = batch_mode
        if is_same_seed_style_cycle and batch_mode == "single" and seed_count > 1:
            effective_batch_mode = "increment_per_item"

        if effective_batch_mode == "random_per_item":
            seeds = [rng.randint(0, SEED_MAX) for _ in range(seed_count)]
        elif effective_batch_mode == "increment_per_item":
            seeds = [_normalize_seed(base_seed + (i * step)) for i in range(seed_count)]
        else:
            seeds = [base_seed for _ in range(seed_count)]

        return (int(seeds[0]), seeds)

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        mode = kwargs.get("mode", "fixed")
        batch_mode = kwargs.get("batch_mode", "single")
        repeat_behavior = kwargs.get("repeat_behavior", "increment_per_repeat")
        style_seed_behavior = kwargs.get("style_seed_behavior", "normal")
        if style_seed_behavior == "same_seed_style_cycle":
            return float("nan")
        if mode == "randomize":
            return float("nan")
        if batch_mode == "random_per_item":
            return float("nan")
        if repeat_behavior != "none":
            return float("nan")
        stable_seed = _normalize_seed(kwargs.get("seed", 0))
        stable_count = _to_int(kwargs.get("batch_count", 1), default=1, minimum=1)
        stable_step = _to_int(kwargs.get("increment_step", 1), default=1, minimum=1, maximum=SEED_MAX)
        return f"umbra_seed_stable:{stable_seed}:{stable_count}:{stable_step}"


def _safe_filename_list(folder_key):
    try:
        entries = list(folder_paths.get_filename_list(folder_key) or [])
    except Exception:
        entries = []
    entries = [str(v) for v in entries if str(v or "").strip()]
    entries.sort(key=lambda v: v.lower())
    return entries


def _discover_diffusers_models():
    discovered = []
    try:
        search_paths = list(folder_paths.get_folder_paths("diffusers") or [])
    except Exception:
        search_paths = []

    for search_path in search_paths:
        if not os.path.isdir(search_path):
            continue
        for root, _, files in os.walk(search_path, followlinks=True):
            if "model_index.json" not in files:
                continue
            rel = os.path.relpath(root, start=search_path).replace("\\", "/")
            rel = str(rel or "").strip()
            if rel and rel != ".":
                discovered.append(rel)

    discovered = sorted(set(discovered), key=lambda v: v.lower())
    return discovered


def _weight_dtype_model_options(weight_dtype):
    options = {}
    key = str(weight_dtype or "default").strip().lower()
    if key in ("false", "true", "0", "1", "none", "[none]"):
        key = "default"
    if key == "fp8_e4m3fn":
        options["dtype"] = torch.float8_e4m3fn
    elif key == "fp8_e4m3fn_fast":
        options["dtype"] = torch.float8_e4m3fn
        options["fp8_optimizations"] = True
    elif key == "fp8_e5m2":
        options["dtype"] = torch.float8_e5m2
    return options


def _with_legacy_choices(values, extras=None):
    merged = []
    seen = set()
    for entry in list(values or []) + list(extras or []):
        candidate = str(entry or "").strip()
        key = candidate.lower()
        if key in seen:
            continue
        seen.add(key)
        merged.append(candidate)
    if len(merged) == 0:
        return [""]
    return merged


def _sanitize_model_selector(value, empty_tokens=None):
    token = str(value or "").strip().replace("\\", "/")
    lowered = token.lower()
    if lowered in set(empty_tokens or ("", "[none]", "none", "default", "false", "true")):
        return ""
    return token


def _strip_model_folder_prefix(value, folder_name):
    token = str(value or "").strip().replace("\\", "/").lstrip("/")
    parts = [part for part in token.split("/") if part]
    if len(parts) > 1 and parts[0].lower() == str(folder_name or "").lower():
        return "/".join(parts[1:])
    return token


def _resolve_model_file(folder_key, selector, folder_prefix=None):
    name = _sanitize_model_selector(selector)
    if not name:
        raise Exception(f"No {str(folder_key or 'model').replace('_', ' ')} selected.")

    stripped = _strip_model_folder_prefix(name, folder_prefix or folder_key)
    candidates = []
    for candidate in (name, stripped, name.replace("\\", "/"), stripped.replace("\\", "/")):
        candidate = str(candidate or "").strip()
        if candidate and candidate not in candidates:
            candidates.append(candidate)

    for candidate in candidates:
        try:
            path = folder_paths.get_full_path(folder_key, candidate)
        except Exception:
            path = None
        if path:
            return path, candidate

    absolute_candidate = stripped
    if os.path.isabs(absolute_candidate) and os.path.isfile(absolute_candidate):
        return absolute_candidate, name

    basename = os.path.basename(stripped.replace("\\", "/"))
    if basename:
        matches = []
        for entry in _safe_filename_list(folder_key):
            if os.path.basename(str(entry or "").replace("\\", "/")).lower() == basename.lower():
                matches.append(str(entry or "").strip())
        if len(matches) == 1:
            path = folder_paths.get_full_path(folder_key, matches[0])
            if path:
                return path, matches[0]

    raise FileNotFoundError(f"Model in folder '{folder_key}' with filename '{name}' not found.")


def _first_filename_or_empty(folder_key):
    entries = _safe_filename_list(folder_key)
    if len(entries) == 0:
        return ""
    return str(entries[0] or "").strip()


def _discover_gguf_diffusion_models():
    """Return every GGUF diffusion model visible to ComfyUI-GGUF."""
    names = []
    for folder_key in ("unet_gguf", "diffusion_models", "unet"):
        for name in _safe_filename_list(folder_key):
            normalized = str(name or "").strip()
            if normalized.lower().endswith(".gguf") and normalized not in names:
                names.append(normalized)
    return names


class UmbraLoadCheckpoint:
    """
    Umbra unified model loader.
    Supports checkpoints, diffusers directories, diffusion-model weights, UNets, and GGUF.
    """

    @classmethod
    def INPUT_TYPES(cls):
        checkpoints = _with_legacy_choices(_safe_filename_list("checkpoints"), extras=["", "[None]"])
        diffusion_models = _with_legacy_choices(_safe_filename_list("diffusion_models"), extras=["", "[None]"])
        unet_models = _with_legacy_choices(_safe_filename_list("unet"), extras=["", "[None]"])
        gguf_models = _with_legacy_choices(_discover_gguf_diffusion_models(), extras=["", "default", "[None]"])
        diffusers_models = _with_legacy_choices(_discover_diffusers_models(), extras=["", "[None]"])

        return {
            "required": {
                "model_type": (
                    ["checkpoint", "diffusers", "diffusion_model", "unet", "gguf"],
                    {"default": "checkpoint"},
                ),
                "checkpoint_name": (checkpoints,),
                "diffusers_model": (diffusers_models,),
                "diffusion_model_name": (diffusion_models,),
                "unet_name": (unet_models,),
                "gguf_name": (gguf_models,),
                "weight_dtype": (
                    ["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2", "False", "True", "false", "true"],
                    {"default": "default"},
                ),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP", "VAE", "STRING")
    RETURN_NAMES = ("model", "clip", "vae", "model_name")
    FUNCTION = "load_model"
    CATEGORY = "Umbra Lab"

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        return True

    def _load_checkpoint(self, checkpoint_name):
        name = str(checkpoint_name or "").strip()
        if not name:
            raise Exception("No checkpoint selected.")
        ckpt_path, resolved_name = _resolve_model_file("checkpoints", name, "checkpoints")
        out = comfy.sd.load_checkpoint_guess_config(
            ckpt_path,
            output_vae=True,
            output_clip=True,
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
        )
        return (out[0], out[1], out[2], resolved_name)

    def _load_diffusers(self, diffusers_model):
        model_path = str(diffusers_model or "").strip()
        if not model_path:
            raise Exception("No diffusers model selected.")

        resolved_path = model_path
        try:
            search_paths = list(folder_paths.get_folder_paths("diffusers") or [])
        except Exception:
            search_paths = []
        for search_path in search_paths:
            candidate = os.path.join(search_path, model_path)
            if os.path.exists(candidate):
                resolved_path = candidate
                break

        import comfy.diffusers_load

        out = comfy.diffusers_load.load_diffusers(
            resolved_path,
            output_vae=True,
            output_clip=True,
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
        )
        return (out[0], out[1], out[2], model_path)

    def _load_diffusion_model(self, diffusion_model_name, weight_dtype):
        name = str(diffusion_model_name or "").strip()
        if not name:
            raise Exception("No diffusion model selected.")

        model_path, resolved_name = _resolve_model_file("diffusion_models", name, "diffusion_models")
        model = comfy.sd.load_diffusion_model(
            model_path,
            model_options=_weight_dtype_model_options(weight_dtype),
        )
        # Diffusion-model only loaders do not include CLIP/VAE payloads.
        return (model, None, None, resolved_name)

    def _load_unet(self, unet_name, weight_dtype):
        name = str(unet_name or "").strip()
        if not name:
            raise Exception("No UNet selected.")

        model_path, resolved_name = _resolve_model_file("unet", name, "unet")
        model = comfy.sd.load_diffusion_model(
            model_path,
            model_options=_weight_dtype_model_options(weight_dtype),
        )
        # UNet-only loaders do not include CLIP/VAE payloads.
        return (model, None, None, resolved_name)

    def _load_gguf(self, gguf_name):
        name = str(gguf_name or "").strip()
        if not name:
            raise Exception("No GGUF diffusion model selected.")
        if not name.lower().endswith(".gguf"):
            raise Exception("GGUF mode requires a .gguf diffusion model.")

        available = _discover_gguf_diffusion_models()
        resolved_name = next((entry for entry in available if entry.lower() == name.lower()), None)
        if resolved_name is None:
            leaf = name.replace("\\", "/").split("/")[-1].lower()
            matches = [entry for entry in available if entry.replace("\\", "/").split("/")[-1].lower() == leaf]
            if len(matches) == 1:
                resolved_name = matches[0]
        if resolved_name is None:
            raise FileNotFoundError(f"GGUF diffusion model '{name}' was not found in ComfyUI model paths.")

        try:
            import nodes as comfy_nodes
            loader_class = comfy_nodes.NODE_CLASS_MAPPINGS.get("UnetLoaderGGUF")
        except Exception:
            loader_class = None
        if loader_class is None:
            raise RuntimeError("ComfyUI-GGUF is required to load GGUF diffusion models.")

        loaded = loader_class().load_unet(resolved_name)
        model = loaded[0] if isinstance(loaded, (tuple, list)) and loaded else None
        if model is None:
            raise RuntimeError(f"ComfyUI-GGUF could not load '{resolved_name}'.")
        return (model, None, None, resolved_name)

    def load_model(
        self,
        model_type,
        checkpoint_name,
        diffusers_model,
        diffusion_model_name,
        unet_name,
        gguf_name,
        weight_dtype,
    ):
        mode = str(model_type or "checkpoint").strip().lower()
        if mode in ("unets", "unet_model"):
            mode = "unet"
        if mode not in ("checkpoint", "diffusers", "diffusion_model", "unet", "gguf"):
            mode = "checkpoint"

        checkpoint_name = _sanitize_model_selector(checkpoint_name, empty_tokens=("", "[none]", "none"))
        diffusers_model = _sanitize_model_selector(diffusers_model)
        diffusion_model_name = _sanitize_model_selector(diffusion_model_name)
        unet_name = _sanitize_model_selector(unet_name)
        gguf_name = _sanitize_model_selector(gguf_name)

        if mode == "checkpoint":
            if not checkpoint_name:
                checkpoint_name = _first_filename_or_empty("checkpoints")
            return self._load_checkpoint(checkpoint_name)
        if mode == "diffusers":
            if not diffusers_model:
                diffusers_model = (_discover_diffusers_models() or [""])[0]
            return self._load_diffusers(diffusers_model)
        if mode == "unet":
            if not unet_name:
                unet_name = str(diffusion_model_name or "").strip() or _first_filename_or_empty("unet")
            return self._load_unet(unet_name, weight_dtype)
        if mode == "gguf":
            name = str(gguf_name or "").strip() or str(diffusion_model_name or "").strip()
            if not name:
                name = (_discover_gguf_diffusion_models() or [""])[0]
            return self._load_gguf(name)
        return self._load_diffusion_model(diffusion_model_name, weight_dtype)


class UmbraPowerPrompter:
    """
    Unified Power Prompter node.
    Loads the target model, resolves prompt text/seed/resolution, applies prompt
    LoRAs, and exposes the simple outputs needed by samplers, detailers, hires
    fix chains, and UmbraLabSaveImage.
    """

    LEGACY_PREFIX_PATTERN = re.compile(r"^\s*\[(?:x|X| )?\]\s*")

    @classmethod
    def INPUT_TYPES(cls):
        checkpoints = _with_legacy_choices(_safe_filename_list("checkpoints"), extras=["", "[None]"])
        diffusion_models = _with_legacy_choices(_safe_filename_list("diffusion_models"), extras=["", "[None]"])
        unet_models = _with_legacy_choices(_safe_filename_list("unet"), extras=["", "[None]"])
        gguf_models = _with_legacy_choices(_discover_gguf_diffusion_models(), extras=["", "default", "[None]"])
        diffusers_models = _with_legacy_choices(_discover_diffusers_models(), extras=["", "[None]"])

        return {
            "required": {
                "prompt_text": ("STRING", {"default": "", "multiline": True}),
                "negative_prompt": ("STRING", {"default": "", "multiline": True}),
                "model_type": (
                    ["checkpoint", "diffusers", "diffusion_model", "unet", "gguf"],
                    {"default": "checkpoint"},
                ),
                "checkpoint_name": (checkpoints,),
                "diffusers_model": (diffusers_models,),
                "diffusion_model_name": (diffusion_models,),
                "unet_name": (unet_models,),
                "gguf_name": (gguf_models,),
                "weight_dtype": (
                    ["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2", "False", "True", "false", "true"],
                    {"default": "default"},
                ),
                "seed": ("INT", {"default": 0, "min": 0, "max": SEED_MAX}),
                "control_after_generate": (
                    ["fixed", "increment", "decrement", "randomize", "True", "False", "true", "false", "1", "0"],
                    {"default": "fixed"},
                ),
                "style_seed_behavior": (["normal", "same_seed_style_cycle"], {"default": "normal"}),
                "aspect_ratio": (
                    _POWER_PROMPTER_ASPECT_RATIO_OPTIONS,
                    {"default": "SDXL - 1:1 square 1024x1024"},
                ),
                "swap_dimensions": (["Off", "On"], {"default": "Off"}),
                "width": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 8}),
                "height": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 8}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 64}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "sampler_name": (_get_sampler_names(),),
                "scheduler": (_get_scheduler_names(),),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "clip_skip": ("INT", {"default": 1, "min": 1, "max": 12, "step": 1}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = (
        "MODEL",
        "CLIP",
        "VAE",
        "CONDITIONING",
        "CONDITIONING",
        "LATENT",
        "STRING",
        "STRING",
        "STRING",
        "INT",
        "INT",
        "FLOAT",
        comfy.samplers.KSampler.SAMPLERS,
        comfy.samplers.KSampler.SCHEDULERS,
        "FLOAT",
        "INT",
        "INT",
        "INT",
    )
    RETURN_NAMES = (
        "model",
        "clip",
        "vae",
        "positive",
        "negative",
        "empty_latent",
        "prompt_text",
        "negative_prompt_text",
        "model_name",
        "seed",
        "steps",
        "cfg",
        "sampler_name",
        "scheduler",
        "denoise",
        "width",
        "height",
        "batch_size",
    )
    FUNCTION = "build"
    CATEGORY = "Umbra"
    DESCRIPTION = "One-node Power Prompter entry point for API workflows. Outputs model, conditioning, latent, prompt text, and generation metadata."

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        return True

    @classmethod
    def _normalize_prompt_text(cls, value):
        return cls.LEGACY_PREFIX_PATTERN.sub("", str(value or "")).strip()

    @classmethod
    def _normalize_control_mode(cls, control_after_generate):
        if isinstance(control_after_generate, bool):
            return "increment" if control_after_generate else "fixed"
        raw = str(control_after_generate or "fixed").strip().lower()
        if raw in ("true", "1", "yes", "on"):
            return "increment"
        if raw in ("false", "0", "no", "off"):
            return "fixed"
        if raw in ("fixed", "increment", "decrement", "randomize"):
            return raw
        return "fixed"

    @classmethod
    def _repeat_index(cls, key):
        index = int(_POWER_PROMPTER_SEED_COUNTERS.get(key, 0))
        _POWER_PROMPTER_SEED_COUNTERS[key] = index + 1
        if len(_POWER_PROMPTER_SEED_COUNTERS) > _POWER_PROMPTER_SEED_COUNTERS_LIMIT:
            for stale_key in list(_POWER_PROMPTER_SEED_COUNTERS.keys())[:-_POWER_PROMPTER_SEED_COUNTERS_LIMIT]:
                _POWER_PROMPTER_SEED_COUNTERS.pop(stale_key, None)
        return index

    @classmethod
    def _resolve_seed(cls, seed, control_after_generate, increment_step, style_seed_behavior, unique_id=None):
        mode = cls._normalize_control_mode(control_after_generate)
        base_seed = _normalize_seed(seed)
        step = _to_int(increment_step, default=1, minimum=1, maximum=SEED_MAX)
        rng = random.SystemRandom()
        if str(style_seed_behavior or "normal").strip() == "same_seed_style_cycle":
            return base_seed
        if mode == "randomize":
            return rng.randint(1, FRONTEND_SAFE_SEED_MAX)
        if mode == "fixed":
            return base_seed
        payload = {
            "unique_id": str(unique_id or ""),
            "mode": mode,
            "seed": base_seed,
            "step": step,
        }
        key = hashlib.sha1(json.dumps(payload, sort_keys=True, default=str).encode("utf-8", "ignore")).hexdigest()
        idx = cls._repeat_index(key)
        if mode == "decrement":
            return int((base_seed - (idx * step)) % (SEED_MAX + 1))
        return _normalize_seed(base_seed + (idx * step))

    @classmethod
    def _resolve_dimensions(cls, width, height, aspect_ratio, swap_dimensions):
        resolved = _POWER_PROMPTER_ASPECT_RATIO_PRESETS.get(str(aspect_ratio or "").strip())
        if resolved:
            width, height = resolved
        width = _to_int(width, default=1024, minimum=64, maximum=8192)
        height = _to_int(height, default=1024, minimum=64, maximum=8192)
        if str(swap_dimensions or "Off").strip().lower() == "on":
            width, height = height, width
        width = max(64, width - (width % 8))
        height = max(64, height - (height % 8))
        return width, height

    @classmethod
    def _build_empty_latent(cls, width, height, batch_size):
        batch = _to_int(batch_size, default=1, minimum=1, maximum=64)
        latent = torch.zeros([batch, 4, height // 8, width // 8])
        return {"samples": latent, "downscale_ratio_spacial": 8}, batch

    @classmethod
    def _build_conditioning(cls, clip, text):
        if clip is None:
            return []
        tokens = clip.tokenize(text)
        return clip.encode_from_tokens_scheduled(tokens)

    @classmethod
    def _apply_loras_from_prompt(cls, model, clip, prompt_text):
        updated_model = model
        updated_clip = clip
        missing = []
        for tag in _parse_a1111_lora_tags(prompt_text):
            requested_name = str(tag.get("name") or "").strip()
            if not requested_name:
                continue
            resolved_filename, _ = _resolve_lora_path(requested_name)
            if not resolved_filename:
                missing.append(requested_name)
                continue
            try:
                lora_state = _load_lora_state(resolved_filename)
                updated_model, updated_clip = comfy.sd.load_lora_for_models(
                    updated_model,
                    updated_clip,
                    lora_state,
                    _to_float(tag.get("strength_model"), default=1.0),
                    _to_float(tag.get("strength_clip"), default=_to_float(tag.get("strength_model"), default=1.0)),
                )
            except Exception as error:
                logger.warn(f"Failed to apply LoRA '{requested_name}' in UmbraPowerPrompter: {error}")
        if missing:
            logger.warn(f"Missing LoRA(s) in UmbraPowerPrompter: {', '.join(missing)}")
        return updated_model, updated_clip, _strip_a1111_lora_tags(prompt_text)

    def build(
        self,
        prompt_text,
        negative_prompt,
        model_type,
        checkpoint_name,
        diffusers_model,
        diffusion_model_name,
        unet_name,
        gguf_name,
        weight_dtype,
        seed,
        control_after_generate,
        style_seed_behavior,
        aspect_ratio,
        swap_dimensions,
        width,
        height,
        batch_size,
        steps,
        cfg,
        sampler_name,
        scheduler,
        denoise,
        clip=None,
        vae=None,
        clip_skip=1,
        prompt=None,
        unique_id=None,
        increment_step=1,
    ):
        loader = UmbraLoadCheckpoint()
        loaded_model, loaded_clip, loaded_vae, model_name = loader.load_model(
            model_type,
            checkpoint_name,
            diffusers_model,
            diffusion_model_name,
            unet_name,
            gguf_name,
            weight_dtype,
        )
        # Explicit workflow resources are authoritative. Some model-only
        # checkpoints return placeholder CLIP/VAE objects that are non-null but
        # cannot encode or decode; external Anima resources must override them.
        resolved_clip = clip if clip is not None else loaded_clip
        resolved_vae = vae if vae is not None else loaded_vae
        effective_clip_skip = _to_int(clip_skip, default=1, minimum=1, maximum=12)
        if resolved_clip is not None and effective_clip_skip > 1:
            try:
                resolved_clip = resolved_clip.clone()
                resolved_clip.clip_layer(-effective_clip_skip)
            except Exception as error:
                logger.warn(f"UmbraPowerPrompter could not apply CLIP skip {effective_clip_skip}: {error}")
        normalized_prompt = self._normalize_prompt_text(prompt_text)
        normalized_negative = self._normalize_prompt_text(negative_prompt)
        loaded_model, resolved_clip, clean_prompt = self._apply_loras_from_prompt(
            loaded_model,
            resolved_clip,
            normalized_prompt,
        )
        resolved_width, resolved_height = self._resolve_dimensions(width, height, aspect_ratio, swap_dimensions)
        empty_latent, resolved_batch = self._build_empty_latent(resolved_width, resolved_height, batch_size)
        effective_seed = self._resolve_seed(seed, control_after_generate, increment_step, style_seed_behavior, unique_id=unique_id)
        effective_steps = _to_int(steps, default=20, minimum=1)
        effective_cfg = _to_float(cfg, default=7.0)
        effective_sampler = str(sampler_name or "euler")
        effective_scheduler = str(scheduler or "normal")
        effective_denoise = _to_float(denoise, default=1.0, minimum=0.0, maximum=1.0)
        positive = self._build_conditioning(resolved_clip, clean_prompt)
        negative = self._build_conditioning(resolved_clip, normalized_negative)
        return (
            loaded_model,
            resolved_clip,
            resolved_vae,
            positive,
            negative,
            empty_latent,
            clean_prompt,
            normalized_negative,
            model_name,
            int(effective_seed),
            int(effective_steps),
            float(effective_cfg),
            effective_sampler,
            effective_scheduler,
            float(effective_denoise),
            int(resolved_width),
            int(resolved_height),
            int(resolved_batch),
        )

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        mode = cls._normalize_control_mode(kwargs.get("control_after_generate", "fixed"))
        style_seed_mode = str(kwargs.get("style_seed_behavior", "normal")).strip()
        if style_seed_mode != "same_seed_style_cycle" and mode in ("increment", "decrement", "randomize"):
            return float("nan")
        return hashlib.sha1(json.dumps(kwargs, sort_keys=True, default=str).encode("utf-8", "ignore")).hexdigest()


_UMBRA_RESOURCE_LOCK = threading.RLock()
_UMBRA_DETAILER_DETECTORS = {}
_UMBRA_SAM_MODELS = {}
_UMBRA_UPSCALE_MODELS = {}


def _get_registered_node_class(class_type):
    try:
        import nodes as comfy_nodes
    except Exception as exc:
        raise RuntimeError("ComfyUI's node registry is unavailable.") from exc

    node_class = getattr(comfy_nodes, "NODE_CLASS_MAPPINGS", {}).get(class_type)
    if node_class is None:
        raise RuntimeError(
            f"Required ComfyUI node '{class_type}' is unavailable. "
            "Install or update ComfyUI Impact Pack and Impact Subpack, then restart ComfyUI."
        )
    return node_class


def _load_umbra_detailer_detector(model_name):
    normalized_name = str(model_name or "").strip().replace("\\", "/")
    if not normalized_name:
        raise RuntimeError("No detector model was selected for an Umbra UI detailer stage.")
    with _UMBRA_RESOURCE_LOCK:
        cached = _UMBRA_DETAILER_DETECTORS.get(normalized_name)
        if cached is not None:
            return cached
        provider_class = _get_registered_node_class("UltralyticsDetectorProvider")
        provider = provider_class()
        try:
            detector_output = provider.doit(normalized_name)
        except Exception as exc:
            raise RuntimeError(f"Umbra UI could not load detector '{normalized_name}'.") from exc
        segm_detector = detector_output[1]
        if not callable(getattr(segm_detector, "detect", None)):
            segm_detector = None
        detector = {
            "bbox": detector_output[0],
            "segm": segm_detector,
            "model_name": normalized_name,
        }
        _UMBRA_DETAILER_DETECTORS[normalized_name] = detector
        logger.info(f"Loaded Umbra UI detailer detector: {normalized_name}")
        return detector


def _load_umbra_sam_model(model_name, device_mode):
    normalized_name = str(model_name or "").strip().replace("\\", "/")
    if not normalized_name:
        return None
    normalized_device = str(device_mode or "AUTO").strip()
    if normalized_device not in ("AUTO", "Prefer GPU", "CPU"):
        normalized_device = "AUTO"
    cache_key = f"{normalized_name}|{normalized_device}"
    with _UMBRA_RESOURCE_LOCK:
        cached = _UMBRA_SAM_MODELS.get(cache_key)
        if cached is not None:
            return cached
        sam_loader_class = _get_registered_node_class("SAMLoader")
        try:
            sam_model = sam_loader_class().load_model(normalized_name, normalized_device)[0]
        except Exception as exc:
            raise RuntimeError(f"Umbra UI could not load SAM model '{normalized_name}'.") from exc
        _UMBRA_SAM_MODELS[cache_key] = sam_model
        logger.info(f"Loaded Umbra UI SAM model: {normalized_name} ({normalized_device})")
        return sam_model


def _preferred_upscale_model_choices():
    choices = _safe_filename_list("upscale_models")
    preferred_name = "4x-AnimeSharp.pth"
    preferred = [name for name in choices if str(name).lower() == preferred_name.lower()]
    remaining = [name for name in choices if name not in preferred]
    ordered = preferred + remaining
    return ordered or [preferred_name]


def _load_umbra_upscale_model(model_name):
    normalized_name = str(model_name or "").strip()
    if not normalized_name:
        raise RuntimeError("No upscale model was selected for Umbra UI.")

    with _UMBRA_RESOURCE_LOCK:
        cached = _UMBRA_UPSCALE_MODELS.get(normalized_name)
        if cached is not None:
            return cached

        try:
            from comfy_extras.nodes_upscale_model import UpscaleModelLoader
            upscale_model = UpscaleModelLoader.load_model(normalized_name)[0]
        except Exception as exc:
            raise RuntimeError(f"Umbra UI could not load upscale model '{normalized_name}'.") from exc

        _UMBRA_UPSCALE_MODELS[normalized_name] = upscale_model
        logger.info(f"Loaded Umbra UI upscale model: {normalized_name}")
        return upscale_model


class UmbraImageDetailer:
    """Umbra-owned ordered detailer pipeline with dynamically loaded resources."""

    DETECTOR_MODELS = {
        "person": "segm/person_yolov8m-seg.pt",
        "face": "bbox/face_yolov8m.pt",
        "eyes": "bbox/Eyes.pt",
        "hands": "bbox/hand_yolov8s.pt",
    }

    STAGE_SETTINGS = {
        "person": {
            "guide_size": 1024,
            "guide_size_for": True,
            "max_size": 1536,
            "steps": 8,
            "denoise": 0.18,
            "feather": 10,
            "bbox_threshold": 0.5,
            "bbox_dilation": 10,
            "bbox_crop_factor": 2.2,
            "drop_size": 10,
            "noise_mask_feather": 24,
            "wildcard": "[CONCAT] coherent anatomy, natural body proportions, coherent clothing folds",
        },
        "face": {
            "guide_size": 512,
            "guide_size_for": False,
            "max_size": 1024,
            "steps": 8,
            "denoise": 0.18,
            "feather": 5,
            "bbox_threshold": 0.5,
            "bbox_dilation": 10,
            "bbox_crop_factor": 2.5,
            "drop_size": 10,
            "noise_mask_feather": 20,
            "wildcard": "",
        },
        "eyes": {
            "guide_size": 384,
            "guide_size_for": True,
            "max_size": 512,
            "steps": 7,
            "denoise": 0.16,
            "feather": 4,
            "bbox_threshold": 0.4,
            "bbox_dilation": 5,
            "bbox_crop_factor": 2.4,
            "drop_size": 4,
            "noise_mask_feather": 12,
            "wildcard": "[CONCAT] detailed symmetrical eyes, sharp irises, natural pupils",
        },
        "hands": {
            "guide_size": 512,
            "guide_size_for": True,
            "max_size": 768,
            "steps": 10,
            "denoise": 0.28,
            "feather": 10,
            "bbox_threshold": 0.35,
            "bbox_dilation": 14,
            "bbox_crop_factor": 2.8,
            "drop_size": 10,
            "noise_mask_feather": 20,
            "wildcard": "[CONCAT] detailed hands, anatomically correct hands, five fingers, natural finger spacing",
        },
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "seed": ("INT", {"default": 0, "min": 0, "max": SEED_MAX}),
            },
            "optional": {
                "pipeline_json": ("STRING", {"default": "", "multiline": True, "dynamicPrompts": False}),
                "person_detail": ("BOOLEAN", {"default": True, "label_on": "enabled", "label_off": "disabled"}),
                "face_detail": ("BOOLEAN", {"default": True, "label_on": "enabled", "label_off": "disabled"}),
                "eye_detail": ("BOOLEAN", {"default": True, "label_on": "enabled", "label_off": "disabled"}),
                "hand_detail": ("BOOLEAN", {"default": True, "label_on": "enabled", "label_off": "disabled"}),
                "sampling_provider": (DETAILER_SAMPLING_PROVIDER_TYPE,),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "detail_report")
    FUNCTION = "refine"
    CATEGORY = "Umbra/UI"
    DESCRIPTION = "Runs an ordered JSON pipeline of any number of independently configured detailer stages."

    @classmethod
    def _legacy_pipeline(cls, person_detail, face_detail, eye_detail, hand_detail):
        flags = {
            "person": _to_bool(person_detail, default=True),
            "face": _to_bool(face_detail, default=True),
            "eyes": _to_bool(eye_detail, default=True),
            "hands": _to_bool(hand_detail, default=True),
        }
        stages = []
        for offset, stage_name in enumerate(("person", "face", "eyes", "hands"), start=1):
            settings = cls.STAGE_SETTINGS[stage_name]
            stages.append({
                "id": f"detail-{stage_name}",
                "enabled": flags[stage_name],
                "label": stage_name.title(),
                "detectorModel": cls.DETECTOR_MODELS[stage_name],
                "guideSize": settings["guide_size"],
                "guideSizeFor": "bbox" if settings["guide_size_for"] else "crop_region",
                "maxSize": settings["max_size"],
                "seedOffset": offset,
                "steps": settings["steps"],
                "cfg": 4.0,
                "samplerName": "er_sde",
                "scheduler": "simple",
                "denoise": settings["denoise"],
                "feather": settings["feather"],
                "noiseMask": True,
                "forceInpaint": True,
                "bboxThreshold": settings["bbox_threshold"],
                "bboxDilation": settings["bbox_dilation"],
                "bboxCropFactor": settings["bbox_crop_factor"],
                "useSam": True,
                "samModel": "sam_vit_b_01ec64.pth",
                "samDeviceMode": "AUTO",
                "samDetectionHint": "center-1",
                "samDilation": 0,
                "samThreshold": 0.93,
                "samBboxExpansion": 0,
                "samMaskHintThreshold": 0.7,
                "samMaskHintUseNegative": "False",
                "dropSize": settings["drop_size"],
                "wildcard": settings["wildcard"],
                "cycle": 1,
                "noiseMaskFeather": settings["noise_mask_feather"],
                "tiledEncode": False,
                "tiledDecode": False,
            })
        return stages

    @classmethod
    def _normalize_stage(cls, raw_stage, index):
        raw = raw_stage if isinstance(raw_stage, dict) else {}
        label = str(raw.get("label") or f"Detailer {index + 1}").strip()[:80] or f"Detailer {index + 1}"
        detector_name = str(raw.get("detectorModel") or "").strip().replace("\\", "/")
        preset_key = next((
            key for key, value in cls.DETECTOR_MODELS.items()
            if value.lower() == detector_name.lower() or key == label.lower()
        ), "face")
        preset = cls.STAGE_SETTINGS[preset_key]
        sam_device = str(raw.get("samDeviceMode") or "AUTO").strip()
        if sam_device not in ("AUTO", "Prefer GPU", "CPU"):
            sam_device = "AUTO"
        sam_hint = str(raw.get("samDetectionHint") or "center-1").strip()
        if sam_hint not in ("center-1", "horizontal-2", "vertical-2", "rect-4", "diamond-4", "mask-area", "mask-points", "mask-point-bbox", "none"):
            sam_hint = "center-1"
        sam_negative = str(raw.get("samMaskHintUseNegative") or "False").strip()
        if sam_negative not in ("False", "Small", "Outter"):
            sam_negative = "False"
        guide_mode = "crop_region" if str(raw.get("guideSizeFor") or "bbox").strip().lower() == "crop_region" else "bbox"
        return {
            "id": str(raw.get("id") or f"detail-stage-{index + 1}").strip()[:128],
            "enabled": _to_bool(raw.get("enabled"), default=True),
            "label": label,
            "detector_model": detector_name or cls.DETECTOR_MODELS[preset_key],
            "guide_size": _to_int(raw.get("guideSize"), default=preset["guide_size"], minimum=64, maximum=16384),
            "guide_size_for": guide_mode,
            "max_size": _to_int(raw.get("maxSize"), default=preset["max_size"], minimum=64, maximum=16384),
            "seed_offset": _to_int(raw.get("seedOffset"), default=index + 1, minimum=0, maximum=1000000),
            "steps": _to_int(raw.get("steps"), default=preset["steps"], minimum=1, maximum=10000),
            "cfg": _to_float(raw.get("cfg"), default=4.0, minimum=0.0, maximum=100.0),
            "sampler_name": str(raw.get("samplerName") or "er_sde").strip() or "er_sde",
            "scheduler": str(raw.get("scheduler") or "simple").strip() or "simple",
            "denoise": _to_float(raw.get("denoise"), default=preset["denoise"], minimum=0.0001, maximum=1.0),
            "feather": _to_int(raw.get("feather"), default=preset["feather"], minimum=0, maximum=100),
            "noise_mask": _to_bool(raw.get("noiseMask"), default=True),
            "force_inpaint": _to_bool(raw.get("forceInpaint"), default=True),
            "bbox_threshold": _to_float(raw.get("bboxThreshold"), default=preset["bbox_threshold"], minimum=0.0, maximum=1.0),
            "bbox_dilation": _to_int(raw.get("bboxDilation"), default=preset["bbox_dilation"], minimum=-512, maximum=512),
            "bbox_crop_factor": _to_float(raw.get("bboxCropFactor"), default=preset["bbox_crop_factor"], minimum=1.0, maximum=10.0),
            "use_sam": _to_bool(raw.get("useSam"), default=True),
            "sam_model": str(raw.get("samModel") or "sam_vit_b_01ec64.pth").strip().replace("\\", "/"),
            "sam_device_mode": sam_device,
            "sam_detection_hint": sam_hint,
            "sam_dilation": _to_int(raw.get("samDilation"), default=0, minimum=-512, maximum=512),
            "sam_threshold": _to_float(raw.get("samThreshold"), default=0.93, minimum=0.0, maximum=1.0),
            "sam_bbox_expansion": _to_int(raw.get("samBboxExpansion"), default=0, minimum=0, maximum=1000),
            "sam_mask_hint_threshold": _to_float(raw.get("samMaskHintThreshold"), default=0.7, minimum=0.0, maximum=1.0),
            "sam_mask_hint_use_negative": sam_negative,
            "drop_size": _to_int(raw.get("dropSize"), default=preset["drop_size"], minimum=1, maximum=16384),
            "wildcard": str(raw.get("wildcard") or ""),
            "cycle": _to_int(raw.get("cycle"), default=1, minimum=1, maximum=10),
            "noise_mask_feather": _to_int(raw.get("noiseMaskFeather"), default=preset["noise_mask_feather"], minimum=0, maximum=100),
            "tiled_encode": _to_bool(raw.get("tiledEncode"), default=False),
            "tiled_decode": _to_bool(raw.get("tiledDecode"), default=False),
        }

    @classmethod
    def _parse_pipeline(cls, pipeline_json, person_detail, face_detail, eye_detail, hand_detail):
        text = str(pipeline_json or "").strip()
        if not text:
            raw_pipeline = cls._legacy_pipeline(person_detail, face_detail, eye_detail, hand_detail)
        else:
            try:
                payload = json.loads(text)
            except Exception as exc:
                raise RuntimeError("Umbra UI detailer pipeline JSON is invalid.") from exc
            raw_pipeline = payload.get("stages") if isinstance(payload, dict) else payload
            if not isinstance(raw_pipeline, list):
                raise RuntimeError("Umbra UI detailer pipeline must be a JSON array.")
        return [cls._normalize_stage(stage, index) for index, stage in enumerate(raw_pipeline)]

    @classmethod
    def _run_stage(cls, stage, image, model, clip, vae, positive, negative, seed, detailer):
        detector = _load_umbra_detailer_detector(stage["detector_model"])
        sam_model = _load_umbra_sam_model(stage["sam_model"], stage["sam_device_mode"]) if stage["use_sam"] else None
        result = detailer.doit(
            image=image,
            model=model,
            clip=clip,
            vae=vae,
            guide_size=stage["guide_size"],
            guide_size_for=stage["guide_size_for"] == "bbox",
            max_size=stage["max_size"],
            seed=_normalize_seed(int(seed) + stage["seed_offset"]),
            steps=stage["steps"],
            cfg=stage["cfg"],
            sampler_name=stage["sampler_name"],
            scheduler=stage["scheduler"],
            positive=positive,
            negative=negative,
            denoise=stage["denoise"],
            feather=stage["feather"],
            noise_mask=stage["noise_mask"],
            force_inpaint=stage["force_inpaint"],
            bbox_threshold=stage["bbox_threshold"],
            bbox_dilation=stage["bbox_dilation"],
            bbox_crop_factor=stage["bbox_crop_factor"],
            sam_detection_hint=stage["sam_detection_hint"],
            sam_dilation=stage["sam_dilation"],
            sam_threshold=stage["sam_threshold"],
            sam_bbox_expansion=stage["sam_bbox_expansion"],
            sam_mask_hint_threshold=stage["sam_mask_hint_threshold"],
            sam_mask_hint_use_negative=stage["sam_mask_hint_use_negative"],
            drop_size=stage["drop_size"],
            bbox_detector=detector["bbox"],
            wildcard=stage["wildcard"],
            cycle=stage["cycle"],
            sam_model_opt=sam_model,
            segm_detector_opt=detector["segm"],
            inpaint_model=False,
            noise_mask_feather=stage["noise_mask_feather"],
            tiled_encode=stage["tiled_encode"],
            tiled_decode=stage["tiled_decode"],
        )
        return result[0], detector["model_name"], stage["sam_model"] if sam_model is not None else ""

    @classmethod
    def _run_stage_native(cls, stage, image, clip, vae, positive, negative, seed, sampling_provider):
        detector = _load_umbra_detailer_detector(stage["detector_model"])
        sam_model = _load_umbra_sam_model(stage["sam_model"], stage["sam_device_mode"]) if stage["use_sam"] else None
        result = run_native_detailer(
            image=image,
            provider=sampling_provider,
            clip=clip,
            vae=vae,
            guide_size=stage["guide_size"],
            guide_size_for_bbox=stage["guide_size_for"] == "bbox",
            max_size=stage["max_size"],
            seed=_normalize_seed(int(seed) + stage["seed_offset"]),
            steps=stage["steps"],
            cfg=stage["cfg"],
            positive=positive,
            negative=negative,
            denoise=stage["denoise"],
            feather=stage["feather"],
            noise_mask=stage["noise_mask"],
            force_inpaint=stage["force_inpaint"],
            bbox_threshold=stage["bbox_threshold"],
            bbox_dilation=stage["bbox_dilation"],
            bbox_crop_factor=stage["bbox_crop_factor"],
            sam_detection_hint=stage["sam_detection_hint"],
            sam_dilation=stage["sam_dilation"],
            sam_threshold=stage["sam_threshold"],
            sam_bbox_expansion=stage["sam_bbox_expansion"],
            sam_mask_hint_threshold=stage["sam_mask_hint_threshold"],
            sam_mask_hint_use_negative=stage["sam_mask_hint_use_negative"],
            drop_size=stage["drop_size"],
            bbox_detector=detector["bbox"],
            wildcard=stage["wildcard"],
            cycle=stage["cycle"],
            sam_model=sam_model,
            segm_detector=detector["segm"],
            inpaint_model=False,
            noise_mask_feather=stage["noise_mask_feather"],
            tiled_encode=stage["tiled_encode"],
            tiled_decode=stage["tiled_decode"],
        )
        return result, detector["model_name"], stage["sam_model"] if sam_model is not None else ""

    def refine(
        self,
        image,
        model,
        clip,
        vae,
        positive,
        negative,
        seed,
        pipeline_json="",
        person_detail=True,
        face_detail=True,
        eye_detail=True,
        hand_detail=True,
        sampling_provider=None,
    ):
        pipeline = self._parse_pipeline(pipeline_json, person_detail, face_detail, eye_detail, hand_detail)
        enabled_stages = [stage for stage in pipeline if stage["enabled"]]
        current = image
        completed = []
        if enabled_stages and sampling_provider is None:
            detailer = _get_registered_node_class("FaceDetailer")()
        for stage in enabled_stages:
            current, detector_name, sam_name = dispatch_detailer_stage(
                sampling_provider,
                classic_call=lambda: self._run_stage(
                    stage,
                    current,
                    model,
                    clip,
                    vae,
                    positive,
                    negative,
                    seed,
                    detailer,
                ),
                native_call=lambda provider: self._run_stage_native(
                    stage,
                    current,
                    clip,
                    vae,
                    positive,
                    negative,
                    seed,
                    provider,
                ),
            )
            completed_stage = {
                "id": stage["id"],
                "label": stage["label"],
                "detector": detector_name,
                "sam": sam_name,
                "cycle": stage["cycle"],
            }
            if sampling_provider is not None:
                completed_stage["samplingProvider"] = sampling_provider.provider_id
            completed.append(completed_stage)

        report = json.dumps({
            "profile": "umbra-dynamic-v1",
            "completed": completed,
            "configured": len(pipeline),
            "enabled": len(enabled_stages),
        }, separators=(",", ":"))
        return (current, report)


class UmbraImageUpscale:
    """Model-based upscale with a true maximum-dimension cap."""

    @classmethod
    def INPUT_TYPES(cls):
        choices = _preferred_upscale_model_choices()
        default_model = "4x-AnimeSharp.pth" if "4x-AnimeSharp.pth" in choices else choices[0]
        return {
            "required": {
                "image": ("IMAGE",),
                "upscale_model": (choices, {"default": default_model}),
                "max_dimension": ("INT", {"default": 3840, "min": 512, "max": 16384, "step": 8}),
            },
            "optional": {
                "enabled": ("BOOLEAN", {"default": False, "label_on": "enabled", "label_off": "disabled"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT", "STRING")
    RETURN_NAMES = ("image", "width", "height", "upscale_model")
    FUNCTION = "upscale"
    CATEGORY = "Umbra/UI"
    DESCRIPTION = "Optionally runs a selected model-based output upscale, then caps the largest output dimension."

    def upscale(self, image, upscale_model, max_dimension, enabled=True):
        if not _to_bool(enabled, default=True):
            height = int(image.shape[-3])
            width = int(image.shape[-2])
            return (image, width, height, "disabled")
        try:
            from comfy_extras.nodes_upscale_model import ImageUpscaleWithModel
            loaded_model = _load_umbra_upscale_model(upscale_model)
            upscaled = ImageUpscaleWithModel.upscale(loaded_model, image)[0]
        except Exception as exc:
            if isinstance(exc, RuntimeError) and str(exc).startswith("Umbra UI"):
                raise
            raise RuntimeError("Umbra UI failed while applying its upscale model.") from exc

        height = int(upscaled.shape[1])
        width = int(upscaled.shape[2])
        largest = max(width, height)
        limit = max(512, int(max_dimension))

        if largest > limit:
            scale = limit / float(largest)
            target_width = max(1, round(width * scale))
            target_height = max(1, round(height * scale))
            samples = upscaled.movedim(-1, 1)
            samples = comfy_utils.common_upscale(samples, target_width, target_height, "lanczos", "disabled")
            upscaled = samples.movedim(1, -1)
            width = target_width
            height = target_height

        return (upscaled, width, height, str(upscale_model))


class UmbraFrameInterpolate:
    """Runs ComfyUI's native interpolator with a reliable low-VRAM device handoff."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "interp_model": ("INTERP_MODEL",),
                "images": ("IMAGE",),
                "multiplier": ("INT", {"default": 2, "min": 2, "max": 16, "step": 1}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "interpolate"
    CATEGORY = "Umbra/UI"
    DESCRIPTION = "Runs ComfyUI's native frame interpolation with low-VRAM-safe model placement."

    def interpolate(self, interp_model, images, multiplier):
        from tqdm import tqdm

        num_frames = int(images.shape[0])
        multiplier = int(multiplier)
        if num_frames < 2 or multiplier < 2:
            return (images,)

        height = int(images.shape[1])
        width = int(images.shape[2])
        activation_memory = height * width * 3 * images.element_size() * 20
        model_management.load_model_gpu(interp_model)
        model_management.free_memory(activation_memory, interp_model.load_device)

        inference_model = interp_model.model
        device = interp_model.load_device
        offload_device = interp_model.offload_device
        dtype = interp_model.model_dtype()
        inference_model.to(device=device, dtype=dtype)
        align = getattr(inference_model, "pad_align", 1)

        def prepare_frame(index):
            frame = images[index:index + 1].movedim(-1, 1).to(dtype=dtype, device=device)
            if align > 1:
                from comfy.ldm.common_dit import pad_to_patch_size
                frame = pad_to_patch_size(frame, (align, align), padding_mode="reflect")
            return frame

        total_pairs = num_frames - 1
        interpolation_count = multiplier - 1
        progress = comfy_utils.ProgressBar(total_pairs * interpolation_count)
        tqdm_bar = tqdm(total=total_pairs * interpolation_count, desc="Umbra frame interpolation")
        batch = interpolation_count
        time_values = [time_index / multiplier for time_index in range(1, multiplier)]
        output_dtype = model_management.intermediate_dtype()
        output = torch.empty(
            (total_pairs * multiplier + 1, 3, height, width),
            dtype=output_dtype,
            device=model_management.intermediate_device(),
        )
        output[0] = images[0].movedim(-1, 0).to(output_dtype)
        output_index = 1

        sample = prepare_frame(0)
        padded_height, padded_width = int(sample.shape[2]), int(sample.shape[3])
        timesteps = torch.tensor(time_values, device=device, dtype=dtype).reshape(interpolation_count, 1, 1, 1)
        timesteps = timesteps.expand(-1, 1, padded_height, padded_width)
        del sample

        multi_timestep = getattr(inference_model, "forward_multi_timestep", None)
        feature_cache = {}
        previous_frame = None
        try:
            for pair_index in range(total_pairs):
                first_frame = previous_frame if previous_frame is not None else prepare_frame(pair_index)
                second_frame = prepare_frame(pair_index + 1)
                previous_frame = second_frame
                feature_cache["img0"] = (
                    feature_cache.pop("next")
                    if "next" in feature_cache
                    else inference_model.extract_features(first_frame)
                )
                feature_cache["img1"] = inference_model.extract_features(second_frame)
                feature_cache["next"] = feature_cache["img1"]

                used_multi_timestep = False
                if multi_timestep is not None:
                    try:
                        middle_frames = multi_timestep(first_frame, second_frame, time_values, cache=feature_cache)
                        output[output_index:output_index + interpolation_count] = (
                            middle_frames[:, :, :height, :width].to(output_dtype)
                        )
                        output_index += interpolation_count
                        progress.update(interpolation_count)
                        tqdm_bar.update(interpolation_count)
                        used_multi_timestep = True
                    except model_management.OOM_EXCEPTION:
                        model_management.soft_empty_cache()
                        multi_timestep = None

                if not used_multi_timestep:
                    interpolation_index = 0
                    while interpolation_index < interpolation_count:
                        current_batch = min(batch, interpolation_count - interpolation_index)
                        try:
                            first_batch = first_frame.expand(current_batch, -1, -1, -1)
                            second_batch = second_frame.expand(current_batch, -1, -1, -1)
                            middle_frames = inference_model(
                                first_batch,
                                second_batch,
                                timestep=timesteps[interpolation_index:interpolation_index + current_batch],
                                cache=feature_cache,
                            )
                            output[output_index:output_index + current_batch] = (
                                middle_frames[:, :, :height, :width].to(output_dtype)
                            )
                            output_index += current_batch
                            progress.update(current_batch)
                            tqdm_bar.update(current_batch)
                            interpolation_index += current_batch
                        except model_management.OOM_EXCEPTION:
                            if batch <= 1:
                                raise
                            batch = max(1, batch // 2)
                            model_management.soft_empty_cache()

                output[output_index] = images[pair_index + 1].movedim(-1, 0).to(output_dtype)
                output_index += 1
        finally:
            tqdm_bar.close()
            inference_model.to(device=offload_device)
            model_management.soft_empty_cache()

        return (output.movedim(1, -1).clamp_(0.0, 1.0),)


class UmbraVideoUpscale:
    """Frame-at-a-time video upscale or exact target-size fit."""

    @classmethod
    def INPUT_TYPES(cls):
        choices = _preferred_upscale_model_choices()
        default_model = "4x-AnimeSharp.pth" if "4x-AnimeSharp.pth" in choices else choices[0]
        return {
            "required": {
                "images": ("IMAGE",),
                "mode": (["lanczos", "model"], {"default": "lanczos"}),
                "scale_by": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 4.0, "step": 0.25}),
                "max_dimension": ("INT", {"default": 3840, "min": 512, "max": 4096, "step": 8}),
            },
            "optional": {
                "upscale_model": (choices, {"default": default_model}),
                "target_width": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 8}),
                "target_height": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 8}),
                "enabled": ("BOOLEAN", {"default": True, "label_on": "enabled", "label_off": "disabled"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT", "STRING")
    RETURN_NAMES = ("images", "width", "height", "method")
    FUNCTION = "upscale"
    CATEGORY = "Umbra/UI"
    DESCRIPTION = "Processes video frames one at a time and lands on an exact target size when provided."

    @staticmethod
    def _target_size(width, height, scale_by, max_dimension, target_width=0, target_height=0):
        exact_width = max(0, min(8192, int(target_width or 0)))
        exact_height = max(0, min(8192, int(target_height or 0)))
        if exact_width > 0 and exact_height > 0:
            return exact_width, exact_height
        scale = max(1.0, min(4.0, float(scale_by)))
        target_width = max(1, round(int(width) * scale))
        target_height = max(1, round(int(height) * scale))
        limit = max(512, min(4096, int(max_dimension)))
        largest = max(target_width, target_height)
        if largest > limit:
            cap_scale = limit / float(largest)
            target_width = max(1, round(target_width * cap_scale))
            target_height = max(1, round(target_height * cap_scale))
        return target_width, target_height

    def upscale(
        self,
        images,
        mode,
        scale_by,
        max_dimension,
        upscale_model=None,
        target_width=0,
        target_height=0,
        enabled=True,
    ):
        batch_size = int(images.shape[0])
        source_height = int(images.shape[1])
        source_width = int(images.shape[2])
        if not _to_bool(enabled, default=True) or batch_size <= 0:
            return (images, source_width, source_height, "disabled")

        normalized_mode = str(mode or "lanczos").strip().lower()
        if normalized_mode not in ("lanczos", "model"):
            normalized_mode = "lanczos"
        target_width, target_height = self._target_size(
            source_width,
            source_height,
            scale_by,
            max_dimension,
            target_width,
            target_height,
        )

        loaded_model = None
        method = "lanczos"
        if normalized_mode == "model":
            loaded_model = _load_umbra_upscale_model(upscale_model)
            method = str(upscale_model or "model")

        out_device = model_management.intermediate_device()
        out_dtype = model_management.intermediate_dtype()
        output = torch.empty(
            (batch_size, target_height, target_width, int(images.shape[3])),
            device=out_device,
            dtype=out_dtype,
        )

        model_upscaler = None
        if loaded_model is not None:
            from comfy_extras.nodes_upscale_model import ImageUpscaleWithModel
            model_upscaler = ImageUpscaleWithModel()

        for frame_index in range(batch_size):
            frame = images[frame_index:frame_index + 1]
            if model_upscaler is not None:
                frame = model_upscaler.upscale(loaded_model, frame)[0]
            current_height = int(frame.shape[1])
            current_width = int(frame.shape[2])
            if current_width != target_width or current_height != target_height:
                samples = frame.movedim(-1, 1)
                samples = comfy_utils.common_upscale(
                    samples,
                    target_width,
                    target_height,
                    "lanczos",
                    "disabled",
                )
                frame = samples.movedim(1, -1)
            output[frame_index].copy_(frame[0].to(device=out_device, dtype=out_dtype), non_blocking=False)

        return (output, target_width, target_height, method)


class UmbraSoftInpaintComposite:
    """Blend an inpaint result into its source without ghosting the replacement.

    The painted mask interior is always generated at full opacity. Only the
    feathered mask boundary blends with the source image. Earlier versions used
    image-difference strength as alpha across the entire mask, which could leave
    two partially visible subjects in the saved result.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "original": ("IMAGE",),
                "generated": ("IMAGE",),
                "mask": ("MASK",),
                "preservation": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "transition_contrast": ("FLOAT", {"default": 2.0, "min": 0.25, "max": 8.0, "step": 0.05}),
                "mask_influence": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "blend_mask")
    FUNCTION = "composite"
    CATEGORY = "Umbra UI/Inpaint"

    @staticmethod
    def _resize_image(image, width, height):
        if int(image.shape[1]) == height and int(image.shape[2]) == width:
            return image
        resized = torch.nn.functional.interpolate(
            image.movedim(-1, 1),
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )
        return resized.movedim(1, -1)

    @staticmethod
    def _resize_mask(mask, width, height):
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        elif mask.ndim == 4:
            mask = mask[:, 0]
        if int(mask.shape[-2]) == height and int(mask.shape[-1]) == width:
            return mask
        return torch.nn.functional.interpolate(
            mask.unsqueeze(1),
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)

    @staticmethod
    def _match_batch(tensor, batch_size):
        current = int(tensor.shape[0])
        if current == batch_size:
            return tensor
        if current == 1:
            return tensor.expand((batch_size,) + tuple(tensor.shape[1:]))
        indices = torch.arange(batch_size, device=tensor.device) % current
        return tensor.index_select(0, indices)

    def composite(self, original, generated, mask, preservation, transition_contrast, mask_influence):
        if original.ndim != 4 or generated.ndim != 4:
            raise ValueError("Soft inpaint expects IMAGE tensors in BHWC format.")

        height = int(original.shape[1])
        width = int(original.shape[2])
        generated = self._resize_image(generated, width, height)
        mask = self._resize_mask(mask, width, height)
        batch_size = max(int(original.shape[0]), int(generated.shape[0]), int(mask.shape[0]))
        original = self._match_batch(original, batch_size)
        generated = self._match_batch(generated, batch_size)
        mask = self._match_batch(mask, batch_size)

        output_device = original.device
        output_dtype = original.dtype
        original_f = original.to(device=output_device, dtype=torch.float32).clamp(0.0, 1.0)
        generated_f = generated.to(device=output_device, dtype=torch.float32).clamp(0.0, 1.0)
        mask_f = mask.to(device=output_device, dtype=torch.float32).clamp(0.0, 1.0)

        preservation = max(0.0, min(1.0, float(preservation)))
        contrast = max(0.25, min(8.0, float(transition_contrast)))
        influence = max(0.0, min(1.0, float(mask_influence)))

        # Preservation controls how much of the feather belongs to the source,
        # not the opacity of the generated interior. The threshold remains well
        # below a fully painted mask value, so all solid mask pixels become an
        # opaque replacement regardless of the selected preservation value.
        opaque_threshold = 0.35 + preservation * 0.40
        normalized_edge = (mask_f / opaque_threshold).clamp(0.0, 1.0)
        contrasted_edge = ((normalized_edge - 0.5) * contrast + 0.5).clamp(0.0, 1.0)
        smooth_edge = contrasted_edge * contrasted_edge * contrasted_edge * (
            contrasted_edge * (contrasted_edge * 6.0 - 15.0) + 10.0
        )
        edge_alpha = smooth_edge * (1.0 - influence) + normalized_edge * influence
        blend_mask = torch.where(mask_f >= opaque_threshold, torch.ones_like(mask_f), edge_alpha)
        alpha = blend_mask.unsqueeze(-1)
        output = original_f * (1.0 - alpha) + generated_f * alpha
        return (output.to(dtype=output_dtype).clamp(0.0, 1.0), blend_mask)


# Node registration for ComfyUI
NODE_CLASS_MAPPINGS = {
    "UmbraPowerPrompter": UmbraPowerPrompter,
    "UmbraLabSaveImage": UmbraLabSaveImage,
    "UmbraLabSaveImageSimple": UmbraLabSaveImageSimple,
    "UmbraA1111LoraSyntax": UmbraA1111LoraSyntax,
    "UmbraKSampler": UmbraKSampler,
    "UmbraKSamplerNormal": UmbraKSamplerNormal,
    "UmbraKSamplerHiResFix": UmbraKSamplerHiResFix,
    "UmbraCFGValue": UmbraCFGValue,
    "UmbraStepsValue": UmbraStepsValue,
    "UmbraSeedValue": UmbraSeedValue,
    "UmbraLoadCheckpoint": UmbraLoadCheckpoint,
    "UmbraImageDetailer": UmbraImageDetailer,
    "UmbraFlux2DetailerSamplingProvider": UmbraFlux2DetailerProviderNode,
    "UmbraHiDreamO1DetailerSamplingProvider": UmbraHiDreamO1DetailerProviderNode,
    "UmbraIdeogram4DetailerSamplingProvider": UmbraIdeogram4DetailerProviderNode,
    "UmbraOmniGen2DetailerSamplingProvider": UmbraOmniGen2DetailerProviderNode,
    "UmbraImageUpscale": UmbraImageUpscale,
    "UmbraFrameInterpolate": UmbraFrameInterpolate,
    "UmbraVideoUpscale": UmbraVideoUpscale,
    "UmbraSoftInpaintComposite": UmbraSoftInpaintComposite,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "UmbraPowerPrompter": "Power Prompter (Umbra Lab)",
    "UmbraLabSaveImage": "Save Image (Umbra Lab)",
    "UmbraLabSaveImageSimple": "Save Image Simple (Umbra Lab)",
    "UmbraA1111LoraSyntax": "A1111 LoRA Syntax (Umbra Lab)",
    "UmbraKSampler": "KSampler (Umbra Lab)",
    "UmbraKSamplerNormal": "KSampler Normal (Umbra Lab)",
    "UmbraKSamplerHiResFix": "KSampler Hires Fix (Umbra UI)",
    "UmbraCFGValue": "CFG Value (Umbra Lab)",
    "UmbraStepsValue": "Steps Value (Umbra Lab)",
    "UmbraSeedValue": "Seed Value (Umbra Lab)",
    "UmbraLoadCheckpoint": "Load Checkpoint (Umbra Lab)",
    "UmbraImageDetailer": "Image Detailer (Umbra UI)",
    "UmbraFlux2DetailerSamplingProvider": "Flux.2 Detailer Sampling Provider (Umbra)",
    "UmbraHiDreamO1DetailerSamplingProvider": "HiDream O1 Detailer Sampling Provider (Umbra)",
    "UmbraIdeogram4DetailerSamplingProvider": "Ideogram 4 Detailer Sampling Provider (Umbra)",
    "UmbraOmniGen2DetailerSamplingProvider": "OmniGen2 Detailer Sampling Provider (Umbra)",
    "UmbraImageUpscale": "Image Upscale (Umbra UI)",
    "UmbraFrameInterpolate": "Frame Interpolate (Umbra UI)",
    "UmbraVideoUpscale": "Video Upscale (Umbra UI)",
    "UmbraSoftInpaintComposite": "Soft Inpaint Composite (Umbra UI)",
}
