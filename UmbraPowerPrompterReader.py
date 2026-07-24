import hashlib
import json
import random
import re

import torch

try:
    import comfy.model_management as comfy_model_management
except Exception:
    comfy_model_management = None


SEED_MAX = 0xFFFFFFFFFFFFFFFF
FRONTEND_SAFE_SEED_MAX = 9007199254740991
_SEED_COUNTERS = {}
_SEED_COUNTERS_LIMIT = 1024

_ASPECT_RATIO_PRESETS = {
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

_ASPECT_RATIO_OPTIONS = ["custom", *_ASPECT_RATIO_PRESETS.keys()]


class UmbraPowerPrompterReader:
    """
    Power Prompter websocket node.
    Outputs positive/negative conditioning plus an empty latent batch, with integrated seed controls.
    """

    LEGACY_PREFIX_PATTERN = re.compile(r"^\s*\[(?:x|X| )?\]\s*")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt_text": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "Active prompt text synced from Umbra Power Prompter websocket.",
                    },
                ),
                "negative_prompt": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "Negative prompt text synced from Umbra Power Prompter websocket.",
                    },
                ),
                "seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": SEED_MAX,
                        "tooltip": "Base seed value.",
                    },
                ),
                "control_after_generate": (
                    [
                        "fixed",
                        "increment",
                        "decrement",
                        "randomize",
                        "True",
                        "False",
                        "true",
                        "false",
                        "1",
                        "0",
                    ],
                    {
                        "default": "fixed",
                        "tooltip": "How seed changes after each run.",
                    },
                ),
                "increment_step": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": SEED_MAX,
                        "tooltip": "Seed step for increment/decrement.",
                    },
                ),
                "style_seed_behavior": (
                    ["normal", "same_seed_style_cycle"],
                    {
                        "default": "normal",
                        "tooltip": "When same_seed_style_cycle is active, style-expanded prompt jobs reuse the same base seed instead of advancing this websocket node's per-run seed counter.",
                    },
                ),
                "aspect_ratio": (
                    _ASPECT_RATIO_OPTIONS,
                    {
                        "default": "SDXL - 1:1 square 1024x1024",
                        "tooltip": "Resolution presets matching SD1.5 and SDXL aspect ratios.",
                    },
                ),
                "swap_dimensions": (
                    ["Off", "On"],
                    {
                        "default": "Off",
                        "tooltip": "Swap width and height after preset/custom resolution is resolved.",
                    },
                ),
                "width": (
                    "INT",
                    {
                        "default": 1024,
                        "min": 64,
                        "max": 8192,
                        "step": 8,
                        "tooltip": "Used when aspect_ratio is custom.",
                    },
                ),
                "height": (
                    "INT",
                    {
                        "default": 1024,
                        "min": 64,
                        "max": 8192,
                        "step": 8,
                        "tooltip": "Used when aspect_ratio is custom.",
                    },
                ),
                "batch_size": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 64,
                        "tooltip": "Empty latent batch size.",
                    },
                ),
            },
            "optional": {
                "clip": (
                    "CLIP",
                    {
                        "tooltip": "CLIP model for positive conditioning output.",
                    },
                ),
            },
            "hidden": {
                "prompt": "PROMPT",
                "unique_id": "UNIQUE_ID",
            },
        }

    # Keep legacy output indexes stable; append new outputs at the end.
    RETURN_TYPES = ("STRING", "CONDITIONING", "LATENT", "INT", "INT", "INT", "INT", "CONDITIONING", "STRING")
    RETURN_NAMES = (
        "prompt_text",
        "positive",
        "empty_latent",
        "seed",
        "width",
        "height",
        "batch_size",
        "negative",
        "negative_prompt_text",
    )
    FUNCTION = "read_prompt"
    CATEGORY = "Umbra"
    DESCRIPTION = "Power Prompter websocket node with text + positive/negative conditioning outputs, seed controls, and empty latent aspect presets."

    @classmethod
    def _normalize_prompt_text(cls, value):
        return cls.LEGACY_PREFIX_PATTERN.sub("", str(value or "")).strip()

    @classmethod
    def _clamp_seed(cls, seed):
        try:
            seed_value = int(seed)
        except Exception:
            seed_value = 0
        return max(0, min(SEED_MAX, seed_value))

    @classmethod
    def _clamp_step(cls, value):
        try:
            step = int(value)
        except Exception:
            step = 1
        return max(1, min(SEED_MAX, step))

    @classmethod
    def _repeat_index(cls, key):
        index = int(_SEED_COUNTERS.get(key, 0))
        _SEED_COUNTERS[key] = index + 1
        if len(_SEED_COUNTERS) > _SEED_COUNTERS_LIMIT:
            for stale_key in list(_SEED_COUNTERS.keys())[:-_SEED_COUNTERS_LIMIT]:
                _SEED_COUNTERS.pop(stale_key, None)
        return index

    @classmethod
    def _resolve_seed(cls, seed, control_after_generate, increment_step, style_seed_behavior="normal", prompt=None, unique_id=None):
        mode = cls._normalize_control_mode(control_after_generate)
        base_seed = cls._clamp_seed(seed)
        step = cls._clamp_step(increment_step)
        rng = random.SystemRandom()
        if str(style_seed_behavior or "normal").strip() == "same_seed_style_cycle":
            return base_seed

        if mode == "randomize":
            # Keep random seeds positive and JS-safe so UI can display exact values.
            return rng.randint(1, FRONTEND_SAFE_SEED_MAX)
        if mode == "fixed":
            return base_seed

        key_payload = {
            "unique_id": str(unique_id or ""),
            "mode": mode,
            "seed": base_seed,
            "step": step,
        }
        key = hashlib.sha1(json.dumps(key_payload, sort_keys=True, default=str).encode("utf-8", "ignore")).hexdigest()
        idx = cls._repeat_index(key)

        if mode == "decrement":
            # Wrap in 64-bit unsigned range to stay within Comfy seed semantics.
            return int((base_seed - (idx * step)) % (SEED_MAX + 1))

        # increment
        return cls._clamp_seed(base_seed + (idx * step))

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
    def _resolve_dimensions(cls, width, height, aspect_ratio, swap_dimensions):
        resolved = _ASPECT_RATIO_PRESETS.get(str(aspect_ratio or "").strip())
        if resolved:
            width, height = resolved
        try:
            width = int(width)
        except Exception:
            width = 1024
        try:
            height = int(height)
        except Exception:
            height = 1024
        if str(swap_dimensions or "Off").strip().lower() == "on":
            width, height = height, width
        width = max(64, width - (width % 8))
        height = max(64, height - (height % 8))
        return width, height

    @classmethod
    def _build_empty_latent(cls, width, height, batch_size):
        try:
            batch = int(batch_size)
        except Exception:
            batch = 1
        batch = max(1, min(64, batch))

        device = "cpu"
        dtype = torch.float32
        if comfy_model_management is not None:
            try:
                device = comfy_model_management.intermediate_device()
            except Exception:
                device = "cpu"
            try:
                dtype = comfy_model_management.intermediate_dtype()
            except Exception:
                dtype = torch.float32

        latent = torch.zeros([batch, 4, height // 8, width // 8], device=device, dtype=dtype)
        return {"samples": latent, "downscale_ratio_spacial": 8}, batch

    @classmethod
    def _build_conditioning(cls, clip, text):
        if clip is None:
            return []
        tokens = clip.tokenize(text)
        # Match ComfyUI's native CLIPTextEncode path so model-specific encoders can
        # attach any extra conditioning metadata they require.
        return clip.encode_from_tokens_scheduled(tokens)

    def read_prompt(
        self,
        prompt_text,
        negative_prompt,
        seed,
        control_after_generate,
        increment_step,
        style_seed_behavior,
        aspect_ratio,
        swap_dimensions,
        width,
        height,
        batch_size,
        clip=None,
        prompt=None,
        unique_id=None,
    ):
        normalized_prompt = self._normalize_prompt_text(prompt_text)
        normalized_negative_prompt = self._normalize_prompt_text(negative_prompt)
        effective_seed = self._resolve_seed(
            seed=seed,
            control_after_generate=control_after_generate,
            increment_step=increment_step,
            style_seed_behavior=style_seed_behavior,
            prompt=prompt,
            unique_id=unique_id,
        )
        resolved_width, resolved_height = self._resolve_dimensions(width, height, aspect_ratio, swap_dimensions)
        empty_latent, resolved_batch = self._build_empty_latent(resolved_width, resolved_height, batch_size)
        positive = self._build_conditioning(clip, normalized_prompt)
        negative = self._build_conditioning(clip, normalized_negative_prompt)
        return (
            normalized_prompt,
            positive,
            empty_latent,
            int(effective_seed),
            int(resolved_width),
            int(resolved_height),
            int(resolved_batch),
            negative,
            normalized_negative_prompt,
        )

    @classmethod
    def IS_CHANGED(
        cls,
        prompt_text,
        negative_prompt,
        seed,
        control_after_generate,
        increment_step,
        style_seed_behavior,
        aspect_ratio,
        swap_dimensions,
        width,
        height,
        batch_size,
        clip=None,
        prompt=None,
        unique_id=None,
    ):
        del clip
        mode = cls._normalize_control_mode(control_after_generate)
        style_seed_mode = str(style_seed_behavior or "normal").strip()
        if style_seed_mode != "same_seed_style_cycle" and mode in ("increment", "decrement", "randomize"):
            return float("nan")
        payload = {
            "prompt_text": cls._normalize_prompt_text(prompt_text),
            "negative_prompt": cls._normalize_prompt_text(negative_prompt),
            "seed": cls._clamp_seed(seed),
            "control_after_generate": mode,
            "increment_step": cls._clamp_step(increment_step),
            "style_seed_behavior": style_seed_mode,
            "aspect_ratio": str(aspect_ratio or "custom"),
            "swap_dimensions": str(swap_dimensions or "Off"),
            "width": int(width),
            "height": int(height),
            "batch_size": int(batch_size),
            "prompt": prompt,
            "unique_id": unique_id,
        }
        return hashlib.sha1(json.dumps(payload, sort_keys=True, default=str).encode("utf-8", "ignore")).hexdigest()


NODE_CLASS_MAPPINGS = {
    "UmbraPowerPrompterReader": UmbraPowerPrompterReader
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "UmbraPowerPrompterReader": "Power Prompter Websocket"
}
