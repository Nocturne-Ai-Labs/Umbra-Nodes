"""Architecture-native per-crop sampling for Umbra's image detailer.

The classic detailer path remains owned by Impact Pack. This module is only
used when a graph explicitly supplies an Umbra detailer sampling provider.
"""

from __future__ import annotations

import importlib
import inspect
import logging
import math
import numbers
from dataclasses import dataclass, field, replace
from typing import Any, Callable, ClassVar, Protocol, runtime_checkable


DETAILER_SAMPLING_PROVIDER_TYPE = "UMBRA_DETAILER_SAMPLING_PROVIDER"
ConditioningCropper = Callable[[Any], Any]


class ImpactFaceDetailerUnavailableError(RuntimeError):
    """Impact Pack's FaceDetailer node is not registered in ComfyUI."""


class ImpactDetailerCompatibilityError(RuntimeError):
    """The loaded Impact Pack/ComfyUI APIs cannot support the native bridge."""


def _identity_conditioning(conditioning):
    return conditioning


@dataclass(frozen=True)
class DetailerSamplingContext:
    """Inputs that only become authoritative after a detail crop is prepared."""

    model: Any
    positive: Any
    negative: Any
    latent: dict
    seed: int
    steps: int
    cfg: float
    denoise: float
    width: int
    height: int
    crop_conditioning: ConditioningCropper = field(
        default=_identity_conditioning,
        repr=False,
        compare=False,
    )


@runtime_checkable
class DetailerSamplingProviderProtocol(Protocol):
    provider_id: str
    model: Any

    def prepare_crop_size(self, width: int, height: int) -> tuple[int, int]: ...

    def detailer_conditionings(self, positive, negative) -> tuple[Any, Any]: ...

    def sample_crop(self, context: DetailerSamplingContext) -> dict: ...


def validate_sampling_provider(provider):
    required_methods = ("prepare_crop_size", "detailer_conditionings", "sample_crop")
    problems = [
        f"callable {name}()"
        for name in required_methods
        if not callable(getattr(provider, name, None))
    ]
    if not isinstance(getattr(provider, "provider_id", None), str) or not provider.provider_id.strip():
        problems.append("non-empty string provider_id")
    if not hasattr(provider, "model") or provider.model is None:
        problems.append("non-None model")
    if problems:
        raise RuntimeError(
            "Invalid Umbra detailer sampling provider: missing or invalid "
            + ", ".join(problems)
            + "."
        )
    return provider


def _prepare_provider_crop_size(provider, width, height):
    prepared = provider.prepare_crop_size(width, height)
    if not isinstance(prepared, (tuple, list)) or len(prepared) != 2:
        raise RuntimeError(
            f"Umbra detailer provider '{provider.provider_id}' returned an invalid crop size; "
            "expected (width, height)."
        )
    if any(isinstance(value, bool) or not isinstance(value, numbers.Integral) for value in prepared):
        raise RuntimeError(
            f"Umbra detailer provider '{provider.provider_id}' returned an invalid crop size; "
            "width and height must be integers."
        )
    prepared_width, prepared_height = (int(value) for value in prepared)
    if prepared_width <= 0 or prepared_height <= 0:
        raise RuntimeError(
            f"Umbra detailer provider '{provider.provider_id}' returned an invalid crop size; "
            "width and height must be positive."
        )
    return prepared_width, prepared_height


def _prepare_provider_conditionings(provider, positive, negative):
    conditionings = provider.detailer_conditionings(positive, negative)
    if not isinstance(conditionings, (tuple, list)) or len(conditionings) != 2:
        raise RuntimeError(
            f"Umbra detailer provider '{provider.provider_id}' returned invalid detailer conditionings; "
            "expected (positive, negative)."
        )
    return conditionings[0], conditionings[1]


def dispatch_detailer_stage(provider, classic_call, native_call):
    """Keep the provider-free call site an exact pass-through to Impact Pack."""

    if provider is None:
        return classic_call()
    return native_call(validate_sampling_provider(provider))


def sample_prepared_crop(provider, context, cycles=1):
    """Sample one prepared crop, rebuilding provider components every cycle."""

    provider = validate_sampling_provider(provider)
    refined_latent = context.latent
    for cycle_index in range(max(1, int(cycles))):
        cycle_context = replace(
            context,
            latent=refined_latent,
            seed=int(context.seed) + cycle_index,
        )
        refined_latent = provider.sample_crop(cycle_context)
        if (
            not isinstance(refined_latent, dict)
            or "samples" not in refined_latent
            or refined_latent["samples"] is None
        ):
            raise RuntimeError(
                f"Umbra detailer provider '{provider.provider_id}' did not return a valid LATENT "
                f"dictionary for cycle {cycle_index + 1}."
            )
    return refined_latent


def _sampler_names():
    import comfy.samplers

    return tuple(getattr(comfy.samplers, "SAMPLER_NAMES", comfy.samplers.KSampler.SAMPLERS))


def _scheduler_names():
    import comfy.samplers

    return tuple(getattr(comfy.samplers, "SCHEDULER_NAMES", comfy.samplers.KSampler.SCHEDULERS))


def _resolution_sigmas(builder, steps, denoise):
    steps = max(1, int(steps))
    denoise = max(0.0001, min(1.0, float(denoise)))
    total_steps = max(steps, math.floor(steps / denoise))
    sigmas = builder(total_steps)
    return sigmas[-(steps + 1):]


class ComfySamplingBackend:
    """Thin calls into the installed public Comfy sampling nodes."""

    @staticmethod
    def random_noise(seed):
        from comfy_extras.nodes_custom_sampler import RandomNoise

        return RandomNoise.execute(int(seed))[0]

    @staticmethod
    def basic_guider(model, conditioning):
        from comfy_extras.nodes_custom_sampler import BasicGuider

        return BasicGuider.execute(model, conditioning)[0]

    @staticmethod
    def cfg_guider(model, positive, negative, cfg):
        from comfy_extras.nodes_custom_sampler import CFGGuider

        return CFGGuider.execute(model, positive, negative, float(cfg))[0]

    @staticmethod
    def dual_model_guider(model, model_negative, positive, negative, cfg):
        from comfy_extras.nodes_custom_sampler import DualModelGuider

        return DualModelGuider.execute(
            model,
            positive,
            float(cfg),
            model_negative=model_negative,
            negative=negative,
        )[0]

    @staticmethod
    def dual_cfg_guider(model, cond1, cond2, negative, cfg_conds, cfg_cond2_negative, style):
        from comfy_extras.nodes_custom_sampler import DualCFGGuider

        return DualCFGGuider.execute(
            model,
            cond1,
            cond2,
            negative,
            float(cfg_conds),
            float(cfg_cond2_negative),
            style,
        )[0]

    @staticmethod
    def sampler_select(sampler_name):
        from comfy_extras.nodes_custom_sampler import KSamplerSelect

        return KSamplerSelect.execute(str(sampler_name))[0]

    @staticmethod
    def lcm_sampler(s_noise, s_noise_end, noise_clip_std):
        from comfy_extras.nodes_advanced_samplers import SamplerLCM

        return SamplerLCM.execute(
            float(s_noise),
            float(s_noise_end),
            float(noise_clip_std),
        )[0]

    @staticmethod
    def basic_sigmas(model, scheduler, steps, denoise):
        from comfy_extras.nodes_custom_sampler import BasicScheduler

        return BasicScheduler.execute(model, scheduler, int(steps), float(denoise))[0]

    @staticmethod
    def flux2_sigmas(steps, width, height):
        from comfy_extras.nodes_flux import Flux2Scheduler

        return Flux2Scheduler.execute(int(steps), int(width), int(height))[0]

    @staticmethod
    def ideogram4_sigmas(steps, width, height, mu, std):
        from comfy_extras.nodes_ideogram4 import Ideogram4Scheduler

        return Ideogram4Scheduler.execute(
            int(steps),
            int(width),
            int(height),
            float(mu),
            float(std),
        )[0]

    @staticmethod
    def sample_advanced(noise, guider, sampler, sigmas, latent):
        from comfy_extras.nodes_custom_sampler import SamplerCustomAdvanced

        return SamplerCustomAdvanced.execute(noise, guider, sampler, sigmas, latent)[0]

    @staticmethod
    def patch_model_noise_scale(model, noise_scale):
        from comfy_extras.nodes_model_advanced import ModelNoiseScale

        return ModelNoiseScale().patch(model, float(noise_scale))[0]


@dataclass
class DetailerSamplingProvider:
    model: Any
    backend: Any = field(default=None, repr=False, compare=False)

    provider_id: ClassVar[str] = "native"
    dimension_multiple: ClassVar[int] = 8

    def __post_init__(self):
        if self.backend is None:
            self.backend = ComfySamplingBackend()

    def prepare_crop_size(self, width, height):
        multiple = max(1, int(self.dimension_multiple))

        def align(value):
            value = max(multiple, int(value))
            return max(multiple, (value // multiple) * multiple)

        return align(width), align(height)

    def detailer_conditionings(self, positive, negative):
        return positive, negative


@dataclass
class Flux2DetailerSamplingProvider(DetailerSamplingProvider):
    sampler_name: str = "euler"

    provider_id: ClassVar[str] = "flux2"
    dimension_multiple: ClassVar[int] = 16

    def sample_crop(self, context):
        noise = self.backend.random_noise(context.seed)
        guider = self.backend.basic_guider(context.model, context.positive)
        sampler = self.backend.sampler_select(self.sampler_name)
        sigmas = _resolution_sigmas(
            lambda total_steps: self.backend.flux2_sigmas(
                total_steps,
                context.width,
                context.height,
            ),
            context.steps,
            context.denoise,
        )
        return self.backend.sample_advanced(noise, guider, sampler, sigmas, context.latent)


@dataclass
class HiDreamO1DetailerSamplingProvider(DetailerSamplingProvider):
    scheduler: str = "normal"
    s_noise: float = 1.0
    s_noise_end: float = 1.0
    noise_clip_std: float = 2.5

    provider_id: ClassVar[str] = "hidream_o1"
    dimension_multiple: ClassVar[int] = 32

    def sample_crop(self, context):
        noise = self.backend.random_noise(context.seed)
        guider = self.backend.cfg_guider(
            context.model,
            context.positive,
            context.negative,
            context.cfg,
        )
        sampler = self.backend.lcm_sampler(
            self.s_noise,
            self.s_noise_end,
            self.noise_clip_std,
        )
        sigmas = self.backend.basic_sigmas(
            context.model,
            self.scheduler,
            context.steps,
            context.denoise,
        )
        return self.backend.sample_advanced(noise, guider, sampler, sigmas, context.latent)


@dataclass
class Ideogram4DetailerSamplingProvider(DetailerSamplingProvider):
    model_negative: Any = None
    sampler_name: str = "euler"
    mu: float = 0.5
    std: float = 1.75

    provider_id: ClassVar[str] = "ideogram4"
    dimension_multiple: ClassVar[int] = 16

    def sample_crop(self, context):
        noise = self.backend.random_noise(context.seed)
        guider = self.backend.dual_model_guider(
            context.model,
            self.model_negative,
            context.positive,
            context.negative,
            context.cfg,
        )
        sampler = self.backend.sampler_select(self.sampler_name)
        sigmas = _resolution_sigmas(
            lambda total_steps: self.backend.ideogram4_sigmas(
                total_steps,
                context.width,
                context.height,
                self.mu,
                self.std,
            ),
            context.steps,
            context.denoise,
        )
        return self.backend.sample_advanced(noise, guider, sampler, sigmas, context.latent)


@dataclass
class OmniGen2DetailerSamplingProvider(DetailerSamplingProvider):
    cond1: Any = None
    cond2: Any = None
    negative: Any = None
    cfg_conds: float = 5.0
    cfg_cond2_negative: float = 2.0
    style: str = "regular"
    sampler_name: str = "euler"
    scheduler: str = "simple"

    provider_id: ClassVar[str] = "omnigen2"
    dimension_multiple: ClassVar[int] = 16

    def detailer_conditionings(self, positive, negative):
        return self.cond1, self.negative

    def sample_crop(self, context):
        noise = self.backend.random_noise(context.seed)
        guider = self.backend.dual_cfg_guider(
            context.model,
            context.positive,
            context.crop_conditioning(self.cond2),
            context.negative,
            self.cfg_conds,
            self.cfg_cond2_negative,
            self.style,
        )
        sampler = self.backend.sampler_select(self.sampler_name)
        sigmas = self.backend.basic_sigmas(
            context.model,
            self.scheduler,
            context.steps,
            context.denoise,
        )
        return self.backend.sample_advanced(noise, guider, sampler, sigmas, context.latent)


class UmbraFlux2DetailerProviderNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "sampler_name": (_sampler_names(), {"default": "euler"}),
            }
        }

    RETURN_TYPES = (DETAILER_SAMPLING_PROVIDER_TYPE,)
    RETURN_NAMES = ("sampling_provider",)
    FUNCTION = "build"
    CATEGORY = "Umbra/Detailer/Providers"
    DESCRIPTION = "Builds per-crop Flux.2 BasicGuider, sampler, and Flux2Scheduler sampling."

    def build(self, model, sampler_name):
        return (Flux2DetailerSamplingProvider(model=model, sampler_name=sampler_name),)


class UmbraHiDreamO1DetailerProviderNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "noise_scale": ("FLOAT", {"default": 7.5, "min": 0.0, "max": 64.0, "step": 0.01}),
                "scheduler": (_scheduler_names(), {"default": "normal"}),
                "s_noise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 64.0, "step": 0.01}),
                "s_noise_end": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 64.0, "step": 0.01}),
                "noise_clip_std": ("FLOAT", {"default": 2.5, "min": 0.0, "max": 10.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = (DETAILER_SAMPLING_PROVIDER_TYPE,)
    RETURN_NAMES = ("sampling_provider",)
    FUNCTION = "build"
    CATEGORY = "Umbra/Detailer/Providers"
    DESCRIPTION = "Applies ModelNoiseScale and builds per-crop HiDream O1 LCM custom sampling."

    def build(self, model, noise_scale, scheduler, s_noise, s_noise_end, noise_clip_std):
        backend = ComfySamplingBackend()
        patched_model = backend.patch_model_noise_scale(model, noise_scale)
        return (
            HiDreamO1DetailerSamplingProvider(
                model=patched_model,
                scheduler=scheduler,
                s_noise=s_noise,
                s_noise_end=s_noise_end,
                noise_clip_std=noise_clip_std,
                backend=backend,
            ),
        )


class UmbraIdeogram4DetailerProviderNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "model_negative": ("MODEL",),
                "sampler_name": (_sampler_names(), {"default": "euler"}),
                "mu": ("FLOAT", {"default": 0.5, "min": -10.0, "max": 10.0, "step": 0.05}),
                "std": ("FLOAT", {"default": 1.75, "min": 0.1, "max": 5.0, "step": 0.05}),
            }
        }

    RETURN_TYPES = (DETAILER_SAMPLING_PROVIDER_TYPE,)
    RETURN_NAMES = ("sampling_provider",)
    FUNCTION = "build"
    CATEGORY = "Umbra/Detailer/Providers"
    DESCRIPTION = "Builds per-crop Ideogram 4 dual-model guided custom sampling."

    def build(self, model, model_negative, sampler_name, mu, std):
        return (
            Ideogram4DetailerSamplingProvider(
                model=model,
                model_negative=model_negative,
                sampler_name=sampler_name,
                mu=mu,
                std=std,
            ),
        )


class UmbraOmniGen2DetailerProviderNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "cond1": ("CONDITIONING",),
                "cond2": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "cfg_conds": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "cfg_cond2_negative": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "style": (["regular", "nested"], {"default": "regular"}),
                "sampler_name": (_sampler_names(), {"default": "euler"}),
                "scheduler": (_scheduler_names(), {"default": "simple"}),
            }
        }

    RETURN_TYPES = (DETAILER_SAMPLING_PROVIDER_TYPE,)
    RETURN_NAMES = ("sampling_provider",)
    FUNCTION = "build"
    CATEGORY = "Umbra/Detailer/Providers"
    DESCRIPTION = "Builds per-crop OmniGen2 DualCFGGuider custom sampling."

    def build(
        self,
        model,
        cond1,
        cond2,
        negative,
        cfg_conds,
        cfg_cond2_negative,
        style,
        sampler_name,
        scheduler,
    ):
        return (
            OmniGen2DetailerSamplingProvider(
                model=model,
                cond1=cond1,
                cond2=cond2,
                negative=negative,
                cfg_conds=cfg_conds,
                cfg_cond2_negative=cfg_cond2_negative,
                style=style,
                sampler_name=sampler_name,
                scheduler=scheduler,
            ),
        )


@dataclass(frozen=True)
class _ImpactApi:
    face_detailer_class: type
    segs_scale_match: Callable
    make_sam_mask: Callable
    segs_bitwise_and_mask: Callable
    segs_to_combined_mask: Callable
    crop_condition_mask: Callable
    tensor_resize: Callable
    tensor_gaussian_blur_mask: Callable
    apply_differential_diffusion: Callable
    to_latent_image: Callable
    crop_ndarray4: Callable
    to_tensor: Callable
    tensor_paste: Callable
    tensor_convert_rgb: Callable
    process_with_loras: Callable
    process_wildcard_for_segs: Callable
    conditioning_concat_class: type
    inpaint_model_conditioning_class: type
    vae_decode_tiled_class: type
    concat_tensors: Callable


def _impact_compatibility_error(detail):
    return ImpactDetailerCompatibilityError(
        "Umbra native detailer compatibility error: FaceDetailer is registered, but the "
        f"installed Impact Pack/ComfyUI API is incompatible: {detail}. Update Impact Pack "
        "and ComfyUI to compatible versions."
    )


def _validate_call_shape(value, qualified_name, positional_count, keyword_names=()):
    try:
        signature = inspect.signature(value)
    except (TypeError, ValueError):
        return value

    args = [None] * positional_count
    kwargs = {name: None for name in keyword_names}
    try:
        signature.bind(*args, **kwargs)
    except TypeError as exc:
        invocation = f"{positional_count} positional argument(s)"
        if keyword_names:
            invocation += f" and keyword(s) {', '.join(keyword_names)}"
        raise _impact_compatibility_error(
            f"{qualified_name} does not accept the required call shape ({invocation})"
        ) from exc
    return value


def _require_callable(module, module_name, symbol, positional_count, keyword_names=()):
    value = getattr(module, symbol, None)
    qualified_name = f"{module_name}.{symbol}"
    if not callable(value):
        raise _impact_compatibility_error(f"missing callable {qualified_name}")
    return _validate_call_shape(
        value,
        qualified_name,
        positional_count,
        keyword_names,
    )


def _validate_class(value, qualified_name, method_shapes=()):
    if not inspect.isclass(value):
        raise _impact_compatibility_error(f"missing class {qualified_name}")
    try:
        constructor_signature = inspect.signature(value)
    except (TypeError, ValueError):
        constructor_signature = None
    if constructor_signature is not None:
        try:
            constructor_signature.bind()
        except TypeError as exc:
            raise _impact_compatibility_error(
                f"{qualified_name} cannot be constructed without arguments"
            ) from exc

    for method_name, positional_count, keyword_names in method_shapes:
        method = getattr(value, method_name, None)
        if not callable(method):
            raise _impact_compatibility_error(
                f"missing callable {qualified_name}.{method_name}"
            )
        _validate_call_shape(
            method,
            f"{qualified_name}.{method_name}",
            positional_count,
            keyword_names,
        )
    return value


def _require_class(module, module_name, symbol, method_shapes=()):
    return _validate_class(
        getattr(module, symbol, None),
        f"{module_name}.{symbol}",
        method_shapes,
    )


def _import_impact_module(module_name):
    try:
        return importlib.import_module(module_name)
    except Exception as exc:
        raise _impact_compatibility_error(
            f"failed to import required module {module_name!r} "
            f"({type(exc).__name__}: {exc})"
        ) from exc


def _load_impact_api():
    try:
        comfy_nodes = importlib.import_module("nodes")
    except Exception as exc:
        raise ImpactFaceDetailerUnavailableError(
            "Umbra native detailer sampling requires ComfyUI Impact Pack's FaceDetailer, "
            "but the ComfyUI nodes module could not be imported."
        ) from exc

    mappings = getattr(comfy_nodes, "NODE_CLASS_MAPPINGS", None)
    if mappings is None or "FaceDetailer" not in mappings:
        raise ImpactFaceDetailerUnavailableError(
            "Umbra native detailer sampling requires ComfyUI Impact Pack's FaceDetailer, "
            "but 'FaceDetailer' is not registered. Install or enable Impact Pack and restart ComfyUI."
        )

    face_detailer_class = _validate_class(
        mappings["FaceDetailer"],
        "nodes.NODE_CLASS_MAPPINGS['FaceDetailer']",
    )
    if not callable(getattr(face_detailer_class, "doit", None)):
        raise _impact_compatibility_error(
            "nodes.NODE_CLASS_MAPPINGS['FaceDetailer'] has no callable doit method"
        )

    core = _import_impact_module("impact.core")
    utils = _import_impact_module("impact.utils")
    wildcards = _import_impact_module("impact.wildcards")
    torch_module = _import_impact_module("torch")

    conditioning_concat_class = _require_class(
        comfy_nodes,
        "nodes",
        "ConditioningConcat",
        (("concat", 3, ()),),
    )
    inpaint_model_conditioning_class = _require_class(
        comfy_nodes,
        "nodes",
        "InpaintModelConditioning",
    )
    inpaint_encode = getattr(inpaint_model_conditioning_class, "encode", None)
    if not callable(inpaint_encode):
        raise _impact_compatibility_error(
            "missing callable nodes.InpaintModelConditioning.encode"
        )
    try:
        has_noise_mask_parameter = "noise_mask" in inspect.signature(inpaint_encode).parameters
    except (TypeError, ValueError):
        has_noise_mask_parameter = True
    if has_noise_mask_parameter:
        _validate_call_shape(
            inpaint_encode,
            "nodes.InpaintModelConditioning.encode",
            5,
            ("mask", "noise_mask"),
        )
    else:
        _validate_call_shape(
            inpaint_encode,
            "nodes.InpaintModelConditioning.encode",
            6,
        )

    vae_decode_tiled_class = _require_class(
        comfy_nodes,
        "nodes",
        "VAEDecodeTiled",
        (("decode", 4, ()),),
    )

    return _ImpactApi(
        face_detailer_class=face_detailer_class,
        segs_scale_match=_require_callable(core, "impact.core", "segs_scale_match", 2),
        make_sam_mask=_require_callable(core, "impact.core", "make_sam_mask", 9),
        segs_bitwise_and_mask=_require_callable(
            core,
            "impact.core",
            "segs_bitwise_and_mask",
            2,
        ),
        segs_to_combined_mask=_require_callable(
            core,
            "impact.core",
            "segs_to_combined_mask",
            1,
        ),
        crop_condition_mask=_require_callable(
            core,
            "impact.core",
            "crop_condition_mask",
            3,
        ),
        tensor_resize=_require_callable(utils, "impact.utils", "tensor_resize", 3),
        tensor_gaussian_blur_mask=_require_callable(
            utils,
            "impact.utils",
            "tensor_gaussian_blur_mask",
            2,
        ),
        apply_differential_diffusion=_require_callable(
            utils,
            "impact.utils",
            "apply_differential_diffusion",
            1,
        ),
        to_latent_image=_require_callable(
            utils,
            "impact.utils",
            "to_latent_image",
            2,
            ("vae_tiled_encode",),
        ),
        crop_ndarray4=_require_callable(utils, "impact.utils", "crop_ndarray4", 2),
        to_tensor=_require_callable(utils, "impact.utils", "to_tensor", 1),
        tensor_paste=_require_callable(utils, "impact.utils", "tensor_paste", 4),
        tensor_convert_rgb=_require_callable(
            utils,
            "impact.utils",
            "tensor_convert_rgb",
            1,
        ),
        process_with_loras=_require_callable(
            wildcards,
            "impact.wildcards",
            "process_with_loras",
            3,
        ),
        process_wildcard_for_segs=_require_callable(
            wildcards,
            "impact.wildcards",
            "process_wildcard_for_segs",
            1,
        ),
        conditioning_concat_class=conditioning_concat_class,
        inpaint_model_conditioning_class=inpaint_model_conditioning_class,
        vae_decode_tiled_class=vae_decode_tiled_class,
        concat_tensors=_require_callable(torch_module, "torch", "cat", 1, ("dim",)),
    )


def _crop_conditioning(conditioning, image, crop_region, crop_condition_mask):
    if isinstance(conditioning, str) or conditioning is None:
        return conditioning
    return [
        [condition, {
            key: crop_condition_mask(value, image, crop_region) if key == "mask" else value
            for key, value in details.items()
        }]
        for condition, details in conditioning
    ]


def _is_all_zero(mask):
    result = (mask == 0).all()
    return bool(result.item() if hasattr(result, "item") else result)


def _enhance_detail_native(
    api,
    provider,
    image,
    model,
    clip,
    vae,
    guide_size,
    guide_size_for_bbox,
    max_size,
    bbox,
    seed,
    steps,
    cfg,
    positive,
    negative,
    denoise,
    noise_mask,
    force_inpaint,
    wildcard_item,
    wildcard_concat_mode,
    control_net_wrapper,
    cycle,
    inpaint_model,
    noise_mask_feather,
    tiled_encode,
    tiled_decode,
    crop_conditioning,
):
    if noise_mask is not None:
        noise_mask = api.tensor_gaussian_blur_mask(noise_mask, noise_mask_feather)
        noise_mask = noise_mask.squeeze(3)
        if noise_mask_feather > 0 and "denoise_mask_function" not in getattr(model, "model_options", {}):
            model = api.apply_differential_diffusion(model)

    if wildcard_item:
        model, _, wildcard_positive = api.process_with_loras(wildcard_item, model, clip)
        if wildcard_concat_mode == "concat":
            positive = api.conditioning_concat_class().concat(positive, wildcard_positive)[0]
        else:
            positive = [wildcard_positive[0].copy()]
            if "pooled_output" in wildcard_positive[0][1]:
                positive[0][1]["pooled_output"] = wildcard_positive[0][1]["pooled_output"]
            elif "pooled_output" in positive[0][1]:
                del positive[0][1]["pooled_output"]

    height = int(image.shape[1])
    width = int(image.shape[2])
    bbox_height = bbox[3] - bbox[1]
    bbox_width = bbox[2] - bbox[0]

    if not force_inpaint and bbox_height >= guide_size and bbox_width >= guide_size:
        logging.info("Detailer: segment skip (enough big)")
        return None, None

    if guide_size_for_bbox:
        upscale = guide_size / min(bbox_width, bbox_height)
    else:
        upscale = guide_size / min(width, height)

    new_width = int(width * upscale)
    new_height = int(height * upscale)
    if "aitemplate_keep_loaded" in getattr(model, "model_options", {}):
        max_size = min(4096, max_size)
    if new_width > max_size or new_height > max_size:
        upscale *= max_size / max(new_width, new_height)
        new_width = int(width * upscale)
        new_height = int(height * upscale)

    if not force_inpaint:
        if upscale <= 1.0 or new_width == 0 or new_height == 0:
            logging.info("Detailer: segment skip [upscale not required or zero-sized crop]")
            return None, None
    elif upscale <= 1.0 or new_width == 0 or new_height == 0:
        logging.info("Detailer: force inpaint")
        new_width = width
        new_height = height

    new_width, new_height = _prepare_provider_crop_size(provider, new_width, new_height)
    logging.info(
        "Detailer: native segment upscale for (%s, %s) | crop region %s x %s -> %s x %s",
        bbox_width,
        bbox_height,
        width,
        height,
        new_width,
        new_height,
    )
    upscaled_image = api.tensor_resize(image, new_width, new_height)

    cnet_pils = None
    if control_net_wrapper is not None:
        positive, negative, cnet_pils = control_net_wrapper.apply(
            positive,
            negative,
            upscaled_image,
            noise_mask,
        )
        model, cnet_pils2 = control_net_wrapper.doit_ipadapter(model)
        cnet_pils.extend(cnet_pils2)

    if noise_mask is not None and inpaint_model:
        encode = api.inpaint_model_conditioning_class().encode
        if "noise_mask" in inspect.signature(encode).parameters:
            positive, negative, latent = encode(
                positive,
                negative,
                upscaled_image,
                vae,
                mask=noise_mask,
                noise_mask=True,
            )
        else:
            positive, negative, latent = encode(
                positive,
                negative,
                upscaled_image,
                vae,
                noise_mask,
            )
    else:
        latent = api.to_latent_image(
            upscaled_image,
            vae,
            vae_tiled_encode=tiled_encode,
        )
        if noise_mask is not None:
            latent["noise_mask"] = noise_mask

    final_width = int(upscaled_image.shape[2])
    final_height = int(upscaled_image.shape[1])
    context = DetailerSamplingContext(
        model=model,
        positive=positive,
        negative=negative,
        latent=latent,
        seed=int(seed),
        steps=int(steps),
        cfg=float(cfg),
        denoise=float(denoise),
        width=final_width,
        height=final_height,
        crop_conditioning=crop_conditioning,
    )
    refined_latent = sample_prepared_crop(provider, context, cycles=cycle)

    if tiled_decode:
        refined_image = api.vae_decode_tiled_class().decode(vae, refined_latent, 512)[0]
    else:
        try:
            refined_image = vae.decode(refined_latent["samples"])
        except Exception:
            logging.warning("Umbra native detailer VAE decode failed; retrying tiled decode.")
            refined_image = vae.decode_tiled(refined_latent["samples"], tile_x=64, tile_y=64)

    if len(refined_image.shape) == 5:
        refined_image = refined_image.squeeze(0)
    refined_image = api.tensor_resize(refined_image, width, height)
    return refined_image.cpu(), cnet_pils


def _detail_segments_native(
    api,
    provider,
    image,
    segs,
    model,
    clip,
    vae,
    guide_size,
    guide_size_for_bbox,
    max_size,
    seed,
    steps,
    cfg,
    positive,
    negative,
    denoise,
    feather,
    noise_mask,
    force_inpaint,
    wildcard,
    cycle,
    inpaint_model,
    noise_mask_feather,
    tiled_encode,
    tiled_decode,
):
    if len(image) > 1:
        raise RuntimeError("Umbra native detailer sampling does not accept image batches per segment.")

    image = image.clone()
    segs = api.segs_scale_match(segs, image.shape)

    wildcard_concat_mode = None
    if wildcard is not None:
        if wildcard.startswith("[CONCAT]"):
            wildcard_concat_mode = "concat"
            wildcard = wildcard[8:]
        wildcard_mode, wildcard_chooser = api.process_wildcard_for_segs(wildcard)
    else:
        wildcard_mode, wildcard_chooser = None, None

    if wildcard_mode == "ASC":
        ordered_segs = sorted(segs[1], key=lambda item: (item.bbox[0], item.bbox[1]))
    elif wildcard_mode == "DSC":
        ordered_segs = sorted(segs[1], key=lambda item: (item.bbox[0], item.bbox[1]), reverse=True)
    elif wildcard_mode == "ASC-SIZE":
        ordered_segs = sorted(segs[1], key=lambda item: (item.bbox[2] - item.bbox[0]) * (item.bbox[3] - item.bbox[1]))
    elif wildcard_mode == "DSC-SIZE":
        ordered_segs = sorted(
            segs[1],
            key=lambda item: (item.bbox[2] - item.bbox[0]) * (item.bbox[3] - item.bbox[1]),
            reverse=True,
        )
    else:
        ordered_segs = segs[1]

    if noise_mask_feather > 0 and "denoise_mask_function" not in getattr(model, "model_options", {}):
        model = api.apply_differential_diffusion(model)

    for index, seg in enumerate(ordered_segs):
        cropped_image = api.to_tensor(api.crop_ndarray4(image.cpu().numpy(), seg.crop_region))
        paste_mask = api.tensor_gaussian_blur_mask(api.to_tensor(seg.cropped_mask), feather)
        if _is_all_zero(seg.cropped_mask):
            logging.info("Detailer: segment skip [empty mask]")
            continue

        cropped_mask = seg.cropped_mask if noise_mask else None
        if wildcard_chooser is not None and wildcard_mode != "LAB":
            seg_seed, wildcard_item = wildcard_chooser.get(seg)
        elif wildcard_chooser is not None:
            seg_seed, wildcard_item = None, wildcard_chooser.get(seg)
        else:
            seg_seed, wildcard_item = None, None
        seg_seed = int(seed) + index if seg_seed is None else seg_seed

        if wildcard_item and wildcard_item.strip() == "[SKIP]":
            continue
        if wildcard_item and wildcard_item.strip() == "[STOP]":
            break

        cropped_positive = _crop_conditioning(
            positive,
            image,
            seg.crop_region,
            api.crop_condition_mask,
        )
        cropped_negative = _crop_conditioning(
            negative,
            image,
            seg.crop_region,
            api.crop_condition_mask,
        )

        def crop_extra_conditioning(value, source_image=image, region=seg.crop_region):
            return _crop_conditioning(
                value,
                source_image,
                region,
                api.crop_condition_mask,
            )

        enhanced_image, _ = _enhance_detail_native(
            api=api,
            provider=provider,
            image=cropped_image,
            model=model,
            clip=clip,
            vae=vae,
            guide_size=guide_size,
            guide_size_for_bbox=guide_size_for_bbox,
            max_size=max_size,
            bbox=seg.bbox,
            seed=seg_seed,
            steps=steps,
            cfg=cfg,
            positive=cropped_positive,
            negative=cropped_negative,
            denoise=denoise,
            noise_mask=cropped_mask,
            force_inpaint=force_inpaint,
            wildcard_item=wildcard_item,
            wildcard_concat_mode=wildcard_concat_mode,
            control_net_wrapper=seg.control_net_wrapper,
            cycle=cycle,
            inpaint_model=inpaint_model,
            noise_mask_feather=noise_mask_feather,
            tiled_encode=tiled_encode,
            tiled_decode=tiled_decode,
            crop_conditioning=crop_extra_conditioning,
        )
        if enhanced_image is not None:
            image = image.cpu()
            api.tensor_paste(
                image,
                enhanced_image.cpu(),
                (seg.crop_region[0], seg.crop_region[1]),
                paste_mask,
            )

    return api.tensor_convert_rgb(image)


def run_native_detailer(
    *,
    image,
    provider,
    clip,
    vae,
    guide_size,
    guide_size_for_bbox,
    max_size,
    seed,
    steps,
    cfg,
    positive,
    negative,
    denoise,
    feather,
    noise_mask,
    force_inpaint,
    bbox_threshold,
    bbox_dilation,
    bbox_crop_factor,
    sam_detection_hint,
    sam_dilation,
    sam_threshold,
    sam_bbox_expansion,
    sam_mask_hint_threshold,
    sam_mask_hint_use_negative,
    drop_size,
    bbox_detector,
    wildcard,
    cycle,
    sam_model=None,
    segm_detector=None,
    inpaint_model=False,
    noise_mask_feather=0,
    tiled_encode=False,
    tiled_decode=False,
):
    provider = validate_sampling_provider(provider)
    api = _load_impact_api()
    stage_positive, stage_negative = _prepare_provider_conditionings(
        provider,
        positive,
        negative,
    )
    result_image = None

    if len(image) > 1:
        logging.warning("Umbra native detailer is processing an image batch one image at a time.")

    for batch_index, single_image in enumerate(image):
        single_image = single_image.unsqueeze(0)
        bbox_detector.setAux("face")
        try:
            segs = bbox_detector.detect(
                single_image,
                bbox_threshold,
                bbox_dilation,
                bbox_crop_factor,
                drop_size,
            )
        finally:
            bbox_detector.setAux(None)

        if sam_model is not None:
            sam_mask = api.make_sam_mask(
                sam_model,
                segs,
                single_image,
                sam_detection_hint,
                sam_dilation,
                sam_threshold,
                sam_bbox_expansion,
                sam_mask_hint_threshold,
                sam_mask_hint_use_negative,
            )
            segs = api.segs_bitwise_and_mask(segs, sam_mask)
        elif segm_detector is not None:
            segm_segs = segm_detector.detect(
                single_image,
                bbox_threshold,
                bbox_dilation,
                bbox_crop_factor,
                drop_size,
            )
            if getattr(segm_detector, "override_bbox_by_segm", False):
                segs = segm_segs
            else:
                segs = api.segs_bitwise_and_mask(
                    segs,
                    api.segs_to_combined_mask(segm_segs),
                )

        if len(segs[1]) > 0:
            enhanced = _detail_segments_native(
                api=api,
                provider=provider,
                image=single_image,
                segs=segs,
                model=provider.model,
                clip=clip,
                vae=vae,
                guide_size=guide_size,
                guide_size_for_bbox=guide_size_for_bbox,
                max_size=max_size,
                seed=int(seed) + batch_index,
                steps=steps,
                cfg=cfg,
                positive=stage_positive,
                negative=stage_negative,
                denoise=denoise,
                feather=feather,
                noise_mask=noise_mask,
                force_inpaint=force_inpaint,
                wildcard=wildcard,
                cycle=cycle,
                inpaint_model=inpaint_model,
                noise_mask_feather=noise_mask_feather,
                tiled_encode=tiled_encode,
                tiled_decode=tiled_decode,
            )
        else:
            enhanced = single_image

        result_image = (
            api.concat_tensors((result_image, enhanced), dim=0)
            if result_image is not None
            else enhanced
        )

    return result_image


__all__ = [
    "DETAILER_SAMPLING_PROVIDER_TYPE",
    "DetailerSamplingContext",
    "Flux2DetailerSamplingProvider",
    "HiDreamO1DetailerSamplingProvider",
    "Ideogram4DetailerSamplingProvider",
    "OmniGen2DetailerSamplingProvider",
    "UmbraFlux2DetailerProviderNode",
    "UmbraHiDreamO1DetailerProviderNode",
    "UmbraIdeogram4DetailerProviderNode",
    "UmbraOmniGen2DetailerProviderNode",
    "dispatch_detailer_stage",
    "run_native_detailer",
    "sample_prepared_crop",
    "validate_sampling_provider",
]
