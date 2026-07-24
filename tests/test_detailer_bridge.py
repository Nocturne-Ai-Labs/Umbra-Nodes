import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import torch
import torch.nn.functional as functional


MODULE_PATH = Path(__file__).resolve().parents[1] / "detailer_sampling.py"
SPEC = importlib.util.spec_from_file_location("umbra_detailer_bridge_test", MODULE_PATH)
DETAILER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = DETAILER
SPEC.loader.exec_module(DETAILER)


class TensorUtilsStub:
    def __init__(self):
        self.resize_calls = []
        self.blur_calls = []
        self.differential_calls = []
        self.latent_calls = []
        self.paste_calls = []

    def tensor_resize(self, image, width, height):
        self.resize_calls.append((tuple(image.shape), width, height))
        channels_first = image.movedim(-1, 1)
        resized = functional.interpolate(channels_first, size=(height, width), mode="nearest")
        return resized.movedim(1, -1)

    def tensor_gaussian_blur_mask(self, mask, kernel_size):
        tensor = torch.as_tensor(mask, dtype=torch.float32, device="cpu")
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0).unsqueeze(-1)
        elif tensor.ndim == 3:
            tensor = tensor.unsqueeze(-1)
        self.blur_calls.append((tuple(tensor.shape), kernel_size))
        return tensor.clone()

    def apply_differential_diffusion(self, model):
        patched = types.SimpleNamespace(
            model_options={**getattr(model, "model_options", {}), "denoise_mask_function": object()},
            original_model=model,
        )
        self.differential_calls.append((model, patched))
        return patched

    def to_latent_image(self, image, vae, vae_tiled_encode=False):
        height, width = int(image.shape[1]), int(image.shape[2])
        vae.last_encoded_size = (height, width)
        self.latent_calls.append((tuple(image.shape), vae_tiled_encode))
        return {
            "samples": torch.zeros(
                (int(image.shape[0]), 1, max(1, height // 4), max(1, width // 4)),
                dtype=torch.float32,
                device="cpu",
            )
        }

    @staticmethod
    def crop_ndarray4(array, crop_region):
        x1, y1, x2, y2 = crop_region
        return array[:, y1:y2, x1:x2, :]

    @staticmethod
    def to_tensor(value):
        return torch.as_tensor(value, dtype=torch.float32, device="cpu")

    def tensor_paste(self, destination, source, left_top, mask):
        x, y = left_top
        height, width = int(source.shape[1]), int(source.shape[2])
        if tuple(mask.shape[1:3]) != (height, width):
            mask = functional.interpolate(
                mask.movedim(-1, 1),
                size=(height, width),
                mode="nearest",
            ).movedim(1, -1)
        destination_region = destination[:, y:y + height, x:x + width, :]
        destination_region.copy_(destination_region * (1.0 - mask) + source * mask)
        self.paste_calls.append((left_top, tuple(source.shape), tuple(mask.shape)))

    @staticmethod
    def tensor_convert_rgb(image):
        return image[..., :3]


class CoreStub:
    def __init__(self):
        self.scale_calls = []
        self.sam_calls = []
        self.bitwise_calls = []
        self.combined_mask_calls = []
        self.conditioning_crop_calls = []

    def segs_scale_match(self, segs, target_shape):
        self.scale_calls.append((segs, tuple(target_shape)))
        return segs

    def make_sam_mask(
        self,
        sam,
        segs,
        image,
        detection_hint,
        dilation,
        threshold,
        bbox_expansion,
        mask_hint_threshold,
        mask_hint_use_negative,
    ):
        self.sam_calls.append((
            sam,
            segs,
            tuple(image.shape),
            detection_hint,
            dilation,
            threshold,
            bbox_expansion,
            mask_hint_threshold,
            mask_hint_use_negative,
        ))
        return torch.ones((1, image.shape[1], image.shape[2]), dtype=torch.float32)

    def segs_bitwise_and_mask(self, segs, mask):
        self.bitwise_calls.append((segs, mask))
        return segs

    def segs_to_combined_mask(self, segs):
        self.combined_mask_calls.append(segs)
        return torch.ones((1, 1, 1), dtype=torch.float32)

    def crop_condition_mask(self, mask, _image, crop_region):
        tensor = torch.as_tensor(mask, dtype=torch.float32, device="cpu")
        x1, y1, x2, y2 = crop_region
        if tensor.ndim == 4:
            cropped = tensor[:, y1:y2, x1:x2, :]
        elif tensor.ndim == 3:
            cropped = tensor[:, y1:y2, x1:x2]
        else:
            cropped = tensor[y1:y2, x1:x2]
        self.conditioning_crop_calls.append((crop_region, cropped.clone()))
        return cropped


class ConditioningConcatStub:
    def concat(self, left, right):
        return (left + right,)


class InpaintModelConditioningStub:
    calls = []

    def encode(self, positive, negative, pixels, vae, mask, noise_mask=True):
        vae.last_encoded_size = (int(pixels.shape[1]), int(pixels.shape[2]))
        self.__class__.calls.append({
            "positive": positive,
            "negative": negative,
            "pixels": pixels,
            "mask": mask.clone(),
            "noise_mask": noise_mask,
        })
        return positive, negative, {
            "samples": torch.zeros((pixels.shape[0], 1, 1, 1), dtype=torch.float32),
            "noise_mask": mask,
        }


class VAEDecodeTiledStub:
    calls = []

    def decode(self, vae, latent, tile_size):
        self.__class__.calls.append((latent, tile_size))
        return (vae.render(latent["samples"]),)


class FakeVAE:
    def __init__(self):
        self.last_encoded_size = None
        self.decode_calls = []
        self.decode_tiled_calls = []

    def render(self, samples):
        height, width = self.last_encoded_size
        value = float(samples.mean().item())
        return torch.full(
            (int(samples.shape[0]), height, width, 3),
            value,
            dtype=torch.float32,
            device="cpu",
        )

    def decode(self, samples):
        self.decode_calls.append(samples)
        return self.render(samples)

    def decode_tiled(self, samples, tile_x, tile_y):
        self.decode_tiled_calls.append((samples, tile_x, tile_y))
        return self.render(samples)


class RecordingProvider:
    provider_id = "recording-native"

    def __init__(self, extra_conditioning=None, invalid_latent=False):
        self.model = types.SimpleNamespace(model_options={})
        self.extra_conditioning = extra_conditioning
        self.invalid_latent = invalid_latent
        self.conditioning_calls = []
        self.crop_size_calls = []
        self.contexts = []
        self.extra_crops = []

    def prepare_crop_size(self, width, height):
        self.crop_size_calls.append((width, height))
        return int(width), int(height)

    def detailer_conditionings(self, positive, negative):
        self.conditioning_calls.append((positive, negative))
        return positive, negative

    def sample_crop(self, context):
        self.contexts.append(context)
        if self.extra_conditioning is not None:
            self.extra_crops.append(context.crop_conditioning(self.extra_conditioning))
        if self.invalid_latent:
            return {"not_samples": context.latent}
        return {"samples": context.latent["samples"] + 0.25}


class DetectorStub:
    def __init__(self, segs):
        self.segs = segs
        self.aux_calls = []
        self.detect_calls = []

    def setAux(self, value):
        self.aux_calls.append(value)

    def detect(self, image, threshold, dilation, crop_factor, drop_size):
        self.detect_calls.append((tuple(image.shape), threshold, dilation, crop_factor, drop_size))
        return self.segs


def make_api(core, utils):
    class FaceDetailerStub:
        def doit(self):
            return None

    return DETAILER._ImpactApi(
        face_detailer_class=FaceDetailerStub,
        segs_scale_match=core.segs_scale_match,
        make_sam_mask=core.make_sam_mask,
        segs_bitwise_and_mask=core.segs_bitwise_and_mask,
        segs_to_combined_mask=core.segs_to_combined_mask,
        crop_condition_mask=core.crop_condition_mask,
        tensor_resize=utils.tensor_resize,
        tensor_gaussian_blur_mask=utils.tensor_gaussian_blur_mask,
        apply_differential_diffusion=utils.apply_differential_diffusion,
        to_latent_image=utils.to_latent_image,
        crop_ndarray4=utils.crop_ndarray4,
        to_tensor=utils.to_tensor,
        tensor_paste=utils.tensor_paste,
        tensor_convert_rgb=utils.tensor_convert_rgb,
        process_with_loras=lambda *_args: (_ for _ in ()).throw(
            AssertionError("wildcard LoRA processing was not expected")
        ),
        process_wildcard_for_segs=lambda _wildcard: (None, None),
        conditioning_concat_class=ConditioningConcatStub,
        inpaint_model_conditioning_class=InpaintModelConditioningStub,
        vae_decode_tiled_class=VAEDecodeTiledStub,
        concat_tensors=torch.cat,
    )


def native_kwargs(image, provider, detector, **overrides):
    values = {
        "image": image,
        "provider": provider,
        "clip": object(),
        "vae": FakeVAE(),
        "guide_size": 4,
        "guide_size_for_bbox": True,
        "max_size": 8,
        "seed": 101,
        "steps": 4,
        "cfg": 3.0,
        "positive": "positive",
        "negative": "negative",
        "denoise": 0.5,
        "feather": 0,
        "noise_mask": True,
        "force_inpaint": True,
        "bbox_threshold": 0.5,
        "bbox_dilation": 0,
        "bbox_crop_factor": 1.0,
        "sam_detection_hint": "center-1",
        "sam_dilation": 0,
        "sam_threshold": 0.93,
        "sam_bbox_expansion": 0,
        "sam_mask_hint_threshold": 0.7,
        "sam_mask_hint_use_negative": "False",
        "drop_size": 1,
        "bbox_detector": detector,
        "wildcard": None,
        "cycle": 1,
    }
    values.update(overrides)
    return values


class NativeDetailerBridgeTests(unittest.TestCase):
    """CPU tensor contract tests with realistic stubs; these are not GPU inference tests."""

    def setUp(self):
        InpaintModelConditioningStub.calls.clear()
        VAEDecodeTiledStub.calls.clear()

    def test_run_native_detailer_exercises_masks_cycles_crops_tiles_and_compositing(self):
        image = torch.zeros((1, 8, 8, 3), dtype=torch.float32, device="cpu")
        full_mask = torch.arange(64, dtype=torch.float32).reshape(1, 8, 8)
        positive = [["positive-token", {"mask": full_mask, "kept": "positive-meta"}]]
        negative = [["negative-token", {"mask": full_mask + 100, "kept": "negative-meta"}]]
        extra = [["extra-token", {"mask": full_mask + 200}]]
        active_segment = types.SimpleNamespace(
            crop_region=(2, 1, 6, 5),
            bbox=(2, 1, 6, 5),
            cropped_mask=torch.ones((4, 4), dtype=torch.float32),
            control_net_wrapper=None,
        )
        empty_segment = types.SimpleNamespace(
            crop_region=(0, 0, 2, 2),
            bbox=(0, 0, 2, 2),
            cropped_mask=torch.zeros((2, 2), dtype=torch.float32),
            control_net_wrapper=None,
        )
        segs = ((8, 8), [active_segment, empty_segment])
        detector = DetectorStub(segs)
        provider = RecordingProvider(extra_conditioning=extra)
        core = CoreStub()
        utils = TensorUtilsStub()
        api = make_api(core, utils)
        vae = FakeVAE()
        sam_model = object()

        with mock.patch.object(DETAILER, "_load_impact_api", return_value=api):
            result = DETAILER.run_native_detailer(**native_kwargs(
                image,
                provider,
                detector,
                vae=vae,
                positive=positive,
                negative=negative,
                cycle=2,
                sam_model=sam_model,
                noise_mask_feather=0,
                tiled_encode=True,
                tiled_decode=True,
            ))

        self.assertEqual(result.device.type, "cpu")
        self.assertEqual(tuple(result.shape), (1, 8, 8, 3))
        self.assertTrue(torch.allclose(result[:, 1:5, 2:6, :], torch.full((1, 4, 4, 3), 0.5)))
        outside = result.clone()
        outside[:, 1:5, 2:6, :] = 0
        self.assertEqual(int(torch.count_nonzero(outside)), 0)
        self.assertEqual(detector.aux_calls, ["face", None])
        self.assertEqual(len(core.sam_calls), 1)
        self.assertIs(core.sam_calls[0][0], sam_model)
        self.assertEqual(len(core.bitwise_calls), 1)
        self.assertEqual([context.seed for context in provider.contexts], [101, 102])
        self.assertEqual(provider.crop_size_calls, [(4, 4)])
        self.assertEqual([context.width for context in provider.contexts], [4, 4])
        self.assertEqual([context.height for context in provider.contexts], [4, 4])
        self.assertIn("noise_mask", provider.contexts[0].latent)
        self.assertEqual(tuple(provider.contexts[0].latent["noise_mask"].shape), (1, 4, 4))
        self.assertTrue(torch.equal(
            provider.contexts[0].positive[0][1]["mask"],
            full_mask[:, 1:5, 2:6],
        ))
        self.assertTrue(torch.equal(
            provider.contexts[0].negative[0][1]["mask"],
            (full_mask + 100)[:, 1:5, 2:6],
        ))
        self.assertEqual(provider.contexts[0].positive[0][1]["kept"], "positive-meta")
        self.assertEqual(provider.contexts[0].negative[0][1]["kept"], "negative-meta")
        self.assertEqual(len(provider.extra_crops), 2)
        self.assertTrue(torch.equal(
            provider.extra_crops[0][0][1]["mask"],
            (full_mask + 200)[:, 1:5, 2:6],
        ))
        self.assertEqual(utils.latent_calls, [((1, 4, 4, 3), True)])
        self.assertEqual(len(utils.paste_calls), 1)
        self.assertEqual(utils.paste_calls[0][0], (2, 1))
        self.assertEqual(len(VAEDecodeTiledStub.calls), 1)
        self.assertEqual(VAEDecodeTiledStub.calls[0][1], 512)
        self.assertEqual(len(vae.decode_calls), 0)

    def test_inpaint_noise_mask_and_cycles_reach_provider_and_normal_decode(self):
        image = torch.zeros((1, 4, 4, 3), dtype=torch.float32, device="cpu")
        noise_mask = torch.ones((4, 4), dtype=torch.float32, device="cpu")
        provider = RecordingProvider()
        core = CoreStub()
        utils = TensorUtilsStub()
        api = make_api(core, utils)
        vae = FakeVAE()

        result, cnet_images = DETAILER._enhance_detail_native(
            api=api,
            provider=provider,
            image=image,
            model=provider.model,
            clip=object(),
            vae=vae,
            guide_size=4,
            guide_size_for_bbox=True,
            max_size=8,
            bbox=(0, 0, 4, 4),
            seed=7,
            steps=5,
            cfg=2.5,
            positive="positive",
            negative="negative",
            denoise=0.4,
            noise_mask=noise_mask,
            force_inpaint=True,
            wildcard_item=None,
            wildcard_concat_mode=None,
            control_net_wrapper=None,
            cycle=3,
            inpaint_model=True,
            noise_mask_feather=2,
            tiled_encode=True,
            tiled_decode=False,
            crop_conditioning=lambda value: value,
        )

        self.assertIsNone(cnet_images)
        self.assertEqual(result.device.type, "cpu")
        self.assertTrue(torch.allclose(result, torch.full((1, 4, 4, 3), 0.75)))
        self.assertEqual([context.seed for context in provider.contexts], [7, 8, 9])
        self.assertEqual(len(utils.differential_calls), 1)
        patched_model = utils.differential_calls[0][1]
        self.assertTrue(all(context.model is patched_model for context in provider.contexts))
        self.assertEqual(len(InpaintModelConditioningStub.calls), 1)
        inpaint_call = InpaintModelConditioningStub.calls[0]
        self.assertEqual(tuple(inpaint_call["mask"].shape), (1, 4, 4))
        self.assertTrue(inpaint_call["noise_mask"])
        self.assertIn("noise_mask", provider.contexts[0].latent)
        self.assertEqual(len(utils.latent_calls), 0)
        self.assertEqual(len(vae.decode_calls), 1)
        self.assertEqual(len(VAEDecodeTiledStub.calls), 0)

    def test_invalid_latent_result_fails_inside_native_enhancement_before_decode(self):
        provider = RecordingProvider(invalid_latent=True)
        core = CoreStub()
        utils = TensorUtilsStub()
        api = make_api(core, utils)
        vae = FakeVAE()

        with self.assertRaisesRegex(RuntimeError, "valid LATENT dictionary for cycle 1"):
            DETAILER._enhance_detail_native(
                api=api,
                provider=provider,
                image=torch.zeros((1, 4, 4, 3), dtype=torch.float32),
                model=provider.model,
                clip=object(),
                vae=vae,
                guide_size=4,
                guide_size_for_bbox=True,
                max_size=8,
                bbox=(0, 0, 4, 4),
                seed=1,
                steps=4,
                cfg=2.0,
                positive="positive",
                negative="negative",
                denoise=0.5,
                noise_mask=None,
                force_inpaint=True,
                wildcard_item=None,
                wildcard_concat_mode=None,
                control_net_wrapper=None,
                cycle=1,
                inpaint_model=False,
                noise_mask_feather=0,
                tiled_encode=False,
                tiled_decode=False,
                crop_conditioning=lambda value: value,
            )

        self.assertEqual(len(vae.decode_calls), 0)
        self.assertEqual(len(VAEDecodeTiledStub.calls), 0)

    def test_run_native_rejects_malformed_provider_before_loading_impact(self):
        image = torch.zeros((1, 4, 4, 3), dtype=torch.float32)
        detector = DetectorStub(((4, 4), []))
        with mock.patch.object(DETAILER, "_load_impact_api") as load_api:
            with self.assertRaisesRegex(RuntimeError, "Invalid Umbra detailer sampling provider"):
                DETAILER.run_native_detailer(**native_kwargs(image, object(), detector))
        load_api.assert_not_called()

    def test_run_native_rejects_malformed_conditioning_result_before_detection(self):
        class BadConditioningProvider(RecordingProvider):
            def detailer_conditionings(self, positive, negative):
                return [positive]

        image = torch.zeros((1, 4, 4, 3), dtype=torch.float32)
        detector = DetectorStub(((4, 4), []))
        provider = BadConditioningProvider()
        api = make_api(CoreStub(), TensorUtilsStub())
        with mock.patch.object(DETAILER, "_load_impact_api", return_value=api):
            with self.assertRaisesRegex(RuntimeError, "expected \\(positive, negative\\)"):
                DETAILER.run_native_detailer(**native_kwargs(image, provider, detector))
        self.assertEqual(detector.detect_calls, [])


if __name__ == "__main__":
    unittest.main()
