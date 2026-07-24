import ast
import importlib.util
import inspect
import sys
import types
import unittest
from pathlib import Path
from unittest import mock


MODULE_PATH = Path(__file__).resolve().parents[1] / "detailer_sampling.py"
NODES_PATH = MODULE_PATH.with_name("nodes.py")
SPEC = importlib.util.spec_from_file_location("umbra_detailer_sampling_test", MODULE_PATH)
DETAILER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = DETAILER
SPEC.loader.exec_module(DETAILER)


class FakeSamplingBackend:
    def __init__(self):
        self.calls = []
        self.schedule_id = 0

    def named(self, name):
        return [call for call in self.calls if call[0] == name]

    def random_noise(self, seed):
        value = {"kind": "noise", "seed": seed, "index": len(self.named("random_noise"))}
        self.calls.append(("random_noise", seed, value))
        return value

    def basic_guider(self, model, conditioning):
        value = {"kind": "basic_guider", "model": model, "conditioning": conditioning}
        self.calls.append(("basic_guider", model, conditioning, value))
        return value

    def cfg_guider(self, model, positive, negative, cfg):
        value = {"kind": "cfg_guider", "model": model}
        self.calls.append(("cfg_guider", model, positive, negative, cfg, value))
        return value

    def dual_model_guider(self, model, model_negative, positive, negative, cfg):
        value = {"kind": "dual_model_guider", "model": model}
        self.calls.append((
            "dual_model_guider",
            model,
            model_negative,
            positive,
            negative,
            cfg,
            value,
        ))
        return value

    def dual_cfg_guider(self, model, cond1, cond2, negative, cfg_conds, cfg_cond2_negative, style):
        value = {"kind": "dual_cfg_guider", "model": model}
        self.calls.append((
            "dual_cfg_guider",
            model,
            cond1,
            cond2,
            negative,
            cfg_conds,
            cfg_cond2_negative,
            style,
            value,
        ))
        return value

    def sampler_select(self, sampler_name):
        value = {"kind": "sampler", "name": sampler_name, "index": len(self.named("sampler_select"))}
        self.calls.append(("sampler_select", sampler_name, value))
        return value

    def lcm_sampler(self, s_noise, s_noise_end, noise_clip_std):
        value = {"kind": "lcm_sampler", "index": len(self.named("lcm_sampler"))}
        self.calls.append(("lcm_sampler", s_noise, s_noise_end, noise_clip_std, value))
        return value

    def _schedule(self, name, steps, *args):
        self.schedule_id += 1
        sigmas = [f"{name}-{self.schedule_id}-{index}" for index in range(steps + 1)]
        self.calls.append((name, steps, *args, sigmas))
        return sigmas

    def basic_sigmas(self, model, scheduler, steps, denoise):
        return self._schedule("basic_sigmas", steps, model, scheduler, denoise)

    def flux2_sigmas(self, steps, width, height):
        return self._schedule("flux2_sigmas", steps, width, height)

    def ideogram4_sigmas(self, steps, width, height, mu, std):
        return self._schedule("ideogram4_sigmas", steps, width, height, mu, std)

    def sample_advanced(self, noise, guider, sampler, sigmas, latent):
        output = {
            "samples": f"sample-{len(self.named('sample_advanced')) + 1}",
            "source": latent,
            "sigmas": sigmas,
        }
        self.calls.append(("sample_advanced", noise, guider, sampler, sigmas, latent, output))
        return output

    def patch_model_noise_scale(self, model, noise_scale):
        patched = f"patched:{model}:{noise_scale}"
        self.calls.append(("patch_model_noise_scale", model, noise_scale, patched))
        return patched


def context(**overrides):
    values = {
        "model": "crop-model",
        "positive": "crop-positive",
        "negative": "crop-negative",
        "latent": {"samples": "crop-latent"},
        "seed": 10,
        "steps": 4,
        "cfg": 3.5,
        "denoise": 0.5,
        "width": 512,
        "height": 768,
    }
    values.update(overrides)
    return DETAILER.DetailerSamplingContext(**values)


def compatible_impact_modules():
    def accepts_any_call(*_args, **_kwargs):
        return None

    class FaceDetailer:
        def doit(self, *_args, **_kwargs):
            return None

    class ConditioningConcat:
        def concat(self, left, right):
            return (left, right)

    class InpaintModelConditioning:
        def encode(self, positive, negative, pixels, vae, mask, noise_mask=True):
            return positive, negative, {"samples": pixels}

    class VAEDecodeTiled:
        def decode(self, vae, samples, tile_size):
            return (samples,)

    core_symbols = (
        "segs_scale_match",
        "make_sam_mask",
        "segs_bitwise_and_mask",
        "segs_to_combined_mask",
        "crop_condition_mask",
    )
    utils_symbols = (
        "tensor_resize",
        "tensor_gaussian_blur_mask",
        "apply_differential_diffusion",
        "to_latent_image",
        "crop_ndarray4",
        "to_tensor",
        "tensor_paste",
        "tensor_convert_rgb",
    )
    nodes_module = types.SimpleNamespace(
        NODE_CLASS_MAPPINGS={"FaceDetailer": FaceDetailer},
        ConditioningConcat=ConditioningConcat,
        InpaintModelConditioning=InpaintModelConditioning,
        VAEDecodeTiled=VAEDecodeTiled,
    )
    return {
        "nodes": nodes_module,
        "impact.core": types.SimpleNamespace(**{
            name: accepts_any_call for name in core_symbols
        }),
        "impact.utils": types.SimpleNamespace(**{
            name: accepts_any_call for name in utils_symbols
        }),
        "impact.wildcards": types.SimpleNamespace(
            process_with_loras=accepts_any_call,
            process_wildcard_for_segs=accepts_any_call,
        ),
        "torch": types.SimpleNamespace(cat=accepts_any_call),
    }


class DetailerSamplingTests(unittest.TestCase):
    def test_native_crop_calculates_final_dimensions_before_provider_sampling(self):
        class FakeTensor:
            def __init__(self, height, width):
                self.shape = (1, height, width, 3)

            def cpu(self):
                return self

        class FakeUtils:
            @staticmethod
            def tensor_resize(_image, width, height):
                return FakeTensor(height, width)

            @staticmethod
            def to_latent_image(image, vae, vae_tiled_encode=False):
                vae.encoded_size = (image.shape[1], image.shape[2])
                vae.tiled_encode = vae_tiled_encode
                return {"samples": "encoded-crop"}

        class FakeVae:
            def decode(self, _samples):
                return FakeTensor(*self.encoded_size)

        backend = FakeSamplingBackend()
        provider = DETAILER.Flux2DetailerSamplingProvider(
            model="provider-model",
            backend=backend,
        )
        api = types.SimpleNamespace(
            tensor_resize=FakeUtils.tensor_resize,
            to_latent_image=FakeUtils.to_latent_image,
        )
        vae = FakeVae()

        result, _ = DETAILER._enhance_detail_native(
            api=api,
            provider=provider,
            image=FakeTensor(100, 150),
            model="crop-model",
            clip=None,
            vae=vae,
            guide_size=256,
            guide_size_for_bbox=True,
            max_size=1024,
            bbox=(0, 0, 50, 50),
            seed=42,
            steps=4,
            cfg=3.5,
            positive="adjusted-crop-positive",
            negative="adjusted-crop-negative",
            denoise=0.5,
            noise_mask=None,
            force_inpaint=True,
            wildcard_item=None,
            wildcard_concat_mode=None,
            control_net_wrapper=None,
            cycle=1,
            inpaint_model=False,
            noise_mask_feather=0,
            tiled_encode=True,
            tiled_decode=False,
            crop_conditioning=lambda value: value,
        )

        self.assertEqual(vae.encoded_size, (512, 768))
        self.assertTrue(vae.tiled_encode)
        self.assertEqual(result.shape, (1, 100, 150, 3))
        self.assertEqual(backend.named("flux2_sigmas")[0][1:4], (8, 768, 512))
        self.assertEqual(backend.named("basic_guider")[0][1:3], ("crop-model", "adjusted-crop-positive"))

    def test_flux_rebuilds_sampling_components_for_each_crop_cycle(self):
        backend = FakeSamplingBackend()
        provider = DETAILER.Flux2DetailerSamplingProvider(
            model="provider-model",
            sampler_name="euler",
            backend=backend,
        )

        first = DETAILER.sample_prepared_crop(provider, context(), cycles=2)
        second = DETAILER.sample_prepared_crop(
            provider,
            context(
                model="second-crop-model",
                positive="second-crop-positive",
                negative="second-crop-negative",
                latent={"samples": "second-crop-latent"},
                seed=30,
                steps=5,
                denoise=1.0,
                width=768,
                height=512,
            ),
        )

        self.assertEqual([call[1] for call in backend.named("random_noise")], [10, 11, 30])
        self.assertEqual(len(backend.named("basic_guider")), 3)
        self.assertEqual(backend.named("basic_guider")[0][2], "crop-positive")
        self.assertEqual(backend.named("basic_guider")[2][1:3], ("second-crop-model", "second-crop-positive"))

        schedules = backend.named("flux2_sigmas")
        self.assertEqual(schedules[0][1:4], (8, 512, 768))
        self.assertEqual(schedules[1][1:4], (8, 512, 768))
        self.assertEqual(schedules[2][1:4], (5, 768, 512))
        sampled_sigmas = [call[4] for call in backend.named("sample_advanced")]
        self.assertTrue(all(len(sigmas) == 5 for sigmas in sampled_sigmas[:2]))
        self.assertEqual(len(sampled_sigmas[2]), 6)
        self.assertIsNot(sampled_sigmas[0], sampled_sigmas[1])
        self.assertIsNot(sampled_sigmas[1], sampled_sigmas[2])

        self.assertEqual(first["samples"], "sample-2")
        self.assertEqual(second["samples"], "sample-3")
        self.assertEqual(backend.named("sample_advanced")[1][5]["samples"], "sample-1")

    def test_hidream_contract_uses_noise_scaled_model_scheduler_lcm_and_cfg_guider(self):
        backend = FakeSamplingBackend()
        with mock.patch.object(DETAILER, "ComfySamplingBackend", return_value=backend):
            provider = DETAILER.UmbraHiDreamO1DetailerProviderNode().build(
                "raw-model",
                7.6,
                "normal",
                1.0,
                0.75,
                2.5,
            )[0]

        self.assertEqual(provider.model, "patched:raw-model:7.6")
        DETAILER.sample_prepared_crop(
            provider,
            context(model=provider.model, positive="adjusted-positive", negative="adjusted-negative", cfg=1.25),
        )

        self.assertEqual(backend.named("patch_model_noise_scale")[0][1:3], ("raw-model", 7.6))
        self.assertEqual(
            backend.named("cfg_guider")[0][1:5],
            (provider.model, "adjusted-positive", "adjusted-negative", 1.25),
        )
        self.assertEqual(backend.named("lcm_sampler")[0][1:4], (1.0, 0.75, 2.5))
        self.assertEqual(backend.named("basic_sigmas")[0][1:5], (4, provider.model, "normal", 0.5))

    def test_ideogram_uses_both_models_and_crop_resolution_scheduler(self):
        backend = FakeSamplingBackend()
        provider = DETAILER.Ideogram4DetailerSamplingProvider(
            model="conditional-model",
            model_negative="unconditional-model",
            sampler_name="euler",
            mu=0.5,
            std=1.75,
            backend=backend,
        )
        DETAILER.sample_prepared_crop(
            provider,
            context(
                model="conditional-model-with-crop-patches",
                positive="cropped-positive",
                negative="cropped-negative",
                width=640,
                height=896,
                cfg=7.0,
            ),
        )

        self.assertEqual(
            backend.named("dual_model_guider")[0][1:6],
            (
                "conditional-model-with-crop-patches",
                "unconditional-model",
                "cropped-positive",
                "cropped-negative",
                7.0,
            ),
        )
        self.assertEqual(backend.named("ideogram4_sigmas")[0][1:6], (8, 640, 896, 0.5, 1.75))

    def test_omnigen_uses_all_explicit_conditionings_and_scales(self):
        backend = FakeSamplingBackend()
        provider = DETAILER.OmniGen2DetailerSamplingProvider(
            model="omni-model",
            cond1="full-cond1",
            cond2="full-cond2",
            negative="full-negative",
            cfg_conds=5.0,
            cfg_cond2_negative=2.0,
            style="nested",
            sampler_name="euler",
            scheduler="simple",
            backend=backend,
        )
        self.assertEqual(
            provider.detailer_conditionings("fallback-positive", "fallback-negative"),
            ("full-cond1", "full-negative"),
        )

        DETAILER.sample_prepared_crop(
            provider,
            context(
                model="omni-crop-model",
                positive="cropped-cond1",
                negative="cropped-negative",
                crop_conditioning=lambda value: f"cropped:{value}",
            ),
        )
        self.assertEqual(
            backend.named("dual_cfg_guider")[0][1:8],
            (
                "omni-crop-model",
                "cropped-cond1",
                "cropped:full-cond2",
                "cropped-negative",
                5.0,
                2.0,
                "nested",
            ),
        )
        self.assertEqual(backend.named("basic_sigmas")[0][1:5], (4, "omni-crop-model", "simple", 0.5))

    def test_provider_crop_dimensions_are_aligned_before_sampling(self):
        backend = FakeSamplingBackend()
        providers = [
            DETAILER.Flux2DetailerSamplingProvider("model", backend=backend),
            DETAILER.Ideogram4DetailerSamplingProvider("model", backend=backend),
            DETAILER.OmniGen2DetailerSamplingProvider("model", backend=backend),
        ]
        for provider in providers:
            self.assertEqual(provider.prepare_crop_size(515, 777), (512, 768))
        hidream = DETAILER.HiDreamO1DetailerSamplingProvider("model", backend=backend)
        self.assertEqual(hidream.prepare_crop_size(515, 777), (512, 768))

    def test_malformed_provider_reports_each_required_contract_member(self):
        class MalformedProvider:
            provider_id = ""
            model = None
            prepare_crop_size = "not-callable"

        with self.assertRaises(RuntimeError) as raised:
            DETAILER.validate_sampling_provider(MalformedProvider())

        message = str(raised.exception)
        self.assertIn("callable prepare_crop_size()", message)
        self.assertIn("callable detailer_conditionings()", message)
        self.assertIn("callable sample_crop()", message)
        self.assertIn("non-empty string provider_id", message)
        self.assertIn("non-None model", message)

    def test_provider_outputs_are_validated_at_the_native_boundary(self):
        class Provider:
            provider_id = "bad-output"
            model = object()

            def prepare_crop_size(self, _width, _height):
                return 32.5, 64

            def detailer_conditionings(self, _positive, _negative):
                return "only-one"

            def sample_crop(self, _context):
                return {"samples": None}

        provider = Provider()
        DETAILER.validate_sampling_provider(provider)
        with self.assertRaisesRegex(RuntimeError, "width and height must be integers"):
            DETAILER._prepare_provider_crop_size(provider, 32, 64)
        with self.assertRaisesRegex(RuntimeError, "expected \\(positive, negative\\)"):
            DETAILER._prepare_provider_conditionings(provider, "positive", "negative")
        with self.assertRaisesRegex(RuntimeError, "valid LATENT dictionary for cycle 1"):
            DETAILER.sample_prepared_crop(provider, context(), cycles=2)

    def test_impact_facade_resolves_explicit_modules_and_required_symbols(self):
        modules = compatible_impact_modules()
        with mock.patch.object(
            DETAILER.importlib,
            "import_module",
            side_effect=modules.__getitem__,
        ) as import_module:
            api = DETAILER._load_impact_api()

        self.assertIs(
            api.face_detailer_class,
            modules["nodes"].NODE_CLASS_MAPPINGS["FaceDetailer"],
        )
        self.assertIs(api.segs_scale_match, modules["impact.core"].segs_scale_match)
        self.assertIs(api.tensor_paste, modules["impact.utils"].tensor_paste)
        self.assertIs(
            api.process_with_loras,
            modules["impact.wildcards"].process_with_loras,
        )
        self.assertEqual(
            [call.args[0] for call in import_module.call_args_list],
            ["nodes", "impact.core", "impact.utils", "impact.wildcards", "torch"],
        )

    def test_impact_facade_distinguishes_missing_face_detailer_from_incompatible_internals(self):
        modules = compatible_impact_modules()
        modules["nodes"].NODE_CLASS_MAPPINGS = {}
        with mock.patch.object(
            DETAILER.importlib,
            "import_module",
            side_effect=modules.__getitem__,
        ):
            with self.assertRaisesRegex(
                DETAILER.ImpactFaceDetailerUnavailableError,
                "'FaceDetailer' is not registered",
            ):
                DETAILER._load_impact_api()

        incompatible_modules = compatible_impact_modules()
        incompatible_modules["impact.utils"].tensor_paste = None
        with mock.patch.object(
            DETAILER.importlib,
            "import_module",
            side_effect=incompatible_modules.__getitem__,
        ):
            with self.assertRaises(DETAILER.ImpactDetailerCompatibilityError) as raised:
                DETAILER._load_impact_api()
        self.assertIn("missing callable impact.utils.tensor_paste", str(raised.exception))
        self.assertNotIsInstance(
            raised.exception,
            DETAILER.ImpactFaceDetailerUnavailableError,
        )

    def test_impact_facade_rejects_incompatible_classes_and_call_shapes(self):
        incompatible_class_modules = compatible_impact_modules()
        incompatible_class_modules["nodes"].VAEDecodeTiled = None
        with mock.patch.object(
            DETAILER.importlib,
            "import_module",
            side_effect=incompatible_class_modules.__getitem__,
        ):
            with self.assertRaisesRegex(
                DETAILER.ImpactDetailerCompatibilityError,
                "missing class nodes.VAEDecodeTiled",
            ):
                DETAILER._load_impact_api()

        incompatible_shape_modules = compatible_impact_modules()
        incompatible_shape_modules["impact.core"].make_sam_mask = lambda first, second: None
        with mock.patch.object(
            DETAILER.importlib,
            "import_module",
            side_effect=incompatible_shape_modules.__getitem__,
        ):
            with self.assertRaisesRegex(
                DETAILER.ImpactDetailerCompatibilityError,
                "impact.core.make_sam_mask does not accept the required call shape",
            ):
                DETAILER._load_impact_api()

    def test_all_provider_node_input_contracts_and_mappings(self):
        expected_inputs = {
            DETAILER.UmbraFlux2DetailerProviderNode: (
                "model",
                "sampler_name",
            ),
            DETAILER.UmbraHiDreamO1DetailerProviderNode: (
                "model",
                "noise_scale",
                "scheduler",
                "s_noise",
                "s_noise_end",
                "noise_clip_std",
            ),
            DETAILER.UmbraIdeogram4DetailerProviderNode: (
                "model",
                "model_negative",
                "sampler_name",
                "mu",
                "std",
            ),
            DETAILER.UmbraOmniGen2DetailerProviderNode: (
                "model",
                "cond1",
                "cond2",
                "negative",
                "cfg_conds",
                "cfg_cond2_negative",
                "style",
                "sampler_name",
                "scheduler",
            ),
        }
        with (
            mock.patch.object(DETAILER, "_sampler_names", return_value=("euler",)),
            mock.patch.object(DETAILER, "_scheduler_names", return_value=("simple", "normal")),
        ):
            for node_class, expected in expected_inputs.items():
                with self.subTest(node=node_class.__name__):
                    input_types = node_class.INPUT_TYPES()
                    self.assertEqual(tuple(input_types), ("required",))
                    self.assertEqual(tuple(input_types["required"]), expected)
                    self.assertEqual(
                        tuple(inspect.signature(node_class.build).parameters)[1:],
                        expected,
                    )
                    self.assertEqual(
                        node_class.RETURN_TYPES,
                        (DETAILER.DETAILER_SAMPLING_PROVIDER_TYPE,),
                    )
                    self.assertEqual(node_class.RETURN_NAMES, ("sampling_provider",))
                    self.assertEqual(node_class.FUNCTION, "build")
                    self.assertTrue(callable(getattr(node_class(), node_class.FUNCTION)))

        tree = ast.parse(NODES_PATH.read_text(encoding="utf-8"))
        mapping_assignment = next(
            node for node in tree.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "NODE_CLASS_MAPPINGS"
                for target in node.targets
            )
        )
        mappings = {
            ast.literal_eval(key): ast.unparse(value)
            for key, value in zip(mapping_assignment.value.keys, mapping_assignment.value.values)
        }
        self.assertEqual(
            {
                name: mappings[name]
                for name in (
                    "UmbraFlux2DetailerSamplingProvider",
                    "UmbraHiDreamO1DetailerSamplingProvider",
                    "UmbraIdeogram4DetailerSamplingProvider",
                    "UmbraOmniGen2DetailerSamplingProvider",
                )
            },
            {
                "UmbraFlux2DetailerSamplingProvider": "UmbraFlux2DetailerProviderNode",
                "UmbraHiDreamO1DetailerSamplingProvider": "UmbraHiDreamO1DetailerProviderNode",
                "UmbraIdeogram4DetailerSamplingProvider": "UmbraIdeogram4DetailerProviderNode",
                "UmbraOmniGen2DetailerSamplingProvider": "UmbraOmniGen2DetailerProviderNode",
            },
        )

    def test_absent_provider_is_a_strict_classic_fallback(self):
        calls = []

        result = DETAILER.dispatch_detailer_stage(
            None,
            classic_call=lambda: calls.append("classic") or ("classic-image", "detector", "sam"),
            native_call=lambda provider: calls.append("native") or provider,
        )

        self.assertEqual(result, ("classic-image", "detector", "sam"))
        self.assertEqual(calls, ["classic"])

    def test_seed_offset_migration_is_once_and_classic_face_detailer_is_unchanged(self):
        tree = ast.parse(NODES_PATH.read_text(encoding="utf-8"))
        detailer_class = next(
            node for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "UmbraImageDetailer"
        )
        run_stage = next(
            node for node in detailer_class.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "_run_stage"
        )
        detailer_call = next(
            node for node in ast.walk(run_stage)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "doit"
        )
        seed_keyword = next(keyword for keyword in detailer_call.keywords if keyword.arg == "seed")
        self.assertEqual(
            ast.unparse(seed_keyword.value),
            "_normalize_seed(int(seed) + stage['seed_offset'])",
        )
        seed_offset_reads = [
            node for node in ast.walk(run_stage)
            if isinstance(node, ast.Subscript)
            and ast.unparse(node) == "stage['seed_offset']"
        ]
        self.assertEqual(len(seed_offset_reads), 1)

        refine = next(
            node for node in detailer_class.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "refine"
        )
        dispatch_call = next(
            node for node in ast.walk(refine)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "dispatch_detailer_stage"
        )
        classic_lambda = next(keyword.value for keyword in dispatch_call.keywords if keyword.arg == "classic_call")
        self.assertIsInstance(classic_lambda, ast.Lambda)
        self.assertEqual(ast.unparse(classic_lambda.body.args[-2]), "seed")
        self.assertEqual(ast.unparse(classic_lambda.body.args[-1]), "detailer")

        detector = {
            "bbox": object(),
            "segm": object(),
            "model_name": "face-detector.pt",
        }
        namespace = {
            "DETAILER_SAMPLING_PROVIDER_TYPE": DETAILER.DETAILER_SAMPLING_PROVIDER_TYPE,
            "SEED_MAX": (1 << 64) - 1,
            "_load_umbra_detailer_detector": lambda _name: detector,
            "_load_umbra_sam_model": lambda *_args: None,
            "_normalize_seed": lambda value: value,
        }
        class_module = ast.fix_missing_locations(
            ast.Module(body=[detailer_class], type_ignores=[])
        )
        exec(compile(class_module, str(NODES_PATH), "exec"), namespace)
        extracted_class = namespace["UmbraImageDetailer"]

        class OriginalFaceDetailer:
            def __init__(self):
                self.calls = []
                self.output = object()

            def doit(self, **kwargs):
                self.calls.append(kwargs)
                return (self.output, "all-other-face-detailer-outputs")

        stage = {
            "detector_model": "face-detector.pt",
            "use_sam": False,
            "sam_model": "unused.pth",
            "sam_device_mode": "CPU",
            "guide_size": 512,
            "guide_size_for": "bbox",
            "max_size": 1024,
            "seed_offset": 7,
            "steps": 8,
            "cfg": 4.0,
            "sampler_name": "er_sde",
            "scheduler": "simple",
            "denoise": 0.2,
            "feather": 5,
            "noise_mask": True,
            "force_inpaint": True,
            "bbox_threshold": 0.5,
            "bbox_dilation": 10,
            "bbox_crop_factor": 2.5,
            "sam_detection_hint": "center-1",
            "sam_dilation": 0,
            "sam_threshold": 0.93,
            "sam_bbox_expansion": 0,
            "sam_mask_hint_threshold": 0.7,
            "sam_mask_hint_use_negative": "False",
            "drop_size": 10,
            "wildcard": "",
            "cycle": 1,
            "noise_mask_feather": 20,
            "tiled_encode": False,
            "tiled_decode": False,
        }
        face_detailer = OriginalFaceDetailer()
        image = object()
        model = object()
        clip = object()
        vae = object()
        positive = object()
        negative = object()

        result = DETAILER.dispatch_detailer_stage(
            None,
            classic_call=lambda: extracted_class._run_stage(
                stage,
                image,
                model,
                clip,
                vae,
                positive,
                negative,
                100,
                face_detailer,
            ),
            native_call=lambda _provider: self.fail("native path must not run"),
        )

        self.assertIs(result[0], face_detailer.output)
        self.assertEqual(result[1:], ("face-detector.pt", ""))
        self.assertEqual(len(face_detailer.calls), 1)
        face_call = face_detailer.calls[0]
        self.assertEqual(face_call["seed"], 107)
        self.assertIs(face_call["image"], image)
        self.assertIs(face_call["model"], model)
        self.assertIs(face_call["clip"], clip)
        self.assertIs(face_call["vae"], vae)
        self.assertIs(face_call["positive"], positive)
        self.assertIs(face_call["negative"], negative)
        self.assertIs(face_call["bbox_detector"], detector["bbox"])
        self.assertIs(face_call["segm_detector_opt"], detector["segm"])
        self.assertEqual(face_call["sampler_name"], "er_sde")
        self.assertEqual(face_call["scheduler"], "simple")


if __name__ == "__main__":
    unittest.main()
