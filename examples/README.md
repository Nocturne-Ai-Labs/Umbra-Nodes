# Umbra Nodes Examples

This folder contains small starter workflows for Umbra Nodes.

## `power-prompter-basic-api-workflow.json`

A minimal ComfyUI API workflow for image generation:

- `UmbraPowerPrompter` owns prompt text, model selection, seed, size, sampler, and CFG.
- Standard `KSampler` and `VAEDecode` perform the image generation.
- `UmbraLabSaveImage` saves the image with Umbra, ComfyUI, and A1111-compatible metadata.

Use this as a starting point for Umbra Studio API workflow targets. Replace the
prompt text, negative prompt, model selector, and generation settings as needed.

## `umbra-ui-anima-api-workflow.json`

The full Umbra UI Anima image path. Power Prompter supplies each prompt and the
generation controls, then Umbra's detailer and AnimeSharp 4K nodes finish the
image before `UmbraLabSaveImage` writes metadata and output folders.

## `umbra-ui-image-upscaler-api-workflow.json`

The standalone Umbra UI Extras path: `LoadImage` feeds `UmbraImageUpscale`, then
`UmbraLabSaveImage` writes the result into the dated `Umbra UI/extras` tree.
Umbra Studio builds this same three-node graph dynamically for preview actions
and multi-image Extras batches.
