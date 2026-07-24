# Umbra Nodes

Umbra Nodes is the ComfyUI custom-node package used by Umbra Studio.

It provides Umbra metadata save nodes, Power Prompter bridge nodes, prompt/runtime
helpers, and Umbra-specific workflow support for ComfyUI.

## Included Nodes

- `Power Prompter (Umbra Lab)`
- `Power Prompter Websocket`
- `Save Image (Umbra Lab)`
- `Save Image Simple (Umbra Lab)`
- `A1111 LoRA Syntax (Umbra Lab)`
- `KSampler (Umbra Lab)`
- `KSampler Normal (Umbra Lab)`
- `CFG Value (Umbra Lab)`
- `Steps Value (Umbra Lab)`
- `Seed Value (Umbra Lab)`
- `Load Checkpoint (Umbra Lab)`
- `Image Detailer (Umbra UI)`
- `Image Upscale (Umbra UI)`

The Umbra UI detailer presents a stable person, face, eye, and hand refinement
contract while keeping detector/SAM wiring inside Umbra Nodes. The upscale node
uses a selected super-resolution model and applies a true maximum-dimension cap;
it never enlarges an already-smaller model output a second time.

## Umbra UI Dependencies

The image detailer requires ComfyUI Impact Pack and Impact Subpack. Its balanced
profile currently expects these model files in the normal ComfyUI model folders:

- `models/sams/sam_vit_b_01ec64.pth`
- `models/ultralytics/segm/person_yolov8m-seg.pt`
- `models/ultralytics/bbox/face_yolov8m.pt`
- `models/ultralytics/bbox/Eyes.pt`
- `models/ultralytics/bbox/hand_yolov8s.pt`

The default image upscale profile uses
`models/upscale_models/4x-AnimeSharp.pth` and caps the final long edge at 3840
pixels.

## Manual Install

Clone this repository into your ComfyUI custom nodes folder:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/Nocturne-Ai-Labs/Umbra-Nodes.git Umbra-Nodes
```

Then install Python requirements if ComfyUI-Manager does not do it for you:

```bash
cd Umbra-Nodes
pip install -r requirements.txt
```

Restart ComfyUI after installing or updating.

## Example Workflow

The `example_workflows/` folder includes
`power-prompter-basic-api-workflow.json`, a minimal image API workflow using
`UmbraPowerPrompter`, ComfyUI's `KSampler`, and `UmbraLabSaveImage`.

It also includes `umbra-ui-anima-api-workflow.json`, the Umbra UI image
pipeline for Anima/SDXL-style prompting. It keeps Power Prompter as the batch
source while routing each prompt through Umbra's detail and 4K upscale nodes.

The same file is also mirrored in `examples/` for normal repository browsing.

## ComfyUI-Manager / Registry

This repository includes:

- `pyproject.toml` for Comfy Registry metadata
- `requirements.txt` for ComfyUI-Manager dependency install
- `.comfyignore` for registry package cleanup

Before publishing to Comfy Registry, make sure the `PublisherId` in
`pyproject.toml` matches the registered Comfy Registry publisher account.

## Relationship To Umbra Studio

Umbra Studio can bundle or sync this node package into its managed ComfyUI
runtime. This repository is the standalone source package so users can also
install/update Umbra Nodes independently through ComfyUI-Manager or a normal Git
clone.

## License

MIT. See `LICENSE`.
