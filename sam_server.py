"""Umbra-owned interactive SAM endpoint for the Umbra UI canvas."""

import asyncio
import json
import threading
from io import BytesIO

import numpy as np
from aiohttp import web
from PIL import Image
from server import PromptServer

import folder_paths

from .nodes import _load_umbra_sam_model


_MAX_IMAGE_BYTES = 64 * 1024 * 1024
_SAM_INFERENCE_LOCK = threading.RLock()


def _json_error(message, status=400):
  return web.json_response({"ok": False, "error": str(message)}, status=status)


def _clamp(value, minimum, maximum):
  return max(minimum, min(maximum, float(value)))


def _parse_points(raw_points, width, height):
  try:
    source = json.loads(raw_points or "[]")
  except json.JSONDecodeError as exc:
    raise ValueError("SAM points must be valid JSON.") from exc
  if not isinstance(source, list):
    raise ValueError("SAM points must be a list.")

  points = []
  labels = []
  for entry in source[:256]:
    if isinstance(entry, dict):
      x = entry.get("x")
      y = entry.get("y")
      positive = entry.get("positive", True)
    elif isinstance(entry, (list, tuple)) and len(entry) >= 2:
      x = entry[0]
      y = entry[1]
      positive = entry[2] if len(entry) >= 3 else True
    else:
      continue
    try:
      points.append((_clamp(x, 0, max(0, width - 1)), _clamp(y, 0, max(0, height - 1))))
    except (TypeError, ValueError):
      continue
    labels.append(1 if bool(positive) else 0)
  return points, labels


def _parse_box(raw_box, width, height):
  if not raw_box:
    return None
  try:
    source = json.loads(raw_box)
  except json.JSONDecodeError as exc:
    raise ValueError("The SAM box must be valid JSON.") from exc

  if isinstance(source, dict):
    x1 = source.get("x", 0)
    y1 = source.get("y", 0)
    x2 = float(x1) + float(source.get("width", 0))
    y2 = float(y1) + float(source.get("height", 0))
  elif isinstance(source, (list, tuple)) and len(source) >= 4:
    x1, y1, x2, y2 = source[:4]
  else:
    raise ValueError("The SAM box must contain x, y, width, and height.")

  left = _clamp(min(float(x1), float(x2)), 0, max(0, width - 1))
  top = _clamp(min(float(y1), float(y2)), 0, max(0, height - 1))
  right = _clamp(max(float(x1), float(x2)), 0, max(0, width - 1))
  bottom = _clamp(max(float(y1), float(y2)), 0, max(0, height - 1))
  if right - left < 2 or bottom - top < 2:
    raise ValueError("The SAM box is too small.")
  return [left, top, right, bottom]


def _run_sam(image_bytes, model_name, device_mode, threshold, raw_points, raw_box):
  try:
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
  except Exception as exc:
    raise ValueError("The SAM source image could not be decoded.") from exc
  width, height = image.size
  points, labels = _parse_points(raw_points, width, height)
  box = _parse_box(raw_box, width, height)
  if not points and box is None:
    raise ValueError("Add at least one SAM point or box guide.")

  image_array = np.asarray(image, dtype=np.uint8)
  with _SAM_INFERENCE_LOCK:
    sam_model = _load_umbra_sam_model(model_name, device_mode)
    wrapper = getattr(sam_model, "sam_wrapper", sam_model)
    if not callable(getattr(wrapper, "predict", None)):
      raise RuntimeError("The selected SAM model does not expose image prediction.")
    prepare = getattr(wrapper, "prepare_device", None)
    release = getattr(wrapper, "release_device", None)
    if callable(prepare):
      prepare()
    try:
      masks = wrapper.predict(image_array, points, labels, box, threshold)
    finally:
      if callable(release):
        release()

  combined = np.zeros((height, width), dtype=np.bool_)
  for mask in masks or []:
    if hasattr(mask, "detach"):
      mask = mask.detach().cpu().numpy()
    array = np.asarray(mask).squeeze()
    if array.shape != combined.shape:
      continue
    combined |= array > 0.5
  if not np.any(combined):
    raise RuntimeError("SAM did not find a selection for those guides.")

  output = BytesIO()
  Image.fromarray((combined.astype(np.uint8) * 255), mode="L").save(output, format="PNG")
  return output.getvalue()


@PromptServer.instance.routes.get("/umbra/sam/capabilities")
async def umbra_sam_capabilities(_request):
  models = [
    name for name in folder_paths.get_filename_list("sams")
    if str(name).lower().endswith((".pt", ".pth", ".safetensors"))
  ]
  return web.json_response({
    "ok": True,
    "available": bool(models),
    "models": models,
    "devices": ["CPU", "AUTO", "Prefer GPU"],
    "supportsPoints": True,
    "supportsBoxes": True,
  })


@PromptServer.instance.routes.post("/umbra/sam/detect")
async def umbra_sam_detect(request):
  if not request.content_type.startswith("multipart/"):
    return _json_error("SAM detection requires multipart form data.", 415)

  fields = {}
  image_bytes = b""
  try:
    reader = await request.multipart()
    while True:
      field = await reader.next()
      if field is None:
        break
      if field.name == "image":
        image_bytes = await field.read(decode=False)
        if len(image_bytes) > _MAX_IMAGE_BYTES:
          return _json_error("The SAM source image is larger than 64 MB.", 413)
      else:
        fields[field.name] = await field.text()
  except Exception as exc:
    return _json_error(f"The SAM request could not be read: {exc}")

  if not image_bytes:
    return _json_error("A source image is required.")
  model_name = str(fields.get("model_name") or "").strip()
  if not model_name:
    return _json_error("Choose a SAM model.")
  device_mode = str(fields.get("device_mode") or "CPU").strip()
  if device_mode not in ("CPU", "AUTO", "Prefer GPU"):
    device_mode = "CPU"
  try:
    threshold = _clamp(fields.get("threshold", 0.7), 0.0, 1.0)
    png = await asyncio.to_thread(
      _run_sam,
      image_bytes,
      model_name,
      device_mode,
      threshold,
      fields.get("points", "[]"),
      fields.get("box", ""),
    )
  except ValueError as exc:
    return _json_error(exc)
  except Exception as exc:
    return _json_error(f"SAM selection failed: {exc}", 500)

  return web.Response(
    body=png,
    headers={
      "Content-Type": "image/png",
      "Cache-Control": "no-store",
    },
  )
