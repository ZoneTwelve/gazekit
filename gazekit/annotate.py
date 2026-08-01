"""`gazekit annotate` — offline session-context annotation with Florence-2.

Florence-2 cannot estimate gaze and is ~100x too slow for the realtime loop,
but it is good at describing scenes. Each collection session now saves one
context snapshot (data/context/<session>.jpg, kept OUT of the published
dataset). This tool captions each snapshot and detects objects, producing
per-session environment metadata (lighting, glasses, occluders) in
data/context/annotations.jsonl — the condition labels used for slicing
analysis and, later, domain-adversarial training.

The model (microsoft/Florence-2-base-ft, ~0.5 GB) downloads ON FIRST RUN
only, to the standard HF cache. PyTorch backend, MPS when available.
"""

import json
from pathlib import Path

CONTEXT_DIR = Path("data/context")
OUT = CONTEXT_DIR / "annotations.jsonl"
MODEL_ID = "microsoft/Florence-2-base-ft"


def _load_model():
    import os
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    # fp16 on MPS is flaky for this arch; fp32 is fine at one image/session
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float32,
        trust_remote_code=True).to(device).eval()
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    return model, processor, device


def _run_task(model, processor, device, image, task):
    import torch
    inputs = processor(text=task, images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        ids = model.generate(input_ids=inputs["input_ids"],
                             pixel_values=inputs["pixel_values"],
                             max_new_tokens=512, do_sample=False, num_beams=3)
    text = processor.batch_decode(ids, skip_special_tokens=False)[0]
    return processor.post_process_generation(
        text, task=task, image_size=(image.width, image.height))


def run(redo=False):
    from PIL import Image
    snaps = sorted(CONTEXT_DIR.glob("session_*.jpg"))
    if not snaps:
        raise SystemExit(
            "no context snapshots yet — they are saved automatically by "
            "calibrate/collect/ambient runs from now on (data/context/)")
    done = set()
    if OUT.exists() and not redo:
        done = {json.loads(l)["session"] for l in open(OUT)}
    todo = [s for s in snaps if s.stem not in done]
    if not todo:
        print(f"all {len(snaps)} snapshots already annotated -> {OUT}")
        return

    print(f"annotating {len(todo)} session snapshot(s); first run downloads "
          f"{MODEL_ID} (~0.5 GB) to the HF cache")
    model, processor, device = _load_model()
    print(f"model loaded on {device}")

    with open(OUT, "a") as f:
        for snap in todo:
            image = Image.open(snap).convert("RGB")
            cap = _run_task(model, processor, device, image,
                            "<MORE_DETAILED_CAPTION>")
            od = _run_task(model, processor, device, image, "<OD>")
            labels = sorted(set(od.get("<OD>", {}).get("labels", [])))
            caption = cap.get("<MORE_DETAILED_CAPTION>", "").strip()
            low = caption.lower() + " " + " ".join(labels).lower()
            rec = {
                "session": snap.stem,
                "caption": caption,
                "objects": labels,
                # coarse condition flags for slicing
                "glasses": "glasses" in low,
                "lamp_or_window": any(k in low for k in
                                      ("lamp", "window", "sunlight", "light")),
                "dark": any(k in low for k in ("dark", "dim", "night")),
            }
            f.write(json.dumps(rec) + "\n")
            f.flush()
            print(f"  {snap.stem}: {', '.join(labels) or 'no objects'} | "
                  f"{caption[:80]}")
    print(f"done -> {OUT}")
