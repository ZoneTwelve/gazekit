"""Publish gazekit artifacts to the Hugging Face Hub.

Implements docs/PUBLISH_STANDARD.md: auth gate -> privacy allowlist
(dataset) -> publish gate (models, same held-out protocol as the deploy
gate) -> generated cards -> journal registration. Models are uploaded only
if they beat the currently published ones; the dataset repo is created
private unless --public is passed.
"""

import hashlib
import json
import tempfile
from pathlib import Path

MODEL_REPO = "ZoneTwelve/gazekit"
DATA_REPO = "ZoneTwelve/gazekit"          # repo_type="dataset" namespace
DATASET_ROOT = Path("data/dataset")

# privacy allowlist — nothing outside these patterns ever reaches the
# dataset repo (context snapshots, journal, config, models all live in
# data/ next door)
DATASET_ALLOW = ["session_*/samples.jsonl", "session_*/crops/*.png",
                 "pruned.json", "README.md"]


def _api():
    try:
        from huggingface_hub import HfApi, get_token
    except ImportError:
        raise SystemExit("huggingface_hub missing — "
                         "uv pip install huggingface_hub")
    if not get_token():
        raise SystemExit("no Hugging Face token — run `hf auth login` "
                         "(or set HF_TOKEN) with a write token first")
    return HfApi()


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _published_hashes(api):
    """rfilename -> sha256 (LFS) or None (small git blob) for MODEL_REPO."""
    try:
        info = api.model_info(MODEL_REPO, files_metadata=True)
    except Exception:
        return {}
    return {s.rfilename: (s.lfs.sha256 if s.lfs else None)
            for s in info.siblings}


def _gate_split():
    """Same split as the deploy gate: webcam records, cleaned in memory,
    newest session with >= 6 distinct targets held out."""
    from .evaluate import clean, load_records
    recs, screen = load_records(DATASET_ROOT, source="0")
    if not recs or screen is None:
        return None, None, None
    kept, _, _ = clean(recs, screen)
    sessions = sorted({r["session"] for r in kept})
    big = [s for s in sessions
           if len({tuple(r["Y"]) for r in kept if r["session"] == s}) >= 6]
    if not big:
        return None, None, None
    newest = big[-1]
    return [r for r in kept if r["session"] == newest], screen, newest


def _ridge_err(model_path, test, screen):
    import numpy as np
    from .evaluate import _aligned_err, _cluster_err
    from .model import GazeModel
    m = GazeModel.load(model_path)
    aligned = _aligned_err(m, test, *screen)
    raw = float(np.mean([c["err"] for c in _cluster_err(m, test)]))
    return (aligned if aligned is not None else float("inf")), raw


def _cnn_err(pt_path, session, screen):
    """Mean px error of a .pt checkpoint on one session's crops."""
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from .cnn import CropDataset, GazeNet, device
    from .dataset import load_sessions
    samples = [rest for s, *rest in
               load_sessions(DATASET_ROOT, with_session=True)
               if s == session]
    if len(samples) < 6:
        return None
    dev = device()
    net = GazeNet().to(dev)
    net.load_state_dict(torch.load(pt_path, map_location=dev))
    net.eval()
    errs = []
    with torch.no_grad():
        for r, l, hd, y in DataLoader(CropDataset(samples, False),
                                      batch_size=64):
            pred = net(r.to(dev), l.to(dev), hd.to(dev)).cpu().numpy()
            d = (pred - y.numpy()) * np.array(screen, dtype=np.float32)
            errs.extend(np.linalg.norm(d, axis=1))
    return float(np.mean(errs))


def _dataset_stats():
    sessions, n, domains, feat_dims = 0, 0, {}, set()
    for jl in sorted(DATASET_ROOT.glob("session_*/samples.jsonl")):
        sessions += 1
        cam = "webcam"
        for line in open(jl):
            rec = json.loads(line)
            if rec.get("meta"):
                cam = "phone" if rec.get("camera") == "phone" else "webcam"
                continue
            n += 1
            if "features" in rec:
                feat_dims.add(len(rec["features"]))
        domains[cam] = domains.get(cam, 0) + 1
    crops = sum(1 for _ in DATASET_ROOT.glob("session_*/crops/*.png"))
    return {"sessions": sessions, "samples": n, "crops": crops,
            "domains": domains, "feature_dims": sorted(feat_dims)}


def _dataset_card(stats):
    return f"""---
license: mit
pretty_name: gazekit personal gaze dataset
tags:
  - gaze-estimation
  - eye-tracking
---

# gazekit personal gaze dataset

Training data collected with [gazekit](https://github.com/ZoneTwelve/gazekit).
**Personalized and single-subject**: one user's eyes, cameras and screen —
a reference/reproduction artifact, not a general-purpose gaze corpus.

- {stats['sessions']} sessions ({', '.join(f"{v} {k}" for k, v in
                                           sorted(stats['domains'].items()))})
- {stats['samples']} labeled samples, {stats['crops']} eye-crop PNGs
- feature vector lengths present: {stats['feature_dims']}

## Layout

- `session_*/samples.jsonl` — first line is meta (`screen_size`, `camera`);
  then one record per sample: `i`, `tag` (calib/vor/posture/edges/probe/
  ambient/...), `target` px, `features`, `yaw`/`pitch`/`roll`, `blink`.
- `session_*/crops/NNNNNN_R.png` / `_L.png` — 64x48 grayscale eye crops
  (input to the CNN backend).
- `pruned.json` — sample ids per session rejected by `gazekit iterate`.

Evaluate session-split only (leave-one-session-out): neighboring frames
are near-duplicates, random row splits score ~2x optimistically.
"""


def publish_dataset(api, public=False):
    from huggingface_hub import upload_folder
    stats = _dataset_stats()
    if not stats["sessions"]:
        return {"dataset": "skipped: no sessions"}
    url = api.create_repo(DATA_REPO, repo_type="dataset",
                          private=not public, exist_ok=True)
    card = DATASET_ROOT / "README.md"
    card.write_text(_dataset_card(stats))
    print(f"dataset: uploading {stats['sessions']} sessions "
          f"({stats['samples']} samples, {stats['crops']} crops) "
          f"-> {url} ({'public' if public else 'private'})")
    upload_folder(repo_id=DATA_REPO, repo_type="dataset",
                  folder_path=str(DATASET_ROOT),
                  allow_patterns=DATASET_ALLOW,
                  ignore_patterns=[".DS_Store"],
                  commit_message=f"sync: {stats['sessions']} sessions, "
                                 f"{stats['samples']} samples")
    return {"dataset": f"uploaded {stats['sessions']} sessions",
            "samples": stats["samples"]}


def publish_models(api):
    """Upload each model file only if it beats the published incumbent on
    the deploy-gate protocol (docs/PUBLISH_STANDARD.md stage 3)."""
    published = _published_hashes(api)
    test, screen, newest = _gate_split()
    out = {}
    for fname in ("gaze_model.pkl", "gaze_cnn.pt"):
        local = Path("data") / fname
        if not local.exists():
            out[fname] = "skipped: no local file"
            continue
        pub_sha = published.get(fname, "absent")
        if pub_sha == _sha256(local):
            out[fname] = "skipped: identical to published"
            continue
        if fname not in published:
            verdict = "upload: first publication"
        elif test is None:
            out[fname] = "kept published: no gate data (need >=1 big session)"
            continue
        else:
            with tempfile.TemporaryDirectory() as td:
                from huggingface_hub import hf_hub_download
                pub = hf_hub_download(MODEL_REPO, fname, local_dir=td)
                if fname.endswith(".pkl"):
                    loc_e, _ = _ridge_err(local, test, screen)
                    pub_e, _ = _ridge_err(pub, test, screen)
                else:
                    loc_e = _cnn_err(local, newest, screen)
                    pub_e = _cnn_err(pub, newest, screen)
            if loc_e is None or pub_e is None:
                out[fname] = "kept published: gate session has no crops"
                continue
            print(f"publish gate ({newest}, aligned): local {loc_e:.0f}px "
                  f"vs published {pub_e:.0f}px — {fname}")
            if loc_e >= pub_e:
                out[fname] = (f"kept published: local {loc_e:.0f}px not "
                              f"better than published {pub_e:.0f}px")
                continue
            verdict = f"upload: local {loc_e:.0f}px beats {pub_e:.0f}px"
        api.upload_file(path_or_fileobj=str(local), path_in_repo=fname,
                        repo_id=MODEL_REPO,
                        commit_message=f"{fname}: {verdict}")
        out[fname] = verdict
    return out


def run(what="all", public=False):
    api = _api()
    result = {}
    if what in ("all", "models"):
        result.update(publish_models(api))
    if what in ("all", "dataset"):
        result.update(publish_dataset(api, public=public))
    for k, v in result.items():
        print(f"  {k}: {v}")
    return result
