"""MobileNetV2 gaze regressor, post-trained (fine-tuned) on YOUR calibration
dataset — the appearance-based upgrade over the landmark/ridge baseline.

Architecture (iTracker-style, slimmed):
    right-eye crop (64x48 gray) ─┐
    left-eye crop  (64x48 gray) ─┼─ shared MobileNetV2 backbone (ImageNet
                                 │  weights, first conv adapted to 1 channel)
    head pose (yaw,pitch,roll) ──┴─ MLP head -> (x, y) in [0,1] screen coords

Train:   python -m gazekit train-cnn        (needs >= ~2 calibration sessions)
Use:     python -m gazekit live --backend cnn
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2


def device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class GazeNet(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
        # adapt first conv to 1-channel input by summing pretrained RGB kernels
        old = backbone.features[0][0]
        new = nn.Conv2d(1, old.out_channels, old.kernel_size, old.stride,
                        old.padding, bias=False)
        with torch.no_grad():
            new.weight.copy_(old.weight.sum(dim=1, keepdim=True))
        backbone.features[0][0] = new
        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Linear(1280 * 2 + 3, 256), nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 64), nn.ReLU(inplace=True),
            nn.Linear(64, 2),
        )

    def _embed(self, x):
        return self.pool(self.features(x)).flatten(1)

    def forward(self, right, left, head):
        z = torch.cat([self._embed(right), self._embed(left), head], dim=1)
        return self.head(z)


class CropDataset(Dataset):
    def __init__(self, samples, train: bool):
        self.samples = samples
        self.train = train

    def __len__(self):
        return len(self.samples)

    _clahe = None

    @classmethod
    def _eq(cls, img):
        # CLAHE over global hist-eq: local equalization handles the
        # directional shadows MPIIGaze names as a top error source, without
        # blowing out the iris. Applied at load time — stored crops stay raw.
        import cv2
        if cls._clahe is None:
            cls._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        return cls._clahe.apply(img)

    def __getitem__(self, i):
        r, l, head, tgt = self.samples[i]
        r = self._eq(r).astype(np.float32) / 255.0
        l = self._eq(l).astype(np.float32) / 255.0
        if self.train:
            # strong photometric jitter: gain, bias, gamma — the model must
            # not key on the room's lighting
            gain = np.random.uniform(0.8, 1.2)
            bias = np.random.uniform(-0.1, 0.1)
            gamma = np.random.uniform(0.7, 1.4)
            r = np.clip((r * gain + bias), 0, 1) ** gamma
            l = np.clip((l * gain + bias), 0, 1) ** gamma
            # random erasing: occlude a patch so no single texture region
            # (brow, skin, glasses edge) becomes load-bearing
            for img in (r, l):
                if np.random.random() < 0.3:
                    eh = np.random.randint(6, 16)
                    ew = np.random.randint(8, 22)
                    y0 = np.random.randint(0, img.shape[0] - eh)
                    x0 = np.random.randint(0, img.shape[1] - ew)
                    img[y0:y0 + eh, x0:x0 + ew] = img.mean()
        return (torch.from_numpy(r).unsqueeze(0),
                torch.from_numpy(l).unsqueeze(0),
                torch.from_numpy(head / 30.0),
                torch.from_numpy(tgt))


def train(dataset_root="data/dataset", out="data/gaze_cnn.pt",
          epochs=40, batch_size=64, lr=3e-4, patience=6,
          exclude_tags=("pursuit",)):
    # pursuit excluded by default: ablation on a 887-sample held-out session
    # scored 13.7% (without) vs 14.5% (with) — the lag-compensated labels
    # hurt more than the extra coverage helps
    from .dataset import load_sessions
    tagged = list(load_sessions(dataset_root, with_session=True,
                                exclude_tags=exclude_tags))
    if len(tagged) < 300:
        raise SystemExit(
            f"Only {len(tagged)} samples in {dataset_root}. Run "
            "`python -m gazekit calibrate` a couple more times first "
            "(each session adds ~1000).")

    # honest split: hold out the newest session with enough samples to give
    # a stable validation signal (random row splits leak — neighboring
    # frames are near-duplicates and score optimistically; tiny sessions
    # make early stopping a coin flip)
    counts = {}
    for s, *_ in tagged:
        counts[s] = counts.get(s, 0) + 1
    big = [s for s in sorted(counts) if counts[s] >= 200]
    val_sess = big[-1] if big else (sorted(counts)[-1] if len(counts) >= 2
                                    else None)
    train_samples = [rest for s, *rest in tagged if s != val_sess]
    val_samples = [rest for s, *rest in tagged if s == val_sess]
    if not val_samples:  # single session: fall back to a random tail
        cut = max(int(0.85 * len(train_samples)), 1)
        train_samples, val_samples = train_samples[:cut], train_samples[cut:]
    train_ds = CropDataset(train_samples, True)
    val_ds = CropDataset(val_samples, False)

    dev = device()
    print(f"training on {dev} — {len(tagged)} samples "
          f"({len(train_ds)} train / {len(val_ds)} val, "
          f"val session: {val_sess or 'tail split'})")
    net = GazeNet().to(dev)
    # freeze early backbone blocks; fine-tune the rest + head
    for p in net.features[:8].parameters():
        p.requires_grad = False
    opt = torch.optim.AdamW(
        [p for p in net.parameters() if p.requires_grad], lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.SmoothL1Loss()

    tl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    vl = DataLoader(val_ds, batch_size=batch_size)
    best = float("inf")
    since_best = 0
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    tmp_out = str(out) + ".tmp"
    for ep in range(1, epochs + 1):
        net.train()
        for r, l, hd, y in tl:
            r, l, hd, y = r.to(dev), l.to(dev), hd.to(dev), y.to(dev)
            opt.zero_grad()
            loss = loss_fn(net(r, l, hd), y)
            loss.backward()
            opt.step()
        sched.step()

        net.eval()
        errs = []
        with torch.no_grad():
            for r, l, hd, y in vl:
                pred = net(r.to(dev), l.to(dev), hd.to(dev)).cpu()
                errs.append(torch.linalg.norm(pred - y, dim=1))
        val_err = torch.cat(errs).mean().item()  # in screen-normalized units
        marker = ""
        if val_err < best:
            best = val_err
            since_best = 0
            torch.save(net.state_dict(), tmp_out)
            marker = "  * saved"
        else:
            since_best += 1
        print(f"epoch {ep:2d}/{epochs}  val err {val_err:.4f} "
              f"(~{val_err * 100:.1f}% of screen){marker}")
        if since_best >= patience:
            print(f"early stop: no improvement for {patience} epochs")
            break

    # promote gate: never blindly overwrite the deployed CNN — score the
    # incumbent on THIS run's validation set and keep the winner
    import os
    prev_err = float("inf")
    if Path(out).exists():
        try:
            prev = GazeNet().to(dev)
            prev.load_state_dict(torch.load(out, map_location=dev))
            prev.eval()
            errs = []
            with torch.no_grad():
                for r, l, hd, y in vl:
                    pred = prev(r.to(dev), l.to(dev), hd.to(dev)).cpu()
                    errs.append(torch.linalg.norm(pred - y, dim=1))
            prev_err = torch.cat(errs).mean().item()
        except Exception:
            pass
    if prev_err <= best:
        os.remove(tmp_out)
        print(f"deployed CNN kept: previous model scores "
              f"{prev_err * 100:.1f}% on this val set vs new "
              f"{best * 100:.1f}%")
    else:
        os.replace(tmp_out, out)
        print(f"new model promoted -> {out}  (val err {best * 100:.1f}% vs "
              f"previous {prev_err * 100:.1f}%)")


class CnnPredictor:
    """Inference wrapper used by live mode."""

    def __init__(self, path, screen_size):
        self.dev = device()
        self.net = GazeNet().to(self.dev)
        self.net.load_state_dict(torch.load(path, map_location=self.dev))
        self.net.eval()
        self.screen_size = screen_size
        self.bias = np.zeros(2)

    def predict(self, obs) -> np.ndarray | None:
        if obs.eye_crops is None:
            return None
        r, l = obs.eye_crops
        rt = torch.from_numpy(
            CropDataset._eq(r).astype(np.float32) / 255.0)[None, None]
        lt = torch.from_numpy(
            CropDataset._eq(l).astype(np.float32) / 255.0)[None, None]
        hd = torch.tensor([[obs.yaw, obs.pitch, obs.roll]],
                          dtype=torch.float32) / 30.0
        with torch.no_grad():
            out = self.net(rt.to(self.dev), lt.to(self.dev),
                           hd.to(self.dev)).cpu().numpy()[0]
        w, h = self.screen_size
        return np.clip(out * [w, h] + self.bias, [0, 0], [w - 1, h - 1])
