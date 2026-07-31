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

    def __getitem__(self, i):
        r, l, head, tgt = self.samples[i]
        r = r.astype(np.float32) / 255.0
        l = l.astype(np.float32) / 255.0
        if self.train:  # light photometric jitter for lighting robustness
            gain = np.random.uniform(0.85, 1.15)
            bias = np.random.uniform(-0.08, 0.08)
            r = np.clip(r * gain + bias, 0, 1)
            l = np.clip(l * gain + bias, 0, 1)
        return (torch.from_numpy(r).unsqueeze(0),
                torch.from_numpy(l).unsqueeze(0),
                torch.from_numpy(head / 30.0),
                torch.from_numpy(tgt))


def train(dataset_root="data/dataset", out="data/gaze_cnn.pt",
          epochs=40, batch_size=64, lr=3e-4, patience=6):
    from .dataset import load_sessions
    tagged = list(load_sessions(dataset_root, with_session=True))
    if len(tagged) < 300:
        raise SystemExit(
            f"Only {len(tagged)} samples in {dataset_root}. Run "
            "`python -m gazekit calibrate` a couple more times first "
            "(each session adds ~1000).")

    # honest split: hold out the NEWEST session (random row splits leak —
    # neighboring frames are near-duplicates and score optimistically)
    sessions = sorted({s for s, *_ in tagged})
    val_sess = sessions[-1] if len(sessions) >= 2 else None
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
            torch.save(net.state_dict(), out)
            marker = "  * saved"
        else:
            since_best += 1
        print(f"epoch {ep:2d}/{epochs}  val err {val_err:.4f} "
              f"(~{val_err * 100:.1f}% of screen){marker}")
        if since_best >= patience:
            print(f"early stop: no improvement for {patience} epochs")
            break
    print(f"best model -> {out}  (val err {best * 100:.1f}% of screen, "
          "held-out session)")


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
        rt = torch.from_numpy(r.astype(np.float32) / 255.0)[None, None]
        lt = torch.from_numpy(l.astype(np.float32) / 255.0)[None, None]
        hd = torch.tensor([[obs.yaw, obs.pitch, obs.roll]],
                          dtype=torch.float32) / 30.0
        with torch.no_grad():
            out = self.net(rt.to(self.dev), lt.to(self.dev),
                           hd.to(self.dev)).cpu().numpy()[0]
        w, h = self.screen_size
        return np.clip(out * [w, h] + self.bias, [0, 0], [w - 1, h - 1])
