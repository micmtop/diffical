# Diffical Neo

An experimental AI singing-voice renderer for [OpenUtau](https://github.com/stakira/OpenUtau).
A small model assembles per-phoneme mel segments into one continuous phrase log-mel, which is
then rendered by a frozen PC-NSF-HiFiGAN vocoder (44.1 kHz, hop 512, 128 bins).

**The model does the assembly.** There is no waveform crossfade stage and no
`MelOverlapAccumulator` in the inference or training input. Segment boundaries are part of
the model's conditioning, and the model is trained to produce the transition itself.

```
OpenUtau (C#) --HTTP 8573--> Python backend
                               phone slicing / oto stretch (hop-128 source clock)
                               -> PointerGen mel assembler
                               -> frozen pc-nsf-hifigan
                               -> 44.1 kHz wav back to OpenUtau
```

## Status

v0.1, research-grade. It runs end-to-end inside OpenUtau and is being validated by ear and by
objective seam metrics. Expect rough edges; see [Limitations](#limitations).

What is implemented:

- **PointerGen assembler** (`model/pointer.py`): 3.54 M params, `c=416`, `n_ctx=3`.
  Inputs `segs [S,128,L]`, `seg_len`, `seg_idx`, `progress`, `boundary`, `f0`.
  Output `out = gather_seg(segs, tau) + tanh(residual)`; `tau = progress + tanh(dtau) * 0.5`,
  so the model can move the read position within a segment.
- **Production geometry at inference** (seg-norm): each segment only keeps the frames actually
  covered by `seg_idx` and its `progress` spans 0..1, matching how training pairs are built.
- **oto-defect mixed training**: 40% of training utterances get oto-like corruption
  (source-window offsets ±6 frames, 10-25% tail truncation, clipped head, long-consonant
  windows) while 60% stay clean. This teaches the model to smooth seams that bad oto causes,
  without dragging down banks whose oto is good.
- **No output "makeup"**: no per-note loudness matching, no envelope gain on the rendered
  waveform, no spectral EQ/compression. OpenUtau's VOL/envelope are *not* applied as
  post-gains; the phrase gets one constant RMS normalization only. (Reference behaviour:
  HiFiUTAU applies VOL/envelope in the mel domain before vocoding, which we follow.)
- **Training data audit and objective tests**: `tests/test_algorithm_audit.py`,
  `tests/test_dataset_audit.py`.

## Repository layout

| Path | What |
|---|---|
| `backend/` | HTTP server, renderer, phone processor, vocoder wrapper, OpenUtau bridge |
| `model/` | PointerGen (`pointer.py`), legacy baselines (`net.py`, `setgen.py`) |
| `tools/` | dataset builders (`build_*_pairs.py`), render/eval/diagnose scripts |
| `train/` | training loop (`train_set.py`), dataset loader, losses |
| `tests/` | objective audits (algorithm invariants, dataset checks) |
| `docs/` | config template, design notes, root-cause write-ups |

Not in the repo: datasets, model checkpoints, rendered audio, the modified OpenUtau app build.
Datasets used for training are **not redistributable here**; check each dataset licence before
redistributing weights trained on it.

## Quick start (backend only)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp docs/config.example.yaml docs/config.yaml
# edit docs/config.yaml: vocoder/hnsep paths, device

python -m backend.server            # serves http://127.0.0.1:8573
```

OpenUtau integration needs a build that implements `IRenderer` and posts render requests to
this backend; that C# side is maintained separately and is not part of this repository.

The launcher scripts (`run_openutau_*.sh`) are local convenience wrappers; they expect a
modified `OpenUtau.app` next to the checkout and a checkpoint under `train/ckpt_*/`.

## Training

Data builders expect local OpenCpop/GTSinger copies (not bundled):

```bash
# production-geometry pairs, optional oto-like defects
python tools/build_oto_pairs.py --utts <npz_dir> --out train/data/oto_full --shard 0 --shards 2
python tools/build_oto_gtsinger.py --root data/gtsinger/Chinese --out train/data/oto_full --shard 0 --shards 2

# 25k steps on an RTX 2060 (~22 it/s -> ~20 min)
python -m train.train_set --arch pointer --width 416 --pairs train/data/oto_full \
  --steps 25000 --batch 4 --device cuda --amp --init ckpt_gan/last.pt \
  --lambda-b 20 --lambda-seam 1 --out ckpt_oto_full
```

Rules the training data must respect:

- Segments are slices of the **target recording itself** (reachable content), never other
  takes of the same phoneme.
- F0 is extracted from audio (pyworld harvest), never from score MIDI steps.
- `target` is always the original recording mel.

## Limitations

- Residual seam jumps: on hard oto/level discontinuities the assembler still leaves a step
  (measured: model output seam jump ~1.5 vs ~0.56 in real recordings).
- Vocoder-inherent start/end transient: a short low-frequency thump at phrase onsets exists in
  both this engine and the legacy concatenation engine; it is not fixed by the mel model.
- Long held notes can sound static on some banks, because the source itself is static after
  stretching.
- oto compensation is learned, not programmatic: content that bad oto removed cannot be
  recovered.

## Licence

Code is released under the MIT licence (see `LICENSE`). Datasets, vocoder weights and voice
banks keep their own licences and are not distributed here.

---

## 中文简介

Diffical Neo 是给 OpenUtau 用的实验性 AI 歌声渲染器：小模型把每个音素的 mel 片段装配成整句
mel，再交给冻结的 PC-NSF-HiFiGAN 出声。装配完全由模型完成，没有波形交叉淡化，也没有
overlap accumulator。包含生产式几何装配、oto 缺陷混合训练（40% 坏 oto + 60% 干净）、以及
不做输出化妆的约束（不做每音符响度匹配/EQ/压缩）。数据集与权重不进仓库。
