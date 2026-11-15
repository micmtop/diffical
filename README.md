# Diffical

## 简介

OpenUtau 的实验性歌声渲染器：一个 3.54M 的小模型把每个音素的 mel 片段装配成整句
log-mel，再由冻结的 PC-NSF-HiFiGAN 声码器出声（44.1 kHz / hop 512 / 128 bins）。
拼接在 mel 域完成，波形上不做交叉淡化。

v0.1，研究阶段，能跑通但有可听瑕疵；模型权重暂不发布，代码 Apache-2.0。

## 前后端分仓说明

- **后端**（本仓库，Python）：模型、训练、评测、HTTP 服务，是唯一代码源。
- **前端**：[micmtop/OpenUtau](https://github.com/micmtop/OpenUtau) 的 `diffical`
  分支，只有 Diffical 渲染器补丁（`OpenUtau.Core/Diffical/` 3 个文件 +
  `Renderers.cs` 一行注册）；Actions 每天同步上游，并构建 Windows / macOS / Linux。
- 两者只通过 HTTP 通信（默认 `http://127.0.0.1:8573/`），改后端不用重编译 app。

## 工作原理

```
OpenUtau ──HTTP /render──▶ 后端：音素切片/拉伸 → mel → PointerGen 装配 → 冻结声码器 → 波形
```

UTAU 声库是「一个音素一段录音」。传统渲染靠交叉淡化拼接，接缝受 oto 标注和两段
录音的相位/电平影响。PointerGen 在 mel 域看整句上下文和 F0，决定每帧取哪段录音的
哪个位置、边界附近补什么；声码器冻结，模型只学装配，不重新学音色。

## 模型

`model/pointer.py` 的 PointerGen：3.54M 参数（`c=416`，`n_ctx=3`）。

- 输入 `segs / seg_len / seg_idx / progress / boundary / f0`
- 输出 `gather_seg(segs, τ) + tanh(residual)`，`τ = progress + tanh(Δτ) · 0.5`

训练口径：`target` 是原录音 mel；片段只取自目标录音自身；40% 的句子注入 oto 式缺陷
（偏移/截短/裁头/长辅音）；F0 用 pyworld 从音频提取，不用 MIDI 台阶。

## 下载安装使用

**声码器**：从 [openvpi/pc-nsf-hifigan](https://huggingface.co/openvpi/pc-nsf-hifigan)
下载 `pc_nsf_hifigan_44.1k_hop512_128bin_2025.02`，放到 `models/`（`model.onnx` +
`config.json`）。许可 CC BY-NC-SA 4.0，不可商用。

**后端**：

```bash
pip install -r requirements.txt
cp docs/config.example.yaml docs/config.yaml   # 改声码器路径、device
python -m backend.server                        # http://127.0.0.1:8573
```

**OpenUtau**：用 fork 的 Actions artifact（`OpenUtau-win-x64` / `-osx-arm64` /
`-linux-x64`），或自行编译 `diffical` 分支；渲染器选 `DIFFICAL`。

**声库**：标准 UTAU 声库（含 `oto.ini`），放进 OpenUtau 的 Singers 目录，macOS 默认
`~/Library/OpenUtau/Singers/`。

**连接**：OpenUtau 会自动拉起后端，找不到时设 `DIFFICAL_BACKEND_PATH` 和
`DIFFICAL_PYTHON`；也可以先手动启动后端。

**模型权重**：未发布。要用模型装配（`DIFFICAL_ENGINE=model`）需要自训检查点，再用
`DIFFICAL_POINTER_CKPT` 指定；不设时走 legacy 拼接。

## 自训练

```bash
python tools/build_oto_pairs.py --out train/data/oto_full          # OpenCpop
python tools/build_oto_gtsinger.py --root data/gtsinger/Chinese --out train/data/oto_full

python -m train.train_set --arch pointer --width 416 --pairs train/data/oto_full \
  --steps 25000 --batch 4 --device cuda --amp --out train/ckpt_oto_full
```

数据要求：片段来自目标录音自身；F0 从音频提取；`target` 是原录音 mel；训练和推理
条件一致（`progress` 定义、`boundary` 宽度、F0 清理阈值）。

评估：

```bash
python tools/eval_set.py --ckpt <ckpt> --data <pairs> --split eval --n 24
python tools/render_case.py --case <case.json> --engine model --out model.wav
```

## 仓库结构

| 路径 | 内容 |
|---|---|
| `backend/` | HTTP 服务、渲染器、音素/oto 处理、声码器封装 |
| `model/pointer.py` | PointerGen 与 `load_pointer` |
| `train/` | 数据集与训练循环 |
| `tools/` | 数据构造、评估、案例渲染 |
| `docs/config.example.yaml` | 配置模板 |

数据集、模型权重、渲染音频、OpenUtau 构建产物不在仓库里。

## 许可

- 代码：Apache-2.0，见 [LICENSE](LICENSE)
- 声码器权重：CC BY-NC-SA 4.0，不可商用，不随仓库分发；见 [NOTICE](NOTICE)
- 数据集、声库：各自遵循原始许可；Diffical 模型权重暂不发布
