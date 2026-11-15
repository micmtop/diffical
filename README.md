# Diffical

给 [OpenUtau](https://github.com/stakira/OpenUtau) 用的实验性歌声渲染器：
一个小模型把每个音素的 mel 片段装配成整句连续的 log-mel，再交给冻结的
PC-NSF-HiFiGAN 声码器出声（44.1 kHz / hop 512 / 128 bins）。

拼接发生在 mel 域，波形上不做交叉淡化。片段边界是模型的条件输入，过渡由模型生成；
模型装配路径不经过 `MelOverlapAccumulator`（它只保留在 legacy 对照路径）。

**状态：v0.1，研究阶段。** 已经能在 OpenUtau 里完整跑通，但仍有可听瑕疵，见「已知局限」。

本仓库是后端（Python）。OpenUtau 的界面补丁在单独的 fork：
[micmtop/OpenUtau](https://github.com/micmtop/OpenUtau)（默认分支 `diffical`）。

## 工作原理

```
OpenUtau (C#) ── HTTP POST /render ──▶ Python 后端
                                         ├─ 按音素切 mel、按音符时长拉伸（源 hop 128）
                                         ├─ PointerGen：整句 mel 装配
                                         └─ pc-nsf-hifigan（冻结）→ 44.1 kHz 波形
```

UTAU 声库是「一个音素一段录音」。传统渲染在拼接处做交叉淡化，接缝是否自然取决于
oto 标注和两段录音的相位/电平，经常留下台阶或断层。

Diffical 把拼接放进 mel 域：模型能看到整句上下文和 F0 曲线，自己决定每帧取哪段
录音的哪个位置、边界附近补什么。声码器保持冻结，所以模型只需要学「怎么放」，
不需要重新学音色。

## 模型

`model/pointer.py` 的 PointerGen：

- 输入：`segs [S,128,L]`（每个音素的 log-mel 片段）、`seg_len`、`seg_idx`（每帧
  属于哪一段）、`progress`（段内进度 0..1）、`boundary`（边界 ±4 帧）、`f0`
  （实测 F0，0 = 清音）
- 输出：`gather_seg(segs, τ) + tanh(residual)`，其中
  `τ = progress + tanh(Δτ) · 0.5`，在段内做可微的双线性插值
- 3.54M 参数（`c=416`，`n_ctx=3`）；残差分支的感受野覆盖边界两侧

训练口径：

- `target` 永远是原录音的整句 mel，不是拼接结果
- 片段只取自目标录音自身（内容可达）；不能拿别的 take 的同音素当片段
- 40% 的句子注入 oto 式缺陷：窗偏移 ±6 帧、尾部截短 10–25%、头部裁 1–3 帧、
  长辅音占窗；其余 60% 保持干净。目的是学会抹平坏 oto 造成的接缝，同时不拖累
  标得好的声库
- F0 用 pyworld harvest 从音频提取，清掉 <55 Hz / >1100 Hz；不用曲谱 MIDI 造台阶

推理时使用生产几何（seg-norm）：每段只保留 `seg_idx` 实际覆盖的帧，`progress`
走满 0..1，与训练对一致。

后端不对渲染波形做「化妆」：没有每音符响度匹配、包络增益、EQ 或压缩；整句只做
一个常数 RMS 归一化。

## 实测

内部审计，200 句留出集（OpenCpop + GTSinger，中/英/日）。三条臂使用同一批片段，
分别用画布（片段按原位硬放回）、交叉淡化和模型装配。审计脚本未随仓库发布：

| 指标 | 画布 | 交叉淡化 | 模型 |
|---|---|---|---|
| 边界误差（±2 帧，越低越好） | 0.5487 | 0.5485 | **0.4574（−16.6%）** |
| 内容保真（输出 Δmel / 输入 Δmel，≥0.8） | — | — | **0.81** |
| 高频占比 >8 kHz（真实录音 0.0902） | — | — | **0.0915** |

边界误差只统计接缝 ±2 帧；内容保真跌破 0.8 说明模型开始抹平细节，应当回退。
mel-L1 和 MCD 会奖励平滑，不能单独当验收指标。

跑 `tools/eval_set.py` 可以在自己的数据上得到 mel-L1 和 `content_keep`：

```bash
python tools/eval_set.py --ckpt train/ckpt_gan/last.pt \
  --data train/data/all_v3 --split eval --n 24
```

## 安装

### 1. 声码器

后端至少需要 pc-nsf-hifigan 声码器（44.1 kHz / hop 512 / 128 bins）：

- 从 https://huggingface.co/openvpi/pc-nsf-hifigan 下载
  `pc_nsf_hifigan_44.1k_hop512_128bin_2025.02`
- 放到仓库的 `models/` 下，保留 `model.onnx` 和 `config.json`：

  ```
  models/pc_nsf_hifigan_44.1k_hop512_128bin_2025.02/
  ├── model.onnx
  └── config.json
  ```

- 许可：CC BY-NC-SA 4.0，**不可商用**。本仓库不分发这些权重

`hnsep`（CascadedNet）是可选的：只有画了 breathiness / tension / voicing 等曲线
才会用到；没有模型时有内置回退。需要时把权重放到 `models/hnsep/vr/`，或在配置里
写绝对路径。

### 2. 后端

Python 3.10+。在仓库根目录运行（配置里的相对路径以当前目录为准）：

```bash
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp docs/config.example.yaml docs/config.yaml
# 编辑 docs/config.yaml：vocoder_path、hnsep 路径、device（cpu / cuda / mps）

python -m backend.server
```

默认监听 `http://127.0.0.1:8573`。自检：

```bash
curl -s http://127.0.0.1:8573/status
```

首次启动要加载 ONNX 声码器，大约几秒到十几秒。如果提示端口被占用，先确认是不是
已有后端在跑：`lsof -nP -iTCP:8573 -sTCP:LISTEN`。

### 3. OpenUtau

界面侧是 OpenUtau 的 fork：**https://github.com/micmtop/OpenUtau**，默认分支 `diffical`。

获取构建的两种方式：

- GitHub Actions → **Build (win/osx/linux)** → 最新一次 run → artifacts：
  `OpenUtau-win-x64` / `OpenUtau-osx-arm64` / `OpenUtau-linux-x64`
  （artifact 有保留期，过期后重新触发 workflow 或自行编译）
- 或按 OpenUtau 官方方式自行编译 `diffical` 分支

fork 里只有 `OpenUtau.Core/Diffical/`（客户端、协议、渲染器）和
`OpenUtau.Core/Render/Renderers.cs` 里的一行注册；自动音高（AutoPitch /
PrettyPitch）不在这个 fork 里。上游同步 workflow 每天 03:00 UTC 跑一次。

### 4. 声库

标准 UTAU 声库（含 `oto.ini`）即可。放进 OpenUtau 的 Singers 目录，macOS 默认是
`~/Library/OpenUtau/Singers/`。新建音轨后把渲染器切成 `DIFFICAL`。

### 5. 让 OpenUtau 找到后端

OpenUtau 会尝试自动拉起后端，查找顺序：

1. 环境变量 `DIFFICAL_BACKEND_PATH`
2. `<OpenUtau 根目录>/diffical/backend`
3. `<OpenUtau 根目录>/backend`
4. `~/.diffical/backend`
5. 从 App 所在目录向上找 `diffical/backend` 或 `backend`

最稳的是显式指定：

```bash
export DIFFICAL_BACKEND_PATH=/path/to/diffical/backend
export DIFFICAL_PYTHON=/path/to/python    # 需要能 import aiohttp
```

也可以先按「2. 后端」手动启动，再开 OpenUtau —— 已经在 8573 上监听的进程会被复用。

## 配置与环境变量

`docs/config.yaml` 里常用的几项：

| 键 | 说明 |
|---|---|
| `model.vocoder_path` | ONNX 声码器路径；相对路径以启动目录为准 |
| `model.device` | `cpu` / `cuda` / `mps`。ONNX 声码器只支持 CPU，选 cuda 会回退到 `.ckpt` 模式 |
| `audio.hop_size` | 必须 512，和声码器、训练保持一致 |
| `server.port` | 默认 8573（与 hifisampler 的 8572 错开） |

环境变量：

| 变量 | 作用 |
|---|---|
| `DIFFICAL_CONFIG` | 配置文件路径。默认 `./config.yaml`，找不到时用 `docs/config.yaml` |
| `DIFFICAL_ENGINE` | `model` = 模型装配；不设或其它值 = 传统拼接（legacy） |
| `DIFFICAL_POINTER_CKPT` | PointerGen 权重，默认 `train/ckpt_gan/last.pt` |
| `DIFFICAL_SEAM_RAMP` | 接缝安全过渡帧数，默认 0（关闭）；把单帧台阶摊到 2k+1 帧，做 A/B 时用 |
| `DIFFICAL_SEG_NORM` | 默认 1，生产几何；设为 0 关闭 |
| `DIFFICAL_SERVER_URL` | 前端用的后端地址，默认 `http://127.0.0.1:8573/` |
| `DIFFICAL_BACKEND_PATH` / `DIFFICAL_PYTHON` | 前端拉起后端时用 |

## 模型权重

Diffical 自己训练的 PointerGen 权重**暂不发布**（正在重训更好的版本）。

要用 `DIFFICAL_ENGINE=model`，需要自己训一份检查点，然后：

```bash
DIFFICAL_POINTER_CKPT=/path/to/last.pt python -m backend.server
```

不设 `DIFFICAL_ENGINE=model` 时走 legacy 拼接路径，不需要这份权重。

## 训练

数据集不随仓库分发。构造器读本地的 OpenCpop / GTSinger 副本：

```bash
# OpenCpop：需要 data/opencpop/{transcriptions.txt,wavs/}
python tools/build_oto_pairs.py --out train/data/oto_full

# GTSinger：48k mono，按歌曲目录划分 train/eval
python tools/build_oto_gtsinger.py --root data/gtsinger/Chinese \
  --out train/data/oto_full
```

数据多时用 `--shard 0 --shards 2` 拆成两份并行跑。

输出是 `train/data/<set>/<split>/*.npz`，每个文件包含 `segs`、`seg_len`、`seg_idx`、
`progress`、`boundary`、`f0`、`target`、`durations`。`--split` 只决定写到哪个子目录，
留出集要自己按句子划分（不要和训练句重叠）。

训练：

```bash
python -m train.train_set --arch pointer --width 416 --pairs train/data/oto_full \
  --steps 25000 --batch 4 --device cuda --amp \
  --lambda-b 20 --lambda-seam 1 --out train/ckpt_oto_full
```

有基线权重时加 `--init train/ckpt_gan/last.pt` 继续训。RTX 2060 Laptop 上 25k 步
约 20 分钟（~22 it/s）；CPU 也能跑，只是慢。

数据规则（踩过的坑）：

- 片段只能来自目标录音自身，否则内容不可达，模型会退化成回归均值 = 过平滑
- F0 必须从音频提取，不能拿曲谱 MIDI 造台阶；人声是连续曲线（颤音、滑音）
- `target` 永远是原录音的 mel
- 训练和推理条件必须一致：`progress` 定义、`boundary` 宽度、F0 清理阈值

## 评测

```bash
# mel-L1 / 恒等指针基线 / content_keep
python tools/eval_set.py --ckpt train/ckpt_gan/last.pt \
  --data train/data/all_v3 --split eval --n 24

# 用真实 OpenUtau 请求做 A/B（legacy vs model）
python tools/render_case.py --case <case.json> --engine model --out model.wav
python tools/render_case.py --case <case.json> --engine legacy --out legacy.wav
```

换成自己的检查点和数据目录即可。`eval_set.py` 打印 constant / identity_gather /
model 三个 L1 和 `content_keep`。`render_case.py` 的 case JSON 是 OpenUtau 发来的
`/render` 请求体（可自行抓取或构造）。

## 仓库结构

| 路径 | 内容 |
|---|---|
| `backend/` | HTTP 服务、渲染器、音素/oto 处理、声码器与 hnsep 封装 |
| `model/pointer.py` | PointerGen 定义与 `load_pointer`（按权重形状推断 `c` / `n_ctx`） |
| `train/` | 数据集（`dataset_set.py`）、训练循环（`train_set.py`）、损失 |
| `tools/` | 数据构造（`build_oto_*.py`）、评测（`eval_set.py`）、案例渲染（`render_case.py`） |
| `docs/config.example.yaml` | 配置模板 |

**不在仓库里**：数据集、模型权重、渲染音频、OpenUtau 构建产物。

## 已知局限

- **接缝残余**：遇到硬 oto 或电平不连续时，装配器仍会留下台阶（实测模型输出接缝
  跳变约 1.5，真实录音约 0.56）
- **声码器起止冲击**：句首/句尾有短促低频冲击，这是声码器在片段边界的固有行为，
  旧拼接引擎同样存在，mel 模型消不掉
- **长音发直**：部分声库长音拉伸后源内容本身就静态，听起来会「电」
- **oto 补偿是学出来的**：坏 oto 已经切掉的内容补不回来，只能把剩余内容放顺
- **权重未发布**：现在只能自己训练；换声库/语种的泛化还在验证

## 许可

- 代码：**Apache-2.0**，见 [LICENSE](LICENSE)
- 声码器权重（pc-nsf-hifigan）：OpenVPI 的 **CC BY-NC-SA 4.0**，不可商用，
  不随本仓库分发；详见 [NOTICE](NOTICE)
- 数据集、声库：各自遵循原始许可，不随本仓库分发
- Diffical 模型权重：暂不发布

## 致谢

- [OpenUtau](https://github.com/stakira/OpenUtau)（MIT）
- [OpenVPI PC-NSF-HiFiGAN](https://huggingface.co/openvpi/pc-nsf-hifigan)（声码器）
- [hifisampler](https://github.com/openhachimi/hifisampler)（参数语义与流程参考）
- HiFiUTAU-engine（VOL/envelope 的 mel 域处理思路参考）
