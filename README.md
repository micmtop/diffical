# Diffical

基于 PointerGen 拼接模型 + 各种 Flag + NSF-HiFiGAN = 自然的歌声

```
OpenUtau --HTTP 8573--> 后端
                          oto 窗口取内容（hop-128 源时钟）-> 按 envelope 跨度放置
                          -> 重叠区按包络增益合成
                          -> 冻结声码器
                          -> 44.1 kHz 波形
```

## Flag 支持

支持曲线Flag （使用前记得去同步！不然没有地方用！）。

当前生效的 flag (注意多是曲线)：

| Flag | 范围 | 默认 | 后端行为 |
|---|---|---|---|
| `Hb` | 0–500 | 100 | 气声，`(Hb-100)/5` 映射到 0–100 |
| `Hv` | 0–150 | 100 | 浊音，`Hv/1.5` |
| `Ht` | -100–100 | 0 | 张力，`(Ht+100)/2` |
| `HG` | 0–100 | 0 | 咆哮强度 |
| `g` | -100–100 | 0 | 性别（共振峰偏移） |
| `Hu` | 0–100 | 0 | 真人化强度，见下一节 |

`Hb`、`Hv`、`Ht` 走谐波-噪声分离，只在值偏离默认时启用。`g` 和 `HG` 直接作用在音素上。

其他 flag（`P`、`Mo`、`Mt`、`Mb`、`Ms`、`Mm`、`B`、`H`）会收到并记录，但不改变输出。
表外的 flag 忽略；值为 NaN 或 Inf 时跳过该条，不影响整句。

## HUM（Humanization）

这是我想出来的 Flag，还是比较创新的，因此单独提出来讲（本质上也是 Flag）。

HUM 是数值表达式（`Humanization`，abbr `HUM`，flag `Hu`），全局设一次、可按音符微调，
不需要画曲线。后端读音符时长和旋律走向生成一条慢速包络，按 0–100 缩放，在装配完成、
进声码器之前施加：音量通道给慢速增益起伏，音色通道用同一条包络做谱倾斜。

`Hu=0` 时不生效，未设置时默认 0，可用 `DIFFICAL_HUM_DEFAULT` 改。工程里若存成 Curve
类型的 `hum` 曲线，后端也读，两条路径取其一。

## AIGC提示

本项目使用了 AI 辅助编码（即人工提出项目框架 + AI实现代码 + 人工复核）。

这个项目不做拿 UTAU 声库训练或蒸馏音色模型的事，走的是神经 resampler 路线：内容仍然是声库里的真实录音，仅在接缝处做了处理。渲染前请核对 UTAU 声库协议。

## 现在做到什么程度

v0.1。已经在 OpenUtau 里完整跑通，日常可用，但仍是研究阶段。

PointerGen 装配器（默认路径）在 271 句留出集上的表现：

| 指标 | 输入直读（画布） | 模型 |
|---|---|---|
| 整句 log-mel L1 | 0.5186 | 0.4698 |
| 边界 ±2 帧误差 | 0.6229 | 0.5440 |
| 接缝处的帧间跳变 | 0.422 | 0.340 |

同一批真实录音自己的接缝跳变是 0.442；模型输出是 0.340，比真实录音更平（0.77 倍）。

装配只由 PointerGen 完成。`DIFFICAL_GEOM_V3=1` 打开的是几何对照实现（envelope 时间轴 + 重叠区功率域合成），
只用于实验比较，不是默认路径，也不作为降级方案。

已知不足：

- 差 oto 补偿不是万能的：已经被 oto 切掉的内容补不回来。
- 项目初期不稳定。 

## 核心拼接模型

PointerGen，3.54M 参数，`c=416`，`n_ctx=3`。

输入：`segs [S,128,L]`、`seg_len`、`seg_idx`、`progress`、`boundary`、`f0`。
输出：`gather_seg(segs, tau) + tanh(residual)`，其中 `tau = progress + tanh(dtau) * 0.5`，模型可以在片段内移动读取位置。

## 目录

| 路径 | 内容 |
|---|---|
| `backend/` | HTTP 服务、渲染器、音素处理、声码器封装 |
| `model/` | PointerGen |
| `tools/` | 数据构造、渲染、评估、ONNX 导出 |
| `train/` | 训练循环与数据加载 |
| `docs/` | 配置模板 |
| `export/` | ONNX 产物（本地生成，不入库） |

## 使用方法

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp docs/config.example.yaml docs/config.yaml
# 编辑 docs/config.yaml：声码器、hnsep、设备

python -m backend.server            # 监听 127.0.0.1:8573
```

声码器权重不随仓库分发，按 `docs/config.example.yaml` 里的路径放好。

OpenUtau 侧的接入代码单独维护（由我）。

## 训练

构造器需要自备语料与转写，默认读取 `data/wavs` 和 `data/transcriptions.txt`：

```bash
python tools/build_pairs.py --wavs-dir data/wavs --transcriptions data/transcriptions.txt \
  --out train/data/pairs --shard 0 --shards 2
python tools/build_pairs_multilingual.py --root data/multilingual --out train/data/pairs

python -m train.train_set --arch pointer --width 416 --pairs train/data/pairs \
  --steps 20000 --batch 4 --lr 1e-4 --device cuda --amp --init ckpt_base/last.pt \
  --lambda-b 8 --eval-every 1000 --save-best --out ckpt_new
```

## ONNX

也可以走 ONNX 推理：

```bash
python tools/export_onnx.py --ckpt train/ckpt_new/best.pt --out export/pointer.onnx
```

导出脚本会检查动态 shape 下的数值一致性（阈值 1e-4），并把 ckpt/onnx 的 sha256、输入输出、mel 契约写进同名 json。

## 权重与许可

- 代码：Apache-2.0，见 `LICENSE`。
- 模型权重：CC BY-NC-SA 4.0。不随仓库分发，见 GitHub Releases。
- 数据集、声码器权重、声库：各自遵循原始许可，不随本仓库分发。

权重来自训练，跟数据集和声码器有衍生关系；再分发时沿用同样的非商用和署名条款。

## README 提示

README采用 AI + 人工复核重写的方式，可能不是最新的，但是在正式发布前一定会同步最新状态。
