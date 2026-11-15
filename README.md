# Diffical

给 OpenUtau 用的歌声渲染后端。模型把每个音素的 mel 片段装配成整句 mel，再交给冻结的
NSF-HiFiGAN 声码器出声（44.1 kHz，hop 512，128 bins）。

装配由模型完成，不做波形交叉淡化，也没有 overlap accumulator。音素边界是模型的输入条件，
过渡由模型自己生成。

```
OpenUtau --HTTP 8573--> 后端
                          音素切片 / oto 拉伸（hop-128 源时钟）
                          -> PointerGen 装配器
                          -> 冻结声码器
                          -> 44.1 kHz 波形
```

## 现在做到什么程度

v0.1。已经在 OpenUtau 里完整跑通，日常可用，但仍是研究阶段。

模型在 271 句留出集上的表现：

| 指标 | 输入直读（画布） | 模型 |
|---|---|---|
| 整句 log-mel L1 | 0.5186 | 0.4698 |
| 边界 ±2 帧误差 | 0.6229 | 0.5440 |
| 接缝处的帧间跳变 | 0.422 | 0.340 |

作为参照，同一批真实录音自己的接缝跳变是 0.442。模型输出的接缝比真实录音更平（0.77 倍），
说明它没有留下比自然演唱更粗糙的接缝。

已知不足：

- 声码器在句首句尾有短促低频冲击，属于声码器本身的边界行为，mel 模型消不掉。
- 长音依赖源素材。源内容静态时，拉伸后听起来会发直。
- 坏 oto 只能部分补偿。已经被 oto 切掉的内容补不回来。

## 性能

纯 CPU（onnxruntime CPUExecutionProvider，Mac M 系列），三次预热后取平均：

| 请求 | 音频时长 | 端到端耗时 | RTF |
|---|---|---|---|
| 短句 | 1.70 s | 1.64 s | 0.97 |
| 长句 | 7.38 s | 2.49 s | 0.34 |

分组件：

| 组件 | 1 s 音频 | 7.4 s 音频 | 17.4 s 音频 |
|---|---|---|---|
| 声码器 | 244 ms | 2.21 s | 5.29 s |
| 装配器（ONNX） | 4.9 ms | 23 ms | 52 ms |

RTF 是渲染耗时除以音频时长，小于 1 表示比实时快。声码器占大头，约 0.24–0.30 RTF，
而且随长度基本线性；装配器可以忽略。三秒以上的句子能实时渲染，一秒级的短句受固定开销
影响，接近但不稳定低于 1。OpenUtau 会缓存每个乐句的 wav，重复播放不重新渲染。

要进一步加速只能从声码器入手。换更快的声码器会牵动音质和许可，目前没有做。

## 模型

PointerGen，3.54M 参数，`c=416`，`n_ctx=3`。

输入：`segs [S,128,L]`、`seg_len`、`seg_idx`、`progress`、`boundary`、`f0`。
输出：`gather_seg(segs, tau) + tanh(residual)`，其中 `tau = progress + tanh(dtau) * 0.5`，
模型可以在片段内移动读取位置。

训练数据用同一段录音自己的切片构造，按 40% 比例注入 oto 式缺陷（起点偏移、尾部截断、
头部裁切、长辅音占窗），其余保持干净。目标是原录音的 mel。

训练脚本每 1000 步在固定留出集上评估并保存 best 权重。数据审计脚本检查训练/验证是否
重叠、字段是否完整，红了一律不训。

## 目录

| 路径 | 内容 |
|---|---|
| `backend/` | HTTP 服务、渲染器、音素处理、声码器封装 |
| `model/` | PointerGen |
| `tools/` | 数据构造、渲染、评估、ONNX 导出 |
| `train/` | 训练循环与数据加载 |
| `docs/` | 配置模板 |
| `export/` | ONNX 产物（本地生成，不入库） |

## 开始

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp docs/config.example.yaml docs/config.yaml
# 编辑 docs/config.yaml：声码器、hnsep、设备

python -m backend.server            # 监听 127.0.0.1:8573
```

声码器权重不随仓库分发，按 `docs/config.example.yaml` 里的路径放好。

OpenUtau 侧的接入代码单独维护，见 `NOTICE` 里的归属说明。

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

Rust 或其它运行时用 ONNX 推理：

```bash
python tools/export_onnx.py --ckpt train/ckpt_new/best.pt --out export/pointer.onnx
```

导出脚本会检查动态 shape 下的数值一致性（阈值 1e-4），并把 ckpt/onnx 的 sha256、输入输出、
mel 契约写进同名 json。当前实测 parity 1e-5 量级，CPU 上 1 秒音频的装配约 5 ms。

## 权重与许可

- 代码：Apache-2.0，见 `LICENSE`。
- 模型权重：CC BY-NC-SA 4.0，与所用声码器一致。不随仓库分发，见 GitHub Releases。
- 数据集、声码器权重、声库：各自遵循原始许可，不随本仓库分发。

权重是训练产物，包含数据集与声码器的衍生关系，再分发时保留同样的非商用与署名条款。
