# Diffical

这是个给 OpenUtau 用的歌声渲染后端。

模型把每个音素的 mel 片段装配成整句 mel，再交给冻结的 NSF-HiFiGAN 声码器出声（44.1 kHz，hop 512，128 bins）。

模型负责装配：不做波形交叉淡化，也不用 overlap accumulator。音素边界只作为条件输入，过渡由模型生成。

```
OpenUtau --HTTP 8573--> 后端
                          音素切片 / oto 拉伸（hop-128 源时钟）
                          -> PointerGen 装配器
                          -> 冻结声码器
                          -> 44.1 kHz 波形
```

## AIGC提示

本项目使用了 AI 辅助编码。

这个项目不做 RVC、DiffSVC、So-VITS 那种拿 UTAU 声库训练或蒸馏音色模型的事，走的是 hifisampler 类似的神经 resampler 路线。渲染前请核对 UTAU 声库协议。

## 现在做到什么程度

v0.1。已经在 OpenUtau 里完整跑通，日常可用，~~但仍是研究阶段~~研究得差不多了，但应该还是有特别小的瑕疵。

模型在 271 句留出集上的表现：

| 指标 | 输入直读（画布） | 模型 |
|---|---|---|
| 整句 log-mel L1 | 0.5186 | 0.4698 |
| 边界 ±2 帧误差 | 0.6229 | 0.5440 |
| 接缝处的帧间跳变 | 0.422 | 0.340 |

同一批真实录音自己的接缝跳变是 0.442；模型输出是 0.340，比真实录音更平（0.77 倍）。

已知不足：

- ~~声码器在句首句尾有短促低频冲击，属于声码器本身的边界行为，mel 模型消不掉。~~（这个比较随机，现在比较少了）
- 差 oto 补偿不是万能的：已经被 oto 切掉的内容补不回来。

## 性能

纯 CPU（onnxruntime CPUExecutionProvider，Mac M1 系列），三次预热后取平均：

| 请求 | 音频时长 | 端到端耗时 | RTF |
|---|---|---|---|
| 短句 | 1.70 s | 1.64 s | 0.97 |
| 长句 | 7.38 s | 2.49 s | 0.34 |

分组件：

| 组件 | 1 s 音频 | 7.4 s 音频 | 17.4 s 音频 |
|---|---|---|---|
| 声码器 | 244 ms | 2.21 s | 5.29 s |
| 装配器（ONNX） | 4.9 ms | 23 ms | 52 ms |

（RTF 是渲染耗时除以音频时长，小于 1 表示比实时快。）

声码器占大头，约 0.24–0.30 RTF，耗时随长度基本线性增长；装配器可以忽略。三秒以上的句子能实时渲染；一秒级的短句受固定开销影响，接近 1 但不够稳定。

## 模型

PointerGen，3.54M 参数，`c=416`，`n_ctx=3`。

输入：`segs [S,128,L]`、`seg_len`、`seg_idx`、`progress`、`boundary`、`f0`。
输出：`gather_seg(segs, tau) + tanh(residual)`，其中 `tau = progress + tanh(dtau) * 0.5`，模型可以在片段内移动读取位置。

训练数据用同一段录音自己的切片构造，按 40% 比例注入 oto 式缺陷（起点偏移、尾部截断、头部裁切、长辅音占窗），其余保持干净。目标是原录音的 mel。

训练脚本每 1000 步在固定留出集上评估并保存 best 权重。数据审计脚本检查训练/验证是否重叠、字段是否完整，红了一律不训。

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

OpenUtau 侧的接入代码单独维护。

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

导出脚本会检查动态 shape 下的数值一致性（阈值 1e-4），并把 ckpt/onnx 的 sha256、输入输出、mel 契约写进同名 json。当前实测 parity 1e-5 量级，CPU 上 1 秒音频的装配约 5 ms。

## 权重与许可

- 代码：Apache-2.0，见 `LICENSE`。
- 模型权重：CC BY-NC-SA 4.0。不随仓库分发，见 GitHub Releases。
- 数据集、声码器权重、声库：各自遵循原始许可，不随本仓库分发。

权重来自训练，跟数据集和声码器有衍生关系；再分发时沿用同样的非商用和署名条款。
