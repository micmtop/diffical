# Diffical

给 [OpenUtau](https://github.com/stakira/OpenUtau) 用的实验性 AI 歌声渲染器：
一个小模型把每个音素的 mel 片段装配成整句连续 log-mel，再交给冻结的
PC-NSF-HiFiGAN 声码器出声（44.1 kHz / hop 512 / 128 bins）。

**装配由模型完成**：没有波形交叉淡化，推理和训练输入里都没有
`MelOverlapAccumulator`。片段边界是模型的条件输入，过渡由模型自己生成。

```
OpenUtau (C#) --HTTP 8573--> Python 后端
                               音素切片 / oto 拉伸（hop-128 源时钟）
                               -> PointerGen mel 装配器
                               -> 冻结的 pc-nsf-hifigan
                               -> 44.1 kHz 波形返回 OpenUtau
```

## 当前状态

v0.1，研究阶段。已经能在 OpenUtau 里完整跑通，但仍有可听瑕疵，见「已知局限」。

已实现：

- **PointerGen 装配器**（`model/pointer.py`）：3.54M 参数，`c=416`，`n_ctx=3`。
  输入 `segs [S,128,L]`、`seg_len`、`seg_idx`、`progress`、`boundary`、`f0`；
  `out = gather_seg(segs, tau) + tanh(residual)`，其中 `tau = progress + tanh(dtau) * 0.5`，
  模型可以在片段内移动读取位置。
- **生产式几何（seg-norm）**：推理时每段只保留 `seg_idx` 实际覆盖的帧，`progress` 走满 0..1，
  与训练对的构造方式一致。
- **oto 缺陷混合训练**：40% 的训练句注入 oto 式缺陷（源窗偏移 ±6 帧、尾部截短 10–25%、
  头部裁 1–3 帧、长辅音占窗），60% 保持干净。目的是让模型学会抹平坏 oto 造成的接缝，
  同时不拖累 oto 标得好的声库。
- **不做输出「化妆」**：没有每音符响度匹配、没有在渲染波形上做包络增益、没有 EQ/压缩；
  OpenUtau 的 VOL/envelope 不做后置增益，整句只做一个常数 RMS 归一化。

## 仓库结构

| 路径 | 内容 |
|---|---|
| `backend/` | HTTP 服务、渲染器、音素处理、声码器封装、OpenUtau 桥接 |
| `model/` | PointerGen（`pointer.py`）与早期基线（`net.py`、`setgen.py`） |
| `tools/` | 数据集构造（`build_*_pairs.py`）、渲染/评估/诊断脚本 |
| `train/` | 训练循环（`train_set.py`）、数据加载、损失 |
| `docs/` | 配置模板 |

**不在仓库里**：数据集、模型权重、渲染音频、修改版 OpenUtau App。
训练用的数据集不可再分发，权重发布前需要单独确认数据集许可。

## 快速开始（只跑后端）

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp docs/config.example.yaml docs/config.yaml
# 编辑 docs/config.yaml：声码器/hnsep 路径、device

python -m backend.server            # 监听 http://127.0.0.1:8573
```

要在 OpenUtau 里使用，需要一个实现了 `IRenderer`、把渲染请求 POST 到本后端的
OpenUtau 构建；这部分 C# 代码单独维护，不在本仓库。

启动器/OpenUtau 侧不在本仓库，请根据你使用的构建自行配置后端地址。

## 训练

构造器需要本地的 OpenCpop / GTSinger 副本（不随仓库分发）：

```bash
# 生产式几何 + 可选 oto 缺陷
python tools/build_oto_pairs.py --utts <npz_dir> --out train/data/oto_full --shard 0 --shards 2
python tools/build_oto_gtsinger.py --root data/gtsinger/Chinese --out train/data/oto_full --shard 0 --shards 2

# RTX 2060 上 25k 步约 20 分钟（~22 it/s）
python -m train.train_set --arch pointer --width 416 --pairs train/data/oto_full \
  --steps 25000 --batch 4 --device cuda --amp --init ckpt_gan/last.pt \
  --lambda-b 20 --lambda-seam 1 --out ckpt_oto_full
```

训练数据必须遵守的规则：

- 片段只能取自**目标录音本身**（内容可达），不能用别的 take 的同音素当目标。
- F0 从音频提取（pyworld harvest），不能用曲谱 MIDI 台阶。
- `target` 永远是原录音的 mel。

## 已知局限

- **接缝残余**：遇到硬 oto/电平不连续时装配器仍会留下台阶（实测模型输出接缝跳变约 1.5，
  真实录音约 0.56）。
- **声码器起止冲击**：句首/句尾有短促低频冲击，这是声码器在片段边界的固有行为
  （旧拼接引擎同样存在），mel 模型无法消除。
- **长音发直**：部分声库长音拉伸后源内容本身就静态，听起来会「电」。
- **oto 补偿是学出来的**：坏 oto 已经切掉的内容补不回来，只能把剩余内容放顺。

## 许可

代码使用 **Apache-2.0**（见 `LICENSE`）。数据集、声码器权重和声库各自遵循其原始许可，
不随本仓库分发。模型权重暂不发布。

## 参考/致谢

- [OpenUtau](https://github.com/stakira/OpenUtau)
- pc-nsf-hifigan（声码器）
- HiFiUTAU-engine（VOL/envelope 的 mel 域处理参考）
