# 模型验证（PointerGen，声码器未改动）

日期：2026-11-14  
模型：`model/pointer.py` PointerGen，0.364M 参数  
数据：OpenCpop，mismatch 片段（同一歌手、**不同歌曲**的同音素）作为输入  
训练：算力机 RTX 2060M，10k 步，batch 4  
留出：24 句（与训练句不同）

## 结论

模型在**三个独立指标**上一致优于"恒等摆放"（不使用模型、直接把片段按进度放到目标位置）：

| 指标 | 恒等摆放 | 模型 | 结果 |
|---|---|---|---|
| mel-l1（越低越好） | 1.991 | 1.881 | −5.5% |
| 音频域 MCD（越低越好，尺度非标准，仅供相对比较） | 130.72 | 108.55 | −17.0% |
| HNR 谐噪比 dB（越高越好） | 11.63 | 16.33 | +4.69 dB，**6/6 句更好** |

## 满数据复测（900 训练 / 100 留出，OpenCpop 全部可读句）

同一 100 句留出集，恒等摆放 2.1392、常数基线 2.2089：

| 参数量 | model l1 | 模型/恒等 | 相对 0.36M |
|---|---|---|---|
| 0.364M（c=192） | 1.6963 | **0.793** | — |
| **1.525M（c=416）** | **1.6284** | **0.761** | **−3.2pp** |
| 2.268M（c=512） | 1.6770 | 0.784 | +2.3pp（过拟合） |

对照 216 句数据（恒等 1.8082）：0.364M=0.917、1.525M=0.915。  
即：**数据 216→900 让误差比值从 0.917 降到 0.793**；数据到位后容量才开始起作用，1.5M 是当前最优点，2.27M 已回升。

MCD 分句：3 好 3 差，均值占优；HNR 分句 6/6 占优，是最稳定的证据。

## 复现

```bash
# 数据
python3 tools/build_setpairs.py --n 240 --out train/data/set_v2
python3 tools/build_mismatch.py --utts train/data/set_v2/train --out train/data/mm_v2 --split train
python3 tools/build_mismatch.py --utts train/data/set_v2/eval  --out train/data/mm_v2 --split eval

# 训练（算力机）
python3 -m train.train_set --arch pointer --pairs train/data/mm_v2 \
    --steps 10000 --batch 4 --device cuda --out ckpt_mm2

# 评估
python3 -m tools.eval_set  --ckpt ckpt_mm2/last.pt --data train/data/mm_v2 --split eval --n 24
python3 tools/audio_check.py     # MCD
python3 tools/hnr_check.py       # HNR
```

## 听感

`listen_test/20_感知验证_2010000353_real_identity_model.wav`（real | identity | model）  
`listen_test/21_声库推理_C5_a_i_u.wav`（真实声库 KAFU C5 的 a/i/u：identity | model）

## 已知边界

1. 目标录音的谱细节在**换歌片段**里不可恢复，所以 l1 有下限；模型学到的是"往自然的方向修正"。
2. 自源数据（片段=目标切片）上模型只能学恒等（1.077×），不能用于训练。
3. 没有做感知测试；上述三项是客观指标，不能替代耳朵。
