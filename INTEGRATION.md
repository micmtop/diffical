# 接入说明（diffical_neo 整句装配模型）

## 代码位置（单一代码源）
- 本仓库 `diffical_neo` 是唯一代码源：模型、训练、评测与 **OpenUtau 接入层**（`backend/`）。
- OpenUtau app 包仍在 `../diffical_new/mvp-openutau-osx-arm64/OpenUtau.app`（不重编译；只走 HTTP）。
- `diffical_new/backend` 是搬迁前的历史副本，不再修改。
- 模型定义唯一在 `model/pointer.py`；`backend/pointer_model.py` 只是转出，避免副本漂移。

## 模型文件
- `model/pointer.py` — PointerGen（c=416，n_ctx=3，约 3.54M 参数）
- `model/discriminator.py` — 判别器（仅训练用，推理不需要）
- 权重：`train/ckpt_gan/last.pt`（默认），或 `train/ckpt_all3/last.pt`（无对抗微调）

## 启动
```bash
cd ~/Documents/diffical/diffical_neo
bash run_openutau_model.sh          # 模型装配 + app
# 可选接缝安全过渡（默认关闭；2 帧 ≈ 把 1.5 的台阶摊到 5 帧）
DIFFICAL_SEAM_RAMP=2 bash run_openutau_model.sh
```
环境变量：`DIFFICAL_ENGINE=model`（脚本已设）、`DIFFICAL_POINTER_CKPT`（默认
`train/ckpt_gan/last.pt`）、`DIFFICAL_SEAM_RAMP`/`DIFFICAL_SEAM_RAMP_GATE`（默认 0 / 1.5）。

## 输入（推理时需要）
对每个**音素片段**：
- `segs[i]`  : (128, L_i) 的 log-mel 片段（hop 512 / 44.1k / 128 bins）
- `seg_len`  : 每段真实长度
- `seg_idx`  : 每帧属于第几段（1 基，0=无）
- `progress` : 帧在段内的归一化位置（0..1）
- `boundary` : 边界 ±4 帧为 1
- `f0`       : 每帧 F0（Hz，0=静音），**必须来自真实曲线或 IRender 曲线，不能是曲谱台阶**

## 输出
- 整句 log-mel (128, T)，直接送冻结的 pc-nsf-hifigan（44.1k hop512 128bin）出声

## 验证指标（200 句留出集）
- 边界误差相对画布 **+16.6%**（交叉淡化 +0.0%）
- 内容保真 **0.81**
- 高频占比/谱平坦度与真实录音差 0.001–0.005
- 听感（双片段盲评）：画布→模型 = 差→好、较好→好

## 注意
- 片段需按**生产装配**语义提供（拉伸到音符时长、硬切），不要喂交叉淡化后的结果
- 训练/推理条件必须一致（`progress` 定义、`boundary` 宽度、`f0` 是否清过 <55Hz）
- `backend/` 下 `renderer.py` 的 `DIFFICAL_ENGINE=model` 开关是唯一装配入口；
  `MelOverlapAccumulator` 只允许出现在 legacy 对照路径，不得进入模型装配管线。

## 运维注意
- app 退出**不会**杀 backend。换环境/换代码后重开前先确认 8573 的来源：
  `lsof -nP -iTCP:8573 -sTCP:LISTEN` 或 `ps -E -p <pid> | tr ' ' '\n' | grep DIFFICAL_BACKEND_PATH`。
  旧 backend 会被 `IsServerHealthy` 直接复用，导致看起来"还在用旧代码"。
- 冒烟：`cd diffical_neo && python -m backend.server`，几秒后 `curl 127.0.0.1:8573/` 应返回 200。
