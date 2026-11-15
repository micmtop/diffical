# diffical_neo — 整句 mel 生成渲染器 SPEC v0.2

状态：**评审后重写，待确认**（v0.1 被三份独立评审各打出 5/10、5/10、6/10，全部 CHANGES）
日期：2026-11-14
评审记录见 §10。v0.1 的因果前提（"病在接缝"）被仓库自身测量 t44 反驳，v0.2 已改目标。

---

## 0. 目标（已按 t44 改写）

**主目标**：在 held-out 真实录音上，让模型产出的整句 mel 比现有拼接路径**更接近真实录音**（mel 距离 + 人工偏好），从而让整句粗糙度下降。

**不再把"边界过渡"作为主目标**，理由（仓库自身测量，`docs/ALGORITHM_RESEARCH.md` §94）：

- 边界逐帧 `|Δmel|` 中位 **0.156 < 内部 0.193**（p95 0.274 < 0.389）；
- 二阶差分 边界 ±6 帧 **0.243 < 内部 0.269**；
- **8 个频带组边界 Δ 全部低于内部 Δ**（0.09–0.17 vs 0.14–0.27）；
- §94.4 明写：t38→t53 共 8 个任务、约 16 轮"顺滑问题在接缝"的方向**已被否掉**；
- 接缝处真实存在的缺陷是**帧 44 的 f0 硬阶跃**（已由 t51 修），不是 mel 混合。

**推论**：V2 在边界处更可能是**过度平滑**（交叉淡入压平字头），不是"粗糙"。用户听感"overlap 不流畅"与此一致。

**前提门（P0.3，唯一否决权）**：在 12 个硬边界上算 `r = median(|Δmel|_boundary) / median(|Δmel|_interior)`，边界窗固定 **±50ms**（`boundary_window_ms=50`，与 §7.2 同源），另报 p95。三条臂：**(a) 真实连续录音**、**(b) 当前 V2 渲染（= 当前拼接路径，二者是同一实体，不再并列）**、**(c) hifisampler B1**。

预注册判据（`Δ = r_V2 − r_real`）：
- `|Δ| ≤ 0.10` → V2 ≈ 真实：边界假设作废，主目标改为整句粗糙度，**删除**边界加权损失与边界专项门禁；
- `Δ < −0.15` → 过度平滑：主目标改为"**恢复字头瞬态**"，指标方向反转；
- `Δ > +0.15` → 边界过粗糙：保留原方向；
- 灰区（`−0.15…−0.10` 或 `+0.10…+0.15`）→ 扩到 30 个边界再判。

**P0.3 判定（2026-11-14 闭合，详见 `docs/P0.3.md`）**：
- 真实录音：边界 Δmel **0.4409** / 内部 **0.2761** / 比 **1.597**（12 例，±50ms）
- V2 渲染：边界 **0.3719**（真实 ×0.844）/ 内部 **0.1551**（真实 ×**0.562**）/ 比 **2.399**（6 例真实请求）
- 合成装配：1.58–1.83；生产 mel 路径 7 个平滑函数全为死代码

**诊断**：接缝不是被抹平，而是**单元内部被压平**（内部动态只剩真实的 56%），接缝才显形。主目标据此定为：**内部 Δmel 恢复到 ≥0.85×真实、边界 Δmel 保持 0.9–1.1×真实**（比值 ≈1.6 是结果，不是目标）。t44 的 0.808 与本口径不可比（它量装配后的 mel，本诊断量过完声码器的完整波形）。

**非目标**：不做完整 SVS（不预测 F0/时长）；不训声码器；不做波形域生成；不做帧级音色曲线；不接管 offset/consonant/cutoff。

---

## 1. 决策表（全部已定）

| # | 决策 | 取值 |
|---|---|---|
| D1 | 架构 | 整句 mel 生成，**以逐音素源 mel 为参考**（非音色 ID） |
| D2 | 接缝 | **绝对生成 + 中性门控域（P0.3 后改）**：`_concatenate_mels` 用中性 envelope（Y=[0,1,1,1,0]）装配 → 模型输出**绝对整句 mel** → 用户 envelope 在模型之后施加 → 按 `compose()` 同式重算 `expression_gain`。理由见 §2.3 关键点 2 |
| D3 | mel 约定 | 44100 / hop 512 / 128 bins / fmin 40 / fmax 16000 / n_fft=win 2048 / **自然对数 DRC，clip 1e-9**，`center=False` |
| D4 | 规范前端 | `openvpi/SingingVocoders` `utils/wav2mel.py`（sha `350663f3`）+ `process.py:19-20,45`。已核实与 `diffical_new/backend/vocoder/wav2mel.py` 同式；`hifisampler/util/wav2mel.py` 的 `clip_val=1e-5` 是错的 |
| D5 | 音色 | 音符级 `clr`，协议用 `suffix` 代理；v1 不改 C# |
| D6 | 边界控制 | ~~overlapMs 作为用户可调过渡宽度~~ **v1 降级**：`overlapMs` 只用于素材装配与诊断，不承诺为模型条件（见 §2.3） |
| D7 | flag | 全部透传；模型不以 flag 为条件（`Hb/Hv/Ht` 在源 wav、`g` 在 mel 前、`HG/A/P` 在声码器后） |
| D8 | 数据许可 | NC/ND 进训练，权重发布接受风险（§9 R2 附最低合规动作） |
| D9 | 新仓 | `diffical_neo`（GitHub），旧仓保留 A/B |
| D10 | 算力 | Colab 免费 T4 16G + 本机 M1 |
| D11 | 验收 | 单人 2AFC 偏好（计数门禁）+ 自动指标；措辞固定"单人偏好" |
| D12 | 训练对 | **同歌手**片段（跨歌手配对教错任务，v0.1 已撤回） |

---

## 2. 系统接缝（v0.1 的接缝定义是错的）

### 2.1 现有链路
```
server.py:91 → renderer.py:1114 render
  :1175 _process_phones      每音素：声库 wav → hop128 mel → oto 裁窗 → np.interp 到 hop512
  :1181 _concatenate_mels    MelOverlapAccumulator 装配 + 门控 + 落帧 + expression_gain
  :1212 _synthesize_per_phone 整句一次 spec2wav
```

### 2.2 `_concatenate_mels` 实际做四件事（`:1953-1972`，v0.1 只写了第一件）
1. **相干性感知的幅度/功率交接**（`overlap.py:314-322`）：高相干→幅度相加；低相干→等功率交接；3dB 补偿上限。**不是** `exp(mel*2)*gain²`。
2. **OpenUtau envelope 门控**：`phone.envelope`（p0..p4）传给 `accumulator.add()`（`:1963-1968`）。
3. **分数帧起点积分**：`start_frame_exact`（`:1966`）。
4. **产出 `expression_gain`**（`:1971`）→ `:1436` `match_note_loudness(..., expression_gain=...)`。

### 2.3 D2：绝对生成 + 中性门控域（P0.3 后由"残差"改为"绝对生成"）

**关键点 1：envelope 不能既在输入里出现、又在目标里缺失**。若把用户 envelope 施加在模型输入上、目标却是未门控的真实录音，L1 会训练模型**抹掉用户的表情**——这比 v0.1 的"静默失效"更隐蔽。

**关键点 2（P0.3 新增）**：V2 的内部 Δmel 只有真实的 56%，模型必须整体把内部动态补回来（约 +58%）。这是**全局改写**，不是局部修补；残差头初始化偏向小修正，会系统性欠拟合这一项。因此 v1 改为**绝对生成**。

三段职责：
1. **中性装配**：**不能传 `envelope=None`**——`overlap.py:350-352` 会把该段标 `blocked` 并早退，跳过 `neutral/neutral_power/expression_power/unit_sum` 累加；`compose()` 的 `fade` 要求 `~blocked`，于是**相干性感知交接与结构淡变一起被关掉**，`expression_gain[blocked]=0` 还会污染 `match_note_loudness`。正确做法：用**几何相同、Y=[0,1,1,1,0] 的中性 envelope**（VOL=100 / ATK=100 / DEC=0）调用 `_concatenate_mels`，让 `structural` replay 正常跑、`expression = actual_weight / weight = 1`。
2. **模型**：`mel_out = model(source, conditions)`，输出绝对 mel；与 envelope 完全解耦，训练目标就是未门控的真实 mel。
3. **模型后施加 envelope**：对 `mel_out` 在幅度域施加 `g_user`（中性域 `g_neutral=1`），并按 `compose()` 同式用施加前后能量比重算 `expression_gain`（供 `match_note_loudness` 消费）。

这样 VOL/ATK/DEC、音符响度、分数帧全部保留，且模型不需要 envelope 通道。

**参数化说明**：绝对生成与残差属同一函数类，但绝对头不偏向"少改"，更匹配"整句重写内部动态"的需求。若 P0.5 显示绝对头难收敛，可退回残差——数据管线不变。

**`overlapMs` 的真实语义**（`UPhoneme.cs:175-212`）：`overlap` 可为负；**`!overlapped` 时** `GetFadeIn()=5ms`、`GetFadeOut()=35ms`，**overlapped 时**结构淡变宽度才是 `overlap` / `tailOverlap`。它不是稳定的"过渡宽度"控制量，v1 不作为模型条件。

---

## 3. 数据

### 3.1 OpenCpop 标注（已实测核验）
7 段 `|` 分隔：`name | lyrics | ph_seq | note_pitch | midi_dur | ph_dur | is_slur`
- **`ph_dur` 单位是秒**（不是 ms）；官方解析 `start_frame=int(t*sr/hop+0.5)`。
- **没有歌曲列**：`song_id = name // 10**6`（id 段 2001…2100 恰 100 组 = 100 首）。
- 官方解析器**不校验** `len(ph_seq)==len(ph_dur)`；neo 必须 `assert`。
- `SP`（静音）/`AP`（吸气）在音素列里且各带时长；音高列用 `rest` 占位。
- **SP/AP 政策**：target 直接自复制，不采 source，F0=0，只保留 phoneme id 条件；G0 报 SP/AP 帧占比。

### 3.2 语料
| 语料 | 时长 | 许可 | 获取 |
|---|---|---|---|
| OpenCpop | 5.23h | CC BY-NC-ND 4.0 | 已在本地（ZipCrypto，密码已核验） |
| M4Singer | ~20h | CC BY-NC-SA 4.0 | 官方 README 给**公开 Google Drive 直链、"for free use"**，**不需要申请** |
| GTSinger | 80.6h | CC BY-NC-SA 4.0 | 可选；T4 上不现实 |
| VocalSet | 10.1h | CC BY 4.0 | 仅消融对照 |

### 3.3 训练对构造（v0.2 修正）
对每个 target 句（歌手 V 的真实连续演唱）：
1. **target**：整句真实 mel（D4 前端，hop512）。
2. **source**：逐音素从 V 的**其他歌**取同音素片段 → **固定边距裁剪（首尾各 20–30ms）** → `np.interp` 到目标时长。随机裁 0–30% 只作额外增广（原写法定位对了，但不足以模拟孤立采样）。
3. **伪 oto 必须从真实分布采样**（2026-11-14 实测，脚本见 §12）：`~/Library/OpenUtau/Singers` 共 27 个目录，**其中 23 个含 oto.ini**；100 个 oto.ini / 64,823 行，解析成功 64,598，失败 225（空字段/短行）。
   | 参数 | 中位 | p05 – p95 | `=0` 占比 |
   |---|---|---|---|
   | offset | 1668.3 ms | 23.2 – 3930.0 | 1.6% |
   | consonant | 266.7 ms | 107.2 – 509.0 | 0.0% |
   | cutoff | −434.7 ms | −1177.0 – 188.0 | 0.3% |
   | preutter | 178.6 ms | 36.1 – 360.0 | 0.6% |
   | overlap | 83.3 ms | 0.0 – 123.8 | 4.5% |
   v0.1 的"offset=0 / overlap 15–60ms"会让训练只覆盖生产分布的单点与低尾；正确做法是按上表**逐库经验分布**采样。
4. **source 复用上限 ≤3**（同一片段被多个 target 复用会让模型记片段而非学过渡，L1 假性下降）；复用计数写入 `DATA_MANIFEST.json` 的 `source_reuse_count` 字段，便于审计。
5. **拉伸比分布**：生产里 0.5s 采样可拉到 5s（10×）；训练须记录并匹配该分布（直方图对齐）。
6. **域差增广**：真实音符边界处切窗并保留起始瞬态；叠加房间尾音/底噪；通道 EQ。

**禁止**：跨歌手配对（推理时不存在"目标音色"，教错任务）；同曲 source。

### 3.4 划分与测试集
- OpenCpop：`song_id` 90/5/5。
- M4Singer：留 2 位歌手整人做零样本；其余 18 人按曲 90/10。
- 边界测试集：四类硬边界各 3 例 = **12 例固定子集**（P0.3 首轮用），另从**同一哈希清单**的剩余池补 18 例（四类各 +4–5 例）= **30 例完整池**；P0.3 灰区扩样只准用这 30 例里**未参与首轮统计的 18 例**，禁止事后挑边界。
- 另加 **>30s 句 2 例**（块缝指标用）。

### 3.5 版本与存储
- 数据清单 `DATA_MANIFEST.json`（每条：路径、时长、sha256、来源、许可），清单整体 sha 写进 ckpt。
- 已知漂移：文档写 `refine_pairs` 1991 对，磁盘实为 **3550 对**（38 个 npz）。neo 不复用该数据，但所有文档数字须以磁盘为准。
- 本机 **16Gi 可用**；训练数据不进本机，只进 Colab。
- Colab 免费 Drive 仅 15GB：**语料 + mel 缓存 + 3 ckpt 必须 ≤15GB，否则先 OpenCpop-only**。P1 前实测体积。

---

## 4. 模型

### 4.1 输入（v0.2 修正）
| 通道 | 维度 | 说明 |
|---|---|---|
| source mel | 128 | 按目标帧网格（或装配后 mel，见 D2） |
| source mask | 1 | 有效帧 |
| **source 实测 F0 / semitone_delta** | 2 | **v0.1 缺失**：source 是"录音当时音高"，条件是"目标音高"，`np.interp` 不做频率轴归一，差值不受控 |
| F0 | 2 | log-F0 + voiced |
| phoneme emb | 32 | 中日英混合表 |
| 位置特征 | 6 | 音素内相对位置、距边界帧数、边界 mask、preutter、leading、dur |
| **envelope 5 点** | — | **不需要**：模型在"中性门控域"工作，用户 envelope 在模型之后施加（§2.3） |
| 伪子库 emb | **删除** | 训练期伪子库在推理时无对应物，属域增广，改信号域 EQ/房间即可 |

**无 speaker embedding**（音色由 source 自带）。

### 4.2 结构（按字面重算，v0.1 的 1.8M 是错的）
| 层 | 参数 |
|---|---|
| conv1 176→192 k5 | 169,152 |
| conv2 192→192 k5 | 184,512 |
| 6 × 残差块（每块 1 个 k3 conv + GN） | 664,704 |
| down 192→256 stride4 k3 | 147,712 |
| BiGRU ×2 hidden 192（输入 256） | **1,184,256** |
| fusion 384→192 k3 | 221,376 |
| out 192→128 k1 | 24,704 |
| **合计** | **≈2.60M** |

若残差块用 2 个 conv → **3.26M**；若 BiGRU 降为 1 层 hidden 128 → **≈1.71M**。
**v0.2 取 1 层 hidden 128 的 GRU，`count_params()` 断言落在 1.5–2.5M**；感受野 135 帧 ≈1.57s。

### 4.3 损失
```
L = L1(pred, target) + λ_d * L1(Δ_t pred, Δ_t target) + λ_b * boundary_weight ⊙ L1
```
- `boundary_weight` 窗口**与边界 mask 统一**为 `min(100ms, 0.4 × min(邻音素时长))`（v0.1 是 mask ±40ms、loss ±100ms，且短音节会被整单元覆盖）。
- 报告 `weight>1` 的帧占比（语料统计量），避免"边界加权=整体加权"。
- `λ_b=0` 消融 + "随机遮蔽音素内部中段监督"消融，用于分离边界与内部。

### 4.4 推理
- 推理按 **15s 块 + 1.5s 重叠**；**训练必须覆盖同等长度**（§5 的 bucketing 要混入长窗），否则全局瓶颈看到的上下文长度分布与推理不一致。
- 块缝**不用 log 域线性交叉**（会重现几何平均凹陷）；用重叠区内能量最低帧硬切，或复用 `overlap.py` 的一致性感知交接。
- 延迟预算只计模型前向；回退判据改为"模型耗时 > 总渲染时间 10%"。

---

## 5. 训练
- Colab T4 16G，单会话 ≤6h，ckpt 每 15min 存 Drive（最近 3 个 + 最优）。
- **按 8×256 帧 bucketing 为主，但每个 batch 至少混入 1 条 ≥1024 帧的长窗**（对应推理的 15s 块）；P1 的 200 步 smoke 必须测最长窗（≈1292 帧）的 s/step 与显存峰值。纯 256 帧训练会让双向 GRU 的上下文长度与推理错配。
- AdamW lr 2e-4，cosine + 1k warmup，grad clip 1.0，AMP。
- 步数：OpenCpop 30k；25h 混合 80k（2–3 次会话续训）。
- **P1 先跑 200 步计时 smoke**，报 s/step 与显存峰值，再决定整轮。
- ckpt 结构：`{state_dict, cfg_hash, git_sha, data_manifest_sha, versions}`；**cfg_hash 不匹配即拒载**。
- RNG：数据切分/增广/盲测选句各自独立种子，全部写进配置文件。

---

## 6. 集成
- C# 侧 v1 不改（复用 `DifficalRenderer`/`DifficalProtocol`）。
- Python 新服务实现同一 HTTP 契约，`?engine=neo|legacy`。
- **响应体必须带 `engine` 与 `model_sha`**；评测侧断言 `engine=neo` 且**未走回退**，否则该条作废。
- 回退必须显式错误码，不允许静默回退旧路径。

---

## 7. 评测

### 7.1 三基线
- **B1**：hifisampler 真实渲染
- **B2**：当前 diffical V2
- **B3（新增）**：`_concatenate_mels` + 固定余弦交叉淡入（宽度 = 语料 overlap 中位 **83ms**，实现落在同一帧网格，**无学习参数**）——**模型必须打赢 B3，才能声称"学到了"而非"抹平了"**

### 7.2 指标（v0.2 重写）
1. **动态三元组**（P0.3 后定）：分别报 **内部 Δmel**（目标 ≥0.85×真实，即 ≥0.235）、**边界 Δmel**（目标 0.9–1.1×真实，即 0.397–0.485）、**比值**（目标 ≈1.6，是结果不是目标）。三项都报绝对值——只报比值会被分母误导（t44 的 0.808 就是这么来的）。分频段（<1kHz / >4kHz）。
2. **F0 透传断言**：输入 F0 逐帧 bit-exact 出现在输出条件里；**不作质量指标**（F0 由 OU 提供，v0.1 把它当质量指标是错的）。
3. **瑕疵**：爆音复用 `eval/diagnostics/check_burst_regression.py` 的 **10ms RMS 相对元音、>2dB 即回归** 判据；静音洞；金属音用**参考语料分布分位**（如 >P95），不用绝对值 0.35（仓库自己把 flatness 0.2–0.4 判"优秀"）。
4. **内部保真（方向原先写反，已修正）**：分报 `‖pred − target‖`（音素内部、target 参考）与 `‖pred − warp(source)‖`（**copy-bias 指示量**，不是防线）。判读：两者都低=好；`‖pred−source‖` 低而 `‖pred−target‖` 高=在抄 source；另报 **source-swap 敏感度**。**阈值**：内部 `‖pred−target‖` ≤ B2 的 0.90×；copy-bias 比 `‖pred−source‖ / ‖pred−target‖` 不得 < 0.5（低于即判抄 source）；source-swap 后 `‖Δpred‖` ≥ 源片段差异的 10%。
5. **块缝**：>30s 句在缝 ±30ms 的能量突变与谱流。

### 7.3 对照组（防抄近路）
- source 置常数 / 置均值；
- source 跨音素打乱；
- 报告"内部保真 vs 边界改善"分项。

### 7.4 单人盲测（v0.2 修正）
- 范式改名 **2AFC 偏好**（不是 ABX；ABX 是 3 刺激）。
- 删掉 A/A catch（无正确答案）；改为**有正确答案的劣化锚**：`B2` vs `B2 + 带限噪声（1–4kHz，SNR 10dB）`，10 trial；**偏好干净版 ≥8/10 才算听者可信**，不达标则整轮作废重听，锚 trial 不计入偏好率。
- 句单固定 30/60 条，清单 sha、左右顺序种子、trial→对手映射种子全部落盘 `TRIALS.json` 并在听前预注册；响度归一到 −16 LUFS、文件名随机化。
- 结论只报**计数与精确 p**，措辞固定"单人偏好"，不得写"优于拼接器"。

---

## 8. 门禁与排期（全部改成可执行判据）

### P0 — 复现与前提诊断（1 个周末）
- **G0.0 固定输入清单**：`G0_CASES.json` 写死 USTX 路径 + sha、声库路径 + sha、ckpt sha、随机种子、每例期望输出 sha；结果前预注册，任何改动即整门作废。
- legacy 路径**直接 import 与旧仓字节相同的 `wav2mel.py`**（哈希断言）；判据 `max|Δmel| == 0`（不是 1e-4）。
- **G0.1 模型冒烟门**：无 NaN、帧数守恒、响应 `engine=neo` 且 `model_sha` 等于 `G0_CASES.json` 里的 ckpt sha、并断言 **neo 与 legacy 的整句 mel L1 > 0.01**（证明模型真的执行了，堵住"旧输出冒充 neo"）。
- **G0.2 表达式一致性门**（改可判形式）：
  - (a) **门控增益**：对每个音素、每个非零帧，neo 施加的线性增益 `g_user` 与 legacy 的 `actual/neutral` 比值逐帧相对差 ≤0.5%；再以音素 RMS 能量包络复核 ≤0.1 dB（避开增益为 0 帧的 `−inf`）。
  - (b) `expression_gain`：只在 `neutral_energy > −40 dB` 的帧上比，相对差 ≤1%；**仅报警，不作二元门**（neo 是模型后重算，与 legacy 从源装配得来的值天然不完全相等）。
  - (c) **退化用例**：`envelope=None`、`len(envelope)≠5`、非单调 xs、零长 mel 各 1 例；neo 必须与 legacy 一样走 `overlap.py:350-356` 的 `blocked` 分支（输出一致或同错误码）。
  - (d) overlap 非默认用例（crossfade 开/关、负 overlap、5/35ms）只验结构淡变与相干交接与 legacy 一致（逐帧功率 ≤0.05 dB），**不声称覆盖 `blocked`**。
- **P0.3 前提诊断**（见 §0）：12 个硬边界 × 三条臂（真实 / V2 / B1）。**这是全项目唯一否决门**。

### P0.5 — 单句过拟合（半天）
- 1 句 10s，训到 `mel L1 < 1e-2`（或比 baseline 降 10×），留 loss 曲线 + 同句推理三联证据。
- 不过 → 不进全量数据（数据构造 bug 与容量不足都会表现为"听不出来"，这一步用来分离）。

### P1 — 能听（3 个周末）
- 生成器 + 模型 + Colab 脚本 + 200 步 smoke + 30k 步。
- **G1（止损，主对手 B2）**：30 trial **全部对 B2**；`≥20/30`（**单侧 α=0.10**，精确 p=0.0494）；`≤16/30` 回炉；`17–19` 预注册为"扩到 60 trial、用同一句单、按 G2 判"，这一 look 记为探索性、不进结论。B1/B3 只作次要报告，**不参与 G1 计数**（跨基线合并计数不成立单一二项检验）。

### P2 — 指标与集成（3 个周末）
- 3 指标 + 2 消融（10k 步微调，不重训整轮）+ 边界 12 例主集 + 18 例扩展池 + 2 长句 + 60 trial + OU 集成。
- **G2（结论，主对手 B2）**：60 trial 对 B2，`≥37/60`（**单侧 α=0.05**，精确 p=0.0462）。**层级检验**：主指标（2AFC 偏好率）单独用 α=0.05、不被 Bonferroni 稀释；3 个次要自动指标用 **Holm-Bonferroni**（各与 B2 比），margin 预注册：边界/内部比 ±0.10、瑕疵率 ≤ B2 + 2dB 相对、内部保真 ≤ 0.90×。B1/B3 只作报告基线，不进门禁。

---

## 9. 风险登记（v0.2 重写）
| # | 风险 | 处置 |
|---|---|---|
| R1 | ~~声码器 −8/−17dB 缺口~~ **已证伪**：`ALGORITHM_RESEARCH.md:588` 实测往返 ±0.3 dB（1–4.6k +0.30、8–11k +0.11） | 删除该风险；不再据此设计 |
| R2 | NC/ND 进发布权重 | 已接受；最低动作：随权重附 NOTICE（两数据集 + 许可 URL + 论文引用）、两份许可全文、README 写明"权重仅研究/非商用"、若被认定 M4Singer 改编则权重按 **CC BY-NC-SA 4.0** 发布（与冻结声码器同许可，相容）、撤回开关与触发条件 |
| R3 | 单人盲听 | 措辞降级 + 计数门禁；不宣称"优于拼接器" |
| R4 | Colab 限流/断连 | 6h 会话 + 15min ckpt + resume + 200 步 smoke |
| R5 | 3 份 wav2mel 不一致 | 已定 D4；neo 只留一份，G0.1 断言两路径 mel 逐帧一致 |
| R6 | 旧仓 flag bug（`FLAGS_PATTERN` 字符类、`parse_flags` 零调用、`t/A/P/He/G` 未实现、C# 漏 `SHFC/BRE/NORM`） | 只在 neo 修；旧仓不动以保持 A/B 基线 |
| R7 | 本机 16Gi | 训练数据只进 Colab；P1 前实测缓存体积 |
| R8 | 数据获取 | M4Singer **无需申请**（公开 Drive）；瓶颈是 15GB Drive 配额 |
| **R9** | ~~前提风险~~ **已由 P0.3 闭合**：真实 1.597 / V2 2.399；机制是**内部动态只剩 56%**，不是接缝被抹平 | 主目标改为"内部动态恢复 + 边界不劣化"，指标改三元组（§7.2.1） |
| **R10（新增）** | 源音高 ≠ 目标音高，模型需隐式做频率轴重采 | 加 `semitone_delta` 通道；或先对 source 做 PSOLA 音高归一 |
| **R11（新增）** | 复制捷径：source 与 target 逐帧同源，L1 最优解就是抄 | B3 基线 + source 置常数/打乱对照 + 内部保真指标 |

---

## 10. 评审记录（2026-11-14）

| 陪审 | persona | 轮次 | 分数 | 结论 | blocker |
|---|---|---|---|---|---|
| jury-arch | 声学与架构正确性 | 1 | 5/10 | CHANGES | 3 |
| jury-data | 数据/许可/可行性 | 1 | 6/10 | CHANGES | 1 |
| jury-scope | 简化/范围/评测 | 1 | 5/10 | CHANGES | 3 |
| recheck-arch | 声学与架构 | 2 | 7/10 | CHANGES | B5 PARTIAL + 1 新 |
| recheck-data | 数据与可行性 | 2 | 7/10 | CHANGES | B3/B5 PARTIAL |
| recheck-eval | 评测效度 | 2 | 6/10 | CHANGES | 4 PARTIAL + 1 新 |
| final-arch | 正确性与可实现性 | 3 | — | **APPROVE**（10/10 CLOSED） | 0 |
| final-eval | 评测效度与可执行性 | 3 | — | CHANGES（8/10 CLOSED） | 2 新 |

**注**：三人同一模型（本会话唯一可用 provider），非跨厂商陪审；结论强度相应打折。

**2026-11-14 决策**：D2 定为 **(a) 残差 + 中性门控域**。理由：残差与绝对生成属同一函数类（不降上限），但收敛更容易且不必重算 `expression_gain`；用户目标为"高质量"，故取"上限相同但更易训成"的参数化。目标改写（§0）已确认。

**P0.3 判定（2026-11-14）**：真实 1.597 / 合成装配 1.58–1.83 / V2 2.399；绝对值拆解显示 V2 内部 Δmel 仅为真实的 56%。据此把 D2 从"残差"改为"绝对生成"，主目标改为"内部动态恢复 + 边界不劣化"。详见 `docs/P0.3.md`。

**定向复审（第 2 轮，3 人）**：recheck-arch 7/10、recheck-data 7/10、recheck-eval 6/10，均 CHANGES。已修：B5"内部保真"方向反了；中性装配传 `envelope=None` 会触发 `blocked` 早退；P0.3 三臂退化（(b)≡(c)）；G0 无固定输入清单；G1/G2 主对手与 α 不明；锚对无正确答案；句单/顺序/映射种子未预注册；oto 统计口径（23 个含 oto.ini 的声库，§3.3 + §12）。

**终审（第 3 轮，2 人，只验 10 项修复）**：final-arch **APPROVE（10/10 CLOSED，无新 blocker）**；final-eval **CHANGES**（8/10 CLOSED，新报 2 项：P0.3 灰区无预注册扩样池；G0.2 判据不可判——`−inf` 帧、`expression_gain` 无容差、overlap 用例不触发 `blocked`）。

**如实声明**：final-eval 的 2 项是在它评审**之后**修的（§3.4 改为 12 例主集 + 18 例扩展池、同一哈希清单；G0.2 改为线性增益 ≤0.5% + RMS ≤0.1dB、`expression_gain` 仅在 `neutral_energy > −40dB` 帧比且只报警、补 4 类真正触发 `blocked` 的退化用例、overlap 用例不再声称覆盖 `blocked`），**这两项未再经第三方复核**。按 `ds-jury` 三轮上限，此处停止自动循环；如需复核，由用户决定开第 4 轮。

采纳并已改进 v0.2 的条目：arch F1（envelope/expression_gain 回归）、F3（overlap 5/35ms）、F4（semitone_delta）、F5（块缝）、F6（G0 判据）、F7（电平影响 L1）、F8（窗口不统一）、F9（隔离增广弱）、F10（延迟预算）、F11（算法描述错误）、F12（参数/感受野）；data F1（7 段/秒/无歌曲列）、F2（SP/AP）、F3（M4Singer 无需申请）、F4（2.6M）、F6（功效低）、F7（A/A 无正确答案）、F8（oto 分布）、F9（片段复用/单 spk_id）、F10（许可动作）、F11（排期）、F12（bucketing）；scope F1（t44 反证）、F2（二项算术）、F3（门禁可作弊）、F4（参数账）、F5（无源 mel 张量）、F6（`_refine_mel` 双接缝）、F7（缺过拟合门）、F8（爆音判据）、F9（F0 指标无关/flatness 冲突）、F10（窗口/命名）、F11（契约缺失）、F12（R1 引用失效）。

**未采纳 1 条**：scope F6 要求"复用或取代 `_refine_mel`"。v0.2 不改动它（仍 `AI_REFINE_DISABLED`），但在 §2.3 声明本 SPEC 不占用该接缝；理由：并行保留两条 disabled 路径不影响 A/B，改动它反而引入回归面。

---

## 11. 待办
1. ~~D2 接缝~~ 已定：(a) 残差 + 中性门控域（§2.3）
2. ~~§0 目标改写~~ 已确认
3. P0：建仓 + import 同一份 wav2mel + legacy 逐帧一致 + G0.0/G0.1/G0.2
4. P0.3：前提诊断（12 边界 × 三条臂）——唯一否决门
5. P0.5：单句过拟合
6. P1：生成器 + 模型 + Colab smoke

---

## 12. 附录：oto 分布测量脚本（§3.3 的来源）

```python
import glob, os
root = os.path.expanduser('~/Library/OpenUtau/Singers')
paths = glob.glob(root + '/**/oto.ini', recursive=True)
vals = {k: [] for k in ['offset','consonant','cutoff','preutter','overlap']}
total = bad = 0
for p in paths:
    for line in open(p,'rb').read().decode('utf-8','ignore').splitlines():
        line = line.strip()
        if not line or line.startswith('#') or '=' not in line: continue
        total += 1
        parts = [x.strip() for x in line.split('=',1)[1].split(',')]
        if len(parts) < 6: bad += 1; continue
        try: nums = [float(parts[i]) for i in range(1,6)]
        except Exception: bad += 1; continue
        for k, v in zip(['offset','consonant','cutoff','preutter','overlap'], nums):
            vals[k].append(v)
for k, v in vals.items():
    s = sorted(v); n = len(s)
    print(k, 'median', s[n//2], 'p05', s[int(.05*n)], 'p95', s[int(.95*n)],
          'zero_frac', sum(1 for x in s if x == 0)/n)
print('files', len(paths), 'lines', total, 'parsed', len(vals['offset']), 'bad', bad)
```

实测输出：`files 100 / lines 64823 / parsed 64598 / bad 225`；offset 中位 1668.3、preutter 178.6、cutoff −434.7、consonant 266.7、overlap 83.3。任何环境差异须重跑此脚本并更新 §3.3 与 `DATA_MANIFEST.json`。
