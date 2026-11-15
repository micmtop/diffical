"""整句渲染协议数据结构

定义 C# DifficalRenderer 与 Python 后端之间的通信数据结构。
对应 OpenUtau 的 RenderPhrase / RenderNote / RenderPhone。
"""

from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple
import json


@dataclass
class RenderNote:
    """整句中的音符"""
    lyric: str
    tone: int              # MIDI 音数
    position: int          # 相对 phrase 起点的 tick
    duration: int          # tick
    end: int               # position + duration
    position_ms: float     # 绝对毫秒
    duration_ms: float
    end_ms: float


@dataclass
class RenderPhone:
    """整句中的音素（渲染的最小工作单位）"""
    # tick 位置
    position: int
    duration: int
    end: int
    leading: int           # 前导（consonant）长度

    # 毫秒位置
    position_ms: float
    duration_ms: float
    end_ms: float
    leading_ms: float

    phoneme: str           # 音素符号
    tone: int              # 所属音符的 MIDI 音数
    note_index: int        # 所属音符在 notes 中的索引
    tempo: float           # BPM

    # Classic 专属
    preutter_ms: float     # 预测 utter 时间
    overlap_ms: float      # 重叠时间
    suffix: str            # 音素后缀

    # 音素文件信息
    oto_file: str          # 声库中的 oto.ini 条目对应的音频文件路径
    oto_offset: float      # oto 中的 offset
    oto_consonant: float   # oto 中的 consonant
    oto_cutoff: float      # oto 中的 cutoff
    oto_preutter: float    # oto 中的 preutter
    oto_overlap: float     # oto 中的 overlap

    # 表达式参数
    volume: float = 1.0
    velocity: float = 1.0   # VEL/100（0-2，默认 1.0=100），OpenUTAU 传 p.velocity=VEL*0.01
    modulation: float = 0.0

    # 标志
    flags: List[Tuple[str, Optional[float], str]] = field(default_factory=list)
    direct: bool = False

    # UTAU 包络（p0-p4）：x=ms 相对音素起点，y=0-1 增益（OpenUTAU 提供，用于 mel 域拼接）
    envelope: List[Tuple[float, float]] = field(default_factory=list)

    # 音素特征（后端计算，跨语种映射用）
    phoneme_features: Optional[Any] = None
    
    # 跨语种映射语言信息
    cross_lang_source: Optional[str] = None
    cross_lang_target: Optional[str] = None


@dataclass
class RenderPhraseRequest:
    """整句渲染请求（C# → Python）"""
    # 整句位置
    position: int          # tick
    duration: int          # tick
    end: int               # tick
    leading: int           # tick

    position_ms: float
    duration_ms: float
    end_ms: float
    leading_ms: float

    # 音符和音素
    notes: List[RenderNote]
    phones: List[RenderPhone]

    # 整句曲线（5 tick 间隔采样）
    pitches: List[float]           # 音高（cent）
    dynamics: List[float]          # 力度（0-200）
    gender: List[float]            # 性别（0-100，50=默认）
    breathiness: List[float]       # 气声（0-100，50=默认）
    tone_shift: List[float]        # 音高偏移（-1200~1200 cent）
    tension: List[float]           # 张力（0-100，50=默认）
    voicing: List[float]           # 浊音度（0-100，100=默认）
    growl: List[float]             # 咆哮强度（0-100，0=无）
    lowcut: List[float]            # 动态低切（0-100，0=无）
    brel: List[float]              # 低频气声（0-100，0=无）
    breh: List[float]              # 高频气声（0-100，0=无）
    warm: List[float]              # 温暖度（0-100，0=无）
    hcmp: List[float]              # 谐波压缩（0-100，0=无）

    # 自定义曲线
    curves: List[Tuple[str, List[float]]] = field(default_factory=list)

    # 声库信息
    singer_path: str = ""           # 声库根目录
    singer_type: str = "utau"       # 声库类型
    singer_language: str = ""       # 声库语言（如 "ja", "zh", "en"）

    # 渲染选项
    language: str = ""              # 目标语言（用于跨语种映射）

    # 缓存
    hash: int = 0                   # phrase hash（用于缓存）

    # 渲染选项
    is_pre_render: bool = False
    track_no: int = 0

    def to_json(self) -> str:
        """序列化为 JSON"""
        return json.dumps(self, default=lambda o: o.__dict__, ensure_ascii=False)

    @classmethod
    def from_json(cls, json_str: str) -> "RenderPhraseRequest":
        """从 JSON 反序列化"""
        data = json.loads(json_str)
        notes = [RenderNote(**n) for n in data.pop("notes", [])]
        phones = [RenderPhone(**p) for p in data.pop("phones", [])]
        curves = [tuple(c) for c in data.pop("curves", [])]
        for p in phones:
            if "flags" in p.__dict__:
                p.flags = [tuple(f) for f in p.flags]
        for param in ['pitches', 'dynamics', 'gender', 'breathiness', 
                      'tone_shift', 'tension', 'voicing', 'growl', 
                      'lowcut', 'brel', 'breh', 'warm', 'hcmp']:
            if param not in data:
                data[param] = []
        return cls(notes=notes, phones=phones, curves=curves, **data)


@dataclass
class RenderResultResponse:
    """整句渲染结果（Python → C#）"""
    samples: List[float]       # 单声道 PCM（44100Hz）
    leading_ms: float          # 前导长度
    position_ms: float         # 非前导部分起始位置
    estimated_length_ms: float # 预估总长度

    # 渲染后的实际音高曲线（可选，SupportsRenderPitch=true 时）
    pitch_ticks: Optional[List[float]] = None
    pitch_tones: Optional[List[float]] = None

    # 渲染后的实际表达式曲线（可选，SupportsRealCurve=true 时）
    real_curves: Optional[List[Tuple[str, List[float], List[float]]]] = None

    def to_json(self) -> str:
        """序列化为 JSON"""
        return json.dumps({
            "samples": self.samples,
            "leading_ms": self.leading_ms,
            "position_ms": self.position_ms,
            "estimated_length_ms": self.estimated_length_ms,
            "pitch_ticks": self.pitch_ticks,
            "pitch_tones": self.pitch_tones,
            "real_curves": self.real_curves,
        }, ensure_ascii=False)
