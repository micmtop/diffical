"""Diffical 音素器配置

定义中英日韩语言的音素映射规则，使输出音素符合引擎需求。

音素映射策略：
1. 中文：将拼音映射到日文假名
2. 英文：将 ARPABET 音素映射到日文假名
3. 日文：直接使用假名
4. 韩文：将韩文音素映射到日文假名

配置格式：
- PHONEMIZER_CONFIG: 各语言音素器配置
- LANG_PHONEME_MAP: 语言到音素的映射规则
- VOWEL_MAPPING: 元音映射表
- CONSONANT_MAPPING: 辅音映射表
"""

PHONEMIZER_CONFIG = {
    "ZH": {
        "name": "Chinese Phonemizer",
        "tag": "ZH DIFFICAL",
        "g2p_type": "pinyin",
        "primary_vowel_map": {
            "a": "あ", "o": "お", "e": "え", "i": "い", "u": "う",
            "ai": "あい", "ei": "えい", "ao": "あお", "ou": "おう",
            "an": ["あ", "ん"], "en": ["え", "ん"], "in": ["い", "ん"], "un": ["う", "ん"],
            "ang": ["あ", "ん"], "eng": ["え", "ん"], "ing": ["い", "ん"], "ong": ["お", "ん"],
            "ia": "や", "ie": "いぇ", "io": "よ", "iu": "ゆう",
            "ua": "わ", "uo": "を", "ui": "うい", "uai": "わい",
            "ue": "うぇ", "uan": "わん", "uang": "わん", "ueng": "うえん",
            "üe": "ゆぇ", "üan": "ゆあん", "ün": "ゆん",
        },
        "consonant_map": {
            "b": "ば", "p": "ぱ", "m": "ま", "f": "ふ",
            "d": "だ", "t": "た", "n": "な", "l": "ら",
            "g": "が", "k": "か", "h": "は",
            "j": "じ", "q": "ち", "x": "し",
            "zh": "じ", "ch": "ち", "sh": "し", "r": "ら",
            "z": "ざ", "c": "つ", "s": "さ",
            "y": "や", "w": "わ",
            "": "",
        },
        "tone_map": {
            1: "",
            2: "",
            3: "",
            4: "",
            5: "",
        },
        "full_pinyin_map": {
            "ba": "ば", "pa": "ぱ", "ma": "ま", "fa": "ふぁ",
            "bi": "び", "pi": "ぴ", "mi": "み", "fi": "ふぃ",
            "bu": "ぶ", "pu": "ぷ", "mu": "む", "fu": "ふ",
            "bo": "ぼ", "po": "ぽ", "mo": "も", "fo": "ふぉ",
            "bei": "べい", "pei": "ぺい", "mei": "めい", "fei": "ふぇい",
            "bai": "ばい", "pai": "ぱい", "mai": "まい",
            "bao": "ばお", "pao": "ぱお", "mao": "まお", "fao": "ふぁお",
            "ban": "ばん", "pan": "ぱん", "man": "まん", "fan": "ふぁん",
            "bang": "ばん", "pang": "ぱん", "mang": "まん", "fang": "ふぁん",
            "ben": "べん", "pen": "ぺん", "men": "めん", "fen": "ふぇん",
            "beng": "べん", "peng": "ぺん", "meng": "めん", "feng": "ふぇん",
            "da": "だ", "ta": "た", "na": "な", "la": "ら",
            "di": "でぃ", "ti": "てぃ", "ni": "に", "li": "り",
            "du": "づ", "tu": "つ", "nu": "ぬ", "lu": "る",
            "do": "ど", "to": "と", "no": "の", "lo": "ろ",
            "de": "で", "te": "て", "ne": "ね", "le": "れ",
            "dai": "だい", "tai": "たい", "nai": "ない", "lai": "らい",
            "dou": "どう", "tou": "とう", "nou": "のう", "lou": "ろう",
            "dan": "だん", "tan": "たん", "nan": "なん", "lan": "らん",
            "dang": "だん", "tang": "たん", "nang": "なん", "lang": "らん",
            "den": "でん", "ten": "てん", "nen": "ねん", "len": "れん",
            "deng": "でん", "teng": "てん", "neng": "ねん", "leng": "れん",
            "ding": "でん", "ting": "てん", "ning": "にん", "ling": "りん",
            "dong": "どん", "tong": "とん", "nong": "のん", "long": "ろん",
            "ga": "が", "ka": "か", "ha": "は",
            "ge": "げ", "ke": "け", "he": "へ",
            "gu": "ぐ", "ku": "く", "hu": "ふ",
            "go": "ご", "ko": "こ", "ho": "ほ",
            "gui": "ぐい", "kui": "くい", "hui": "ふい",
            "guai": "ぐわい", "kuai": "くわい", "huai": "ふわい",
            "gou": "ごう", "kou": "こう", "hou": "ほう",
            "gan": "がん", "kan": "かん", "han": "はん",
            "gang": "がん", "kang": "かん", "hang": "はん",
            "gen": "げん", "ken": "けん", "hen": "へん",
            "geng": "げん", "keng": "けん", "heng": "へん",
            "gong": "ごん", "kong": "こん", "hong": "ほん",
            "jia": "じゃ", "qia": "ちゃ", "xia": "しゃ",
            "ji": "じ", "qi": "ち", "xi": "し",
            "ju": "じゅ", "qu": "ちゅ", "xu": "しゅ",
            "jie": "じぇ", "qie": "ちぇ", "xie": "しぇ",
            "jiao": "じゃお", "qiao": "ちゃお", "xiao": "しゃお",
            "jiu": "じゅう", "qiu": "ちゅう", "xiu": "しゅう",
            "jian": "じゃん", "qian": "ちゃん", "xian": "しゃん",
            "jiang": "じゃん", "qiang": "ちゃん", "xiang": "しゃん",
            "jin": "じん", "qin": "ちん", "xin": "しん",
            "jing": "じん", "qing": "ちん", "xing": "しん",
            "jiong": "じょん", "qiong": "ちょん", "xiong": "しょん",
            "zhu": "じゅ", "chu": "ちゅ", "shu": "しゅ", "ru": "る",
            "zhi": "じ", "chi": "ち", "shi": "し", "ri": "り",
            "zhe": "じぇ", "che": "ちぇ", "she": "しぇ", "re": "れ",
            "zhuo": "じょ", "chuo": "ちょ", "shuo": "しょ", "ruo": "ろ",
            "zhai": "じぁい", "chai": "ちぁい", "shai": "しぁい",
            "zhou": "じゅう", "chou": "ちゅう", "shou": "しゅう", "rou": "ろう",
            "sha": "しゃ", "cha": "ちゃ", "ra": "ら",
            "she": "しぇ", "che": "ちぇ", "re": "れ",
            "sho": "しょ", "cho": "ちょ", "ro": "ろ",
            "zhan": "じゃん", "chan": "ちゃん", "shan": "しゃん", "ran": "らん",
            "zhang": "じゃん", "chang": "ちゃん", "shang": "しゃん", "rang": "らん",
            "zhen": "じん", "chen": "ちん", "shen": "しん", "ren": "れん",
            "zheng": "じん", "cheng": "ちん", "sheng": "しん", "reng": "れん",
            "zhong": "じょん", "chong": "ちょん", "shong": "しょん", "rong": "ろん",
            "zu": "ず", "cu": "つ", "su": "す",
            "zi": "づ", "ci": "つ", "si": "す",
            "ze": "ぜ", "ce": "つぇ", "se": "せ",
            "zuo": "ざお", "cuo": "つぉ", "suo": "そ",
            "zai": "ざい", "cai": "つぁい", "sai": "さい",
            "zou": "ざお", "cou": "つぉ", "sou": "そう",
            "zan": "ざん", "can": "つぁん", "san": "さん",
            "zang": "ざん", "cang": "つぁん", "sang": "さん",
            "zen": "ぜん", "cen": "つぇん", "sen": "せん",
            "zeng": "ぜん", "ceng": "つぇん", "seng": "せん",
            "zong": "ざん", "cong": "つぉん", "song": "そん",
            "ya": "や", "wa": "わ",
            "yi": "い", "wu": "う",
            "yu": "ゆ",
            "ye": "いぇ", "wo": "を",
            "yao": "やお", "wo": "を",
            "you": "ゆう", "wen": "うえん",
            "yan": "やん", "wan": "わん",
            "yang": "やん", "wang": "わん",
            "yin": "いん", "yun": "ゆん",
            "ying": "いん",
            "yong": "よん",
        },
    },
    "EN": {
        "name": "English Phonemizer",
        "tag": "EN DIFFICAL",
        "g2p_type": "arpabet",
        "primary_vowel_map": {
            "AA": "あ", "AE": "あ", "AH": "あ", "AO": "お", "AW": "あお",
            "AY": "あい", "EH": "え", "ER": "え", "EY": "えい", "IH": "い",
            "IY": "い", "OW": "お", "OY": "おい", "UH": "う", "UW": "う",
            "AA1": "あ", "AA2": "あ", "AA0": "あ",
            "AE1": "あ", "AE2": "あ", "AE0": "あ",
            "AH1": "あ", "AH2": "あ", "AH0": "あ",
            "AO1": "お", "AO2": "お", "AO0": "お",
            "AW1": "あお", "AW2": "あお", "AW0": "あお",
            "AY1": "あい", "AY2": "あい", "AY0": "あい",
            "EH1": "え", "EH2": "え", "EH0": "え",
            "ER1": "え", "ER2": "え", "ER0": "え",
            "EY1": "えい", "EY2": "えい", "EY0": "えい",
            "IH1": "い", "IH2": "い", "IH0": "い",
            "IY1": "い", "IY2": "い", "IY0": "い",
            "OW1": "お", "OW2": "お", "OW0": "お",
            "OY1": "おい", "OY2": "おい", "OY0": "おい",
            "UH1": "う", "UH2": "う", "UH0": "う",
            "UW1": "う", "UW2": "う", "UW0": "う",
        },
        "consonant_map": {
            "B": "ば", "CH": "ち", "D": "だ", "DH": "だ", "F": "ふ",
            "G": "が", "HH": "ひ", "JH": "じ", "K": "か", "L": "ら",
            "M": "ま", "N": "な", "NG": "ん", "P": "ぱ", "R": "ら",
            "S": "さ", "SH": "し", "T": "た", "TH": "た", "V": "ば",
            "W": "わ", "Y": "や", "Z": "ざ", "ZH": "じ",
        },
    },
    "JA": {
        "name": "Japanese Phonemizer",
        "tag": "JA DIFFICAL",
        "g2p_type": "kana",
        "primary_vowel_map": {
            "あ": "あ", "い": "い", "う": "う", "え": "え", "お": "お",
            "ぁ": "あ", "ぃ": "い", "ぅ": "う", "ぇ": "え", "ぉ": "お",
        },
        "consonant_map": {},
        "yoon_map": {
            "きゃ": ["き", "ゃ"], "きゅ": ["き", "ゅ"], "きょ": ["き", "ょ"],
            "しゃ": ["し", "ゃ"], "しゅ": ["し", "ゅ"], "しょ": ["し", "ょ"],
            "ちゃ": ["ち", "ゃ"], "ちゅ": ["ち", "ゅ"], "ちょ": ["ち", "ょ"],
            "にゃ": ["に", "ゃ"], "にゅ": ["に", "ゅ"], "にょ": ["に", "ょ"],
            "ひゃ": ["ひ", "ゃ"], "ひゅ": ["ひ", "ゅ"], "ひょ": ["ひ", "ょ"],
            "みゃ": ["み", "ゃ"], "みゅ": ["み", "ゅ"], "みょ": ["み", "ょ"],
            "りゃ": ["り", "ゃ"], "りゅ": ["り", "ゅ"], "りょ": ["り", "ょ"],
            "ぎゃ": ["ぎ", "ゃ"], "ぎゅ": ["ぎ", "ゅ"], "ぎょ": ["ぎ", "ょ"],
            "じゃ": ["じ", "ゃ"], "じゅ": ["じ", "ゅ"], "じょ": ["じ", "ょ"],
        },
    },
    "KO": {
        "name": "Korean Phonemizer",
        "tag": "KO DIFFICAL",
        "g2p_type": "hangul",
        "primary_vowel_map": {
            "ㅏ": "あ", "ㅐ": "え", "ㅑ": "や", "ㅒ": "いぇ",
            "ㅓ": "え", "ㅔ": "え", "ㅕ": "ゆ", "ㅖ": "ゆぇ",
            "ㅗ": "お", "ㅘ": "わ", "ㅙ": "わえ", "ㅚ": "おい",
            "ㅛ": "よ", "ㅜ": "う", "ㅝ": "うえ", "ㅞ": "うえ",
            "ㅟ": "うい", "ㅠ": "ゆ", "ㅡ": "う", "ㅢ": "うい",
            "ㅣ": "い",
        },
        "consonant_map": {
            "ㄱ": "か", "ㄲ": "か", "ㄴ": "な", "ㄷ": "た", "ㄸ": "た",
            "ㄹ": "ら", "ㅁ": "ま", "ㅂ": "ば", "ㅃ": "ば", "ㅅ": "さ",
            "ㅆ": "さ", "ㅇ": "", "ㅈ": "じ", "ㅉ": "じ", "ㅊ": "ち",
            "ㅋ": "か", "ㅌ": "た", "ㅍ": "ぱ", "ㅎ": "は",
            "ㄳ": "かさ", "ㄵ": "なじ", "ㄶ": "なは", "ㄺ": "らか",
            "ㄻ": "らま", "ㄼ": "らば", "ㄽ": "らさ", "ㄾ": "らた",
            "ㄿ": "らぱ", "ㅀ": "らは", "ㅄ": "ばさ",
        },
    },
}

VOWEL_MAPPING = {
    "a": "あ", "o": "お", "e": "え", "i": "い", "u": "う",
    "ai": "あい", "ei": "えい", "ao": "あお", "ou": "おう",
    "an": ["あ", "ん"], "en": ["え", "ん"], "in": ["い", "ん"], "un": ["う", "ん"],
    "ang": ["あ", "ん"], "eng": ["え", "ん"], "ing": ["い", "ん"], "ong": ["お", "ん"],
    "AA": "あ", "AE": "あ", "AH": "あ", "AO": "お", "AW": "あお",
    "AY": "あい", "EH": "え", "ER": "え", "EY": "えい", "IH": "い",
    "IY": "い", "OW": "お", "OY": "おい", "UH": "う", "UW": "う",
    "あ": "あ", "い": "い", "う": "う", "え": "え", "お": "お",
    "ㅏ": "あ", "ㅐ": "え", "ㅑ": "や", "ㅒ": "いぇ",
    "ㅓ": "え", "ㅔ": "え", "ㅕ": "ゆ", "ㅖ": "ゆぇ",
    "ㅗ": "お", "ㅛ": "よ", "ㅜ": "う", "ㅠ": "ゆ",
    "ㅡ": "う", "ㅣ": "い",
}

CONSONANT_MAPPING = {
    "b": "ば", "p": "ぱ", "m": "ま", "f": "ふ",
    "d": "だ", "t": "た", "n": "な", "l": "ら",
    "g": "が", "k": "か", "h": "は",
    "j": "じ", "q": "ち", "x": "し",
    "zh": "じ", "ch": "ち", "sh": "し", "r": "ら",
    "z": "ざ", "c": "つ", "s": "さ",
    "y": "や", "w": "わ",
    "B": "ば", "CH": "ち", "D": "だ", "DH": "だ", "F": "ふ",
    "G": "が", "HH": "ひ", "JH": "じ", "K": "か", "L": "ら",
    "M": "ま", "N": "な", "NG": "ん", "P": "ぱ", "R": "ら",
    "S": "さ", "SH": "し", "T": "た", "TH": "た", "V": "ば",
    "W": "わ", "Y": "や", "Z": "ざ", "ZH": "じ",
    "か": "か", "き": "き", "く": "く", "け": "け", "こ": "こ",
    "が": "が", "ぎ": "ぎ", "ぐ": "ぐ", "げ": "げ", "ご": "ご",
    "さ": "さ", "し": "し", "す": "す", "せ": "せ", "そ": "そ",
    "ざ": "ざ", "じ": "じ", "ず": "ず", "ぜ": "ぜ", "ぞ": "ぞ",
    "た": "た", "ち": "ち", "つ": "つ", "て": "て", "と": "と",
    "だ": "だ", "ぢ": "ぢ", "づ": "づ", "で": "で", "ど": "ど",
    "は": "は", "ひ": "ひ", "ふ": "ふ", "へ": "へ", "ほ": "ほ",
    "ば": "ば", "び": "び", "ぶ": "ぶ", "べ": "べ", "ぼ": "ぼ",
    "ぱ": "ぱ", "ぴ": "ぴ", "ぷ": "ぷ", "ぺ": "ぺ", "ぽ": "ぽ",
    "ま": "ま", "み": "み", "む": "む", "め": "め", "も": "も",
    "な": "な", "に": "に", "ぬ": "ぬ", "ね": "ね", "の": "の",
    "ら": "ら", "り": "り", "る": "る", "れ": "れ", "ろ": "ろ",
    "や": "や", "ゆ": "ゆ", "よ": "よ",
    "わ": "わ", "を": "を", "ん": "ん",
}

DIFFICAL_PHONEME_SETS = {
    "ZH": {
        "vowels": ["a", "o", "e", "i", "u", "ü",
                    "ai", "ei", "ao", "ou",
                    "an", "en", "in", "un", "ün",
                    "ang", "eng", "ing", "ong",
                    "ia", "ie", "iao", "iu",
                    "ua", "uo", "uai", "ui",
                    "üe", "üan", "iong"],
        "consonants": ["b", "p", "m", "f",
                       "d", "t", "n", "l",
                       "g", "k", "h",
                       "j", "q", "x",
                       "zh", "ch", "sh", "r",
                       "z", "c", "s",
                       "y", "w"],
        "tone_marks": ["", "ˊ", "ˇ", "ˋ", "˙"],
    },
    "EN": {
        "vowels": ["AA", "AE", "AH", "AO", "AW", "AY",
                   "EH", "ER", "EY", "IH", "IY",
                   "OW", "OY", "UH", "UW"],
        "consonants": ["B", "CH", "D", "DH", "F", "G", "HH", "JH",
                       "K", "L", "M", "N", "NG", "P", "R",
                       "S", "SH", "T", "TH", "V", "W", "Y", "Z", "ZH"],
        "stress_marks": ["0", "1", "2"],
    },
    "JA": {
        "vowels": ["あ", "い", "う", "え", "お",
                   "ぁ", "ぃ", "ぅ", "ぇ", "ぉ"],
        "consonants": ["か", "き", "く", "け", "こ",
                       "が", "ぎ", "ぐ", "げ", "ご",
                       "さ", "し", "す", "せ", "そ",
                       "ざ", "じ", "ず", "ぜ", "ぞ",
                       "た", "ち", "つ", "て", "と",
                       "だ", "ぢ", "づ", "で", "ど",
                       "は", "ひ", "ふ", "へ", "ほ",
                       "ば", "び", "ぶ", "べ", "ぼ",
                       "ぱ", "ぴ", "ぷ", "ぺ", "ぽ",
                       "ま", "み", "む", "め", "も",
                       "な", "に", "ぬ", "ね", "の",
                       "ら", "り", "る", "れ", "ろ",
                       "や", "ゆ", "よ",
                       "わ", "を", "ん"],
        "yoon": ["きゃ", "きゅ", "きょ",
                 "しゃ", "しゅ", "しょ",
                 "ちゃ", "ちゅ", "ちょ",
                 "にゃ", "にゅ", "にょ",
                 "ひゃ", "ひゅ", "ひょ",
                 "みゃ", "みゅ", "みょ",
                 "りゃ", "りゅ", "りょ",
                 "ぎゃ", "ぎゅ", "ぎょ",
                 "じゃ", "じゅ", "じょ"],
    },
    "KO": {
        "vowels": ["ㅏ", "ㅐ", "ㅑ", "ㅒ", "ㅓ", "ㅔ", "ㅕ", "ㅖ",
                   "ㅗ", "ㅘ", "ㅙ", "ㅚ", "ㅛ", "ㅜ", "ㅝ", "ㅞ",
                   "ㅟ", "ㅠ", "ㅡ", "ㅢ", "ㅣ"],
        "consonants": ["ㄱ", "ㄲ", "ㄳ", "ㄴ", "ㄵ", "ㄶ", "ㄷ", "ㄸ",
                       "ㄹ", "ㄺ", "ㄻ", "ㄼ", "ㄽ", "ㄾ", "ㄿ", "ㅀ",
                       "ㅁ", "ㅂ", "ㅃ", "ㅄ", "ㅅ", "ㅆ", "ㅇ", "ㅈ",
                       "ㅉ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ"],
    },
}


def map_phoneme_to_japanese(source_lang: str, phoneme: str):
    """将源语言音素映射到日文假名

    Args:
        source_lang: 源语言代码 ('ZH', 'EN', 'JA', 'KO')
        phoneme: 源语言音素

    Returns:
        映射后的日文假名（字符串或列表），如果无法映射则返回原音素
    """
    phoneme = phoneme.strip()
    
    if source_lang.upper() == 'JA':
        return phoneme
    
    if phoneme in VOWEL_MAPPING:
        result = VOWEL_MAPPING[phoneme]
        if isinstance(result, list):
            return result
        return result
    
    if phoneme in CONSONANT_MAPPING:
        return CONSONANT_MAPPING[phoneme]
    
    phoneme_upper = phoneme.upper()
    if phoneme_upper in VOWEL_MAPPING:
        result = VOWEL_MAPPING[phoneme_upper]
        if isinstance(result, list):
            return result
        return result
    
    if phoneme_upper in CONSONANT_MAPPING:
        return CONSONANT_MAPPING[phoneme_upper]
    
    config = PHONEMIZER_CONFIG.get(source_lang.upper())
    if config:
        if 'primary_vowel_map' in config:
            if phoneme in config['primary_vowel_map']:
                result = config['primary_vowel_map'][phoneme]
                if isinstance(result, list):
                    return result
                return result
            if phoneme_upper in config['primary_vowel_map']:
                result = config['primary_vowel_map'][phoneme_upper]
                if isinstance(result, list):
                    return result
                return result
        if 'consonant_map' in config:
            if phoneme in config['consonant_map']:
                return config['consonant_map'][phoneme]
            if phoneme_upper in config['consonant_map']:
                return config['consonant_map'][phoneme_upper]
    
    if source_lang.upper() == 'ZH':
        return _map_chinese_phoneme(phoneme)
    
    if source_lang.upper() == 'EN':
        return _map_english_phoneme(phoneme)
    
    if source_lang.upper() == 'KO':
        return _map_korean_phoneme(phoneme)
    
    return phoneme


def _map_chinese_phoneme(phoneme: str):
    """映射中文拼音音素到日文假名"""
    config = PHONEMIZER_CONFIG.get('ZH', {})
    
    if 'full_pinyin_map' in config and phoneme in config['full_pinyin_map']:
        result = config['full_pinyin_map'][phoneme]
        if isinstance(result, list):
            return result
        return result
    
    full_map = {}
    if 'primary_vowel_map' in config:
        full_map.update(config['primary_vowel_map'])
    if 'consonant_map' in config:
        full_map.update(config['consonant_map'])
    
    if phoneme in full_map:
        result = full_map[phoneme]
        if isinstance(result, list):
            return result
        return result
    
    vowels = set(VOWEL_MAPPING.keys())
    
    result = []
    i = 0
    while i < len(phoneme):
        matched = False
        
        for length in [3, 2, 1]:
            if i + length <= len(phoneme):
                substring = phoneme[i:i+length]
                if substring in full_map:
                    mapped = full_map[substring]
                    if isinstance(mapped, list):
                        result.extend(mapped)
                    else:
                        result.append(mapped)
                    i += length
                    matched = True
                    break
        
        if matched:
            continue
        
        if i + 2 <= len(phoneme):
            two_char = phoneme[i:i+2]
            if two_char in {'zh', 'ch', 'sh'}:
                base = CONSONANT_MAPPING.get(two_char, two_char)
                if base:
                    result.append(base)
                    if i + 2 < len(phoneme):
                        next_char = phoneme[i+2]
                        palatal_map = {'a': 'ゃ', 'o': 'ょ', 'u': 'ゅ', 'e': 'ぇ', 'i': ''}
                        if next_char in palatal_map:
                            result.append(palatal_map[next_char])
                            i += 3
                            continue
                i += 2
                continue
        
        if i < len(phoneme):
            one_char = phoneme[i]
            if one_char in CONSONANT_MAPPING:
                consonant_result = CONSONANT_MAPPING[one_char]
                if consonant_result:
                    result.append(consonant_result)
                    if i + 1 < len(phoneme):
                        next_char = phoneme[i+1]
                        if next_char in vowels:
                            vowel_result = VOWEL_MAPPING[next_char]
                            if isinstance(vowel_result, list):
                                result.extend(vowel_result)
                            else:
                                result.append(vowel_result)
                            i += 2
                            continue
                i += 1
                continue
            if one_char in vowels:
                vowel_result = VOWEL_MAPPING[one_char]
                if isinstance(vowel_result, list):
                    result.extend(vowel_result)
                else:
                    result.append(vowel_result)
                i += 1
                continue
            result.append(one_char)
            i += 1
    
    if len(result) == 1:
        return result[0]
    return result


def _map_english_phoneme(phoneme: str) -> str:
    """映射英文音素到日文假名"""
    cleaned = phoneme
    if cleaned[-1].isdigit():
        cleaned = cleaned[:-1]
    
    if cleaned in VOWEL_MAPPING:
        return VOWEL_MAPPING[cleaned]
    if cleaned in CONSONANT_MAPPING:
        return CONSONANT_MAPPING[cleaned]
    
    return phoneme


def _map_korean_phoneme(phoneme: str) -> str:
    """映射韩文音素到日文假名"""
    result = ""
    for char in phoneme:
        if char in VOWEL_MAPPING:
            result += VOWEL_MAPPING[char]
        elif char in CONSONANT_MAPPING:
            mapped = CONSONANT_MAPPING[char]
            if mapped:
                result += mapped
        else:
            result += char
    return result


def map_word_to_japanese(source_lang: str, word: str) -> list:
    """将单词映射到日文假名单词列表

    Args:
        source_lang: 源语言代码
        word: 源语言单词

    Returns:
        日文假名单词列表
    """
    config = PHONEMIZER_CONFIG.get(source_lang.upper())
    if not config:
        return [word]
    
    if source_lang.upper() == 'JA':
        return [word]
    
    if source_lang.upper() == 'ZH':
        return _map_chinese_word(word)
    
    if source_lang.upper() == 'EN':
        return _map_english_word(word)
    
    if source_lang.upper() == 'KO':
        return _map_korean_word(word)
    
    return [word]


def _map_chinese_word(word: str) -> list:
    """映射中文单词"""
    result = []
    i = 0
    while i < len(word):
        if i + 2 <= len(word):
            two_char = word[i:i+2]
            if two_char in VOWEL_MAPPING:
                result.append(VOWEL_MAPPING[two_char])
                i += 2
                continue
        one_char = word[i]
        if one_char in VOWEL_MAPPING:
            result.append(VOWEL_MAPPING[one_char])
        elif one_char in CONSONANT_MAPPING:
            result.append(CONSONANT_MAPPING[one_char])
        else:
            result.append(one_char)
        i += 1
    return result


def _map_english_word(word: str) -> list:
    """映射英文单词"""
    result = []
    phonemes = word.split()
    for phoneme in phonemes:
        cleaned = phoneme
        if cleaned[-1].isdigit():
            cleaned = cleaned[:-1]
        if cleaned in VOWEL_MAPPING:
            result.append(VOWEL_MAPPING[cleaned])
        elif cleaned in CONSONANT_MAPPING:
            result.append(CONSONANT_MAPPING[cleaned])
        else:
            result.append(cleaned)
    return result


def _map_korean_word(word: str) -> list:
    """映射韩文单词"""
    result = []
    for char in word:
        if char in VOWEL_MAPPING:
            result.append(VOWEL_MAPPING[char])
        elif char in CONSONANT_MAPPING:
            mapped = CONSONANT_MAPPING[char]
            if mapped:
                result.append(mapped)
        else:
            result.append(char)
    return result
