"""音素补偿模块

当声库中缺少某些音素文件时，自动查找最接近的可用音素作为补偿。

补偿策略：
1. 基于发音特征的相似度匹配（元音/辅音、发音部位、发音方法）
2. 基于文本相似度的匹配（拼音/假名相似性）
3. 基于音素规则的回退（如拗音分解为基本音素）

发音特征调整：
- 元音高度调整（high→mid→low）
- 元音前后调整（front→central→back）
- 圆唇度调整（rounded↔unrounded）
- 辅音发音部位调整
- 基频偏移调整
"""

import os
import re
import logging
from typing import Optional, List, Dict, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class PhonemeCompensator:
    """音素补偿器"""

    VOWEL_FEATURES = {
        'a': {'type': 'vowel', 'height': 'low', 'backness': 'back', 'rounding': 'unrounded'},
        'i': {'type': 'vowel', 'height': 'high', 'backness': 'front', 'rounding': 'unrounded'},
        'u': {'type': 'vowel', 'height': 'high', 'backness': 'back', 'rounding': 'rounded'},
        'e': {'type': 'vowel', 'height': 'mid', 'backness': 'front', 'rounding': 'unrounded'},
        'o': {'type': 'vowel', 'height': 'mid', 'backness': 'back', 'rounding': 'rounded'},
        'あ': {'type': 'vowel', 'height': 'low', 'backness': 'back', 'rounding': 'unrounded'},
        'い': {'type': 'vowel', 'height': 'high', 'backness': 'front', 'rounding': 'unrounded'},
        'う': {'type': 'vowel', 'height': 'high', 'backness': 'back', 'rounding': 'rounded'},
        'え': {'type': 'vowel', 'height': 'mid', 'backness': 'front', 'rounding': 'unrounded'},
        'お': {'type': 'vowel', 'height': 'mid', 'backness': 'back', 'rounding': 'rounded'},
        'AA': {'type': 'vowel', 'height': 'low', 'backness': 'back', 'rounding': 'unrounded'},
        'AE': {'type': 'vowel', 'height': 'low', 'backness': 'front', 'rounding': 'unrounded'},
        'AH': {'type': 'vowel', 'height': 'low', 'backness': 'central', 'rounding': 'unrounded'},
        'AO': {'type': 'vowel', 'height': 'mid', 'backness': 'back', 'rounding': 'rounded'},
        'IH': {'type': 'vowel', 'height': 'high', 'backness': 'front', 'rounding': 'unrounded'},
        'IY': {'type': 'vowel', 'height': 'high', 'backness': 'front', 'rounding': 'unrounded'},
        'UH': {'type': 'vowel', 'height': 'high', 'backness': 'back', 'rounding': 'rounded'},
        'UW': {'type': 'vowel', 'height': 'high', 'backness': 'back', 'rounding': 'rounded'},
    }

    CONSONANT_FEATURES = {
        'k': {'type': 'consonant', 'place': 'velar', 'manner': 'plosive', 'voicing': 'voiceless'},
        'g': {'type': 'consonant', 'place': 'velar', 'manner': 'plosive', 'voicing': 'voiced'},
        't': {'type': 'consonant', 'place': 'alveolar', 'manner': 'plosive', 'voicing': 'voiceless'},
        'd': {'type': 'consonant', 'place': 'alveolar', 'manner': 'plosive', 'voicing': 'voiced'},
        'p': {'type': 'consonant', 'place': 'bilabial', 'manner': 'plosive', 'voicing': 'voiceless'},
        'b': {'type': 'consonant', 'place': 'bilabial', 'manner': 'plosive', 'voicing': 'voiced'},
        's': {'type': 'consonant', 'place': 'alveolar', 'manner': 'fricative', 'voicing': 'voiceless'},
        'z': {'type': 'consonant', 'place': 'alveolar', 'manner': 'fricative', 'voicing': 'voiced'},
        'sh': {'type': 'consonant', 'place': 'palatal', 'manner': 'fricative', 'voicing': 'voiceless'},
        'zh': {'type': 'consonant', 'place': 'palatal', 'manner': 'fricative', 'voicing': 'voiced'},
        'f': {'type': 'consonant', 'place': 'labiodental', 'manner': 'fricative', 'voicing': 'voiceless'},
        'h': {'type': 'consonant', 'place': 'glottal', 'manner': 'fricative', 'voicing': 'voiceless'},
        'm': {'type': 'consonant', 'place': 'bilabial', 'manner': 'nasal', 'voicing': 'voiced'},
        'n': {'type': 'consonant', 'place': 'alveolar', 'manner': 'nasal', 'voicing': 'voiced'},
        'ng': {'type': 'consonant', 'place': 'velar', 'manner': 'nasal', 'voicing': 'voiced'},
        'l': {'type': 'consonant', 'place': 'alveolar', 'manner': 'lateral', 'voicing': 'voiced'},
        'r': {'type': 'consonant', 'place': 'alveolar', 'manner': 'rhotic', 'voicing': 'voiced'},
        'j': {'type': 'consonant', 'place': 'palatal', 'manner': 'affricate', 'voicing': 'voiceless'},
        'q': {'type': 'consonant', 'place': 'palatal', 'manner': 'affricate', 'voicing': 'voiceless'},
        'x': {'type': 'consonant', 'place': 'palatal', 'manner': 'fricative', 'voicing': 'voiceless'},
        'ch': {'type': 'consonant', 'place': 'retroflex', 'manner': 'affricate', 'voicing': 'voiceless'},
        'zh': {'type': 'consonant', 'place': 'retroflex', 'manner': 'affricate', 'voicing': 'voiced'},
        'sh': {'type': 'consonant', 'place': 'retroflex', 'manner': 'fricative', 'voicing': 'voiceless'},
        'c': {'type': 'consonant', 'place': 'alveolar', 'manner': 'affricate', 'voicing': 'voiceless'},
        'z': {'type': 'consonant', 'place': 'alveolar', 'manner': 'affricate', 'voicing': 'voiced'},
        'w': {'type': 'consonant', 'place': 'bilabial', 'manner': 'semivowel', 'voicing': 'voiced'},
        'y': {'type': 'consonant', 'place': 'palatal', 'manner': 'semivowel', 'voicing': 'voiced'},
        'か': {'type': 'consonant', 'place': 'velar', 'manner': 'plosive', 'voicing': 'voiceless'},
        'が': {'type': 'consonant', 'place': 'velar', 'manner': 'plosive', 'voicing': 'voiced'},
        'た': {'type': 'consonant', 'place': 'alveolar', 'manner': 'plosive', 'voicing': 'voiceless'},
        'だ': {'type': 'consonant', 'place': 'alveolar', 'manner': 'plosive', 'voicing': 'voiced'},
        'ぱ': {'type': 'consonant', 'place': 'bilabial', 'manner': 'plosive', 'voicing': 'voiceless'},
        'ば': {'type': 'consonant', 'place': 'bilabial', 'manner': 'plosive', 'voicing': 'voiced'},
        'さ': {'type': 'consonant', 'place': 'alveolar', 'manner': 'fricative', 'voicing': 'voiceless'},
        'ざ': {'type': 'consonant', 'place': 'alveolar', 'manner': 'fricative', 'voicing': 'voiced'},
        'し': {'type': 'consonant', 'place': 'palatal', 'manner': 'fricative', 'voicing': 'voiceless'},
        'じ': {'type': 'consonant', 'place': 'palatal', 'manner': 'fricative', 'voicing': 'voiced'},
        'は': {'type': 'consonant', 'place': 'glottal', 'manner': 'fricative', 'voicing': 'voiceless'},
        'ひ': {'type': 'consonant', 'place': 'palatal', 'manner': 'fricative', 'voicing': 'voiceless'},
        'ふ': {'type': 'consonant', 'place': 'labiodental', 'manner': 'fricative', 'voicing': 'voiceless'},
        'ま': {'type': 'consonant', 'place': 'bilabial', 'manner': 'nasal', 'voicing': 'voiced'},
        'な': {'type': 'consonant', 'place': 'alveolar', 'manner': 'nasal', 'voicing': 'voiced'},
        'ら': {'type': 'consonant', 'place': 'alveolar', 'manner': 'lateral', 'voicing': 'voiced'},
        'き': {'type': 'consonant', 'place': 'velar', 'manner': 'plosive', 'voicing': 'voiceless'},
        'ぎ': {'type': 'consonant', 'place': 'velar', 'manner': 'plosive', 'voicing': 'voiced'},
        'ち': {'type': 'consonant', 'place': 'palatal', 'manner': 'affricate', 'voicing': 'voiceless'},
        'ぢ': {'type': 'consonant', 'place': 'palatal', 'manner': 'affricate', 'voicing': 'voiced'},
        'し': {'type': 'consonant', 'place': 'palatal', 'manner': 'fricative', 'voicing': 'voiceless'},
        'B': {'type': 'consonant', 'place': 'bilabial', 'manner': 'plosive', 'voicing': 'voiced'},
        'P': {'type': 'consonant', 'place': 'bilabial', 'manner': 'plosive', 'voicing': 'voiceless'},
        'D': {'type': 'consonant', 'place': 'alveolar', 'manner': 'plosive', 'voicing': 'voiced'},
        'T': {'type': 'consonant', 'place': 'alveolar', 'manner': 'plosive', 'voicing': 'voiceless'},
        'G': {'type': 'consonant', 'place': 'velar', 'manner': 'plosive', 'voicing': 'voiced'},
        'K': {'type': 'consonant', 'place': 'velar', 'manner': 'plosive', 'voicing': 'voiceless'},
        'S': {'type': 'consonant', 'place': 'alveolar', 'manner': 'fricative', 'voicing': 'voiceless'},
        'Z': {'type': 'consonant', 'place': 'alveolar', 'manner': 'fricative', 'voicing': 'voiced'},
        'SH': {'type': 'consonant', 'place': 'palatal', 'manner': 'fricative', 'voicing': 'voiceless'},
        'ZH': {'type': 'consonant', 'place': 'palatal', 'manner': 'fricative', 'voicing': 'voiced'},
        'F': {'type': 'consonant', 'place': 'labiodental', 'manner': 'fricative', 'voicing': 'voiceless'},
        'V': {'type': 'consonant', 'place': 'labiodental', 'manner': 'fricative', 'voicing': 'voiced'},
        'HH': {'type': 'consonant', 'place': 'glottal', 'manner': 'fricative', 'voicing': 'voiceless'},
        'M': {'type': 'consonant', 'place': 'bilabial', 'manner': 'nasal', 'voicing': 'voiced'},
        'N': {'type': 'consonant', 'place': 'alveolar', 'manner': 'nasal', 'voicing': 'voiced'},
        'NG': {'type': 'consonant', 'place': 'velar', 'manner': 'nasal', 'voicing': 'voiced'},
        'L': {'type': 'consonant', 'place': 'alveolar', 'manner': 'lateral', 'voicing': 'voiced'},
        'R': {'type': 'consonant', 'place': 'alveolar', 'manner': 'rhotic', 'voicing': 'voiced'},
        'W': {'type': 'consonant', 'place': 'bilabial', 'manner': 'semivowel', 'voicing': 'voiced'},
        'Y': {'type': 'consonant', 'place': 'palatal', 'manner': 'semivowel', 'voicing': 'voiced'},
        'CH': {'type': 'consonant', 'place': 'palatal', 'manner': 'affricate', 'voicing': 'voiceless'},
        'JH': {'type': 'consonant', 'place': 'palatal', 'manner': 'affricate', 'voicing': 'voiced'},
        'TH': {'type': 'consonant', 'place': 'dental', 'manner': 'fricative', 'voicing': 'voiceless'},
        'DH': {'type': 'consonant', 'place': 'dental', 'manner': 'fricative', 'voicing': 'voiced'},
    }

    JAPANESE_DIPHTHONGS = {
        'あい': ['あ', 'い'], 'あう': ['あ', 'う'], 'いう': ['い', 'う'],
        'いえ': ['い', 'え'], 'うい': ['う', 'い'], 'うえ': ['う', 'え'],
        'えい': ['え', 'い'], 'おい': ['お', 'い'], 'おう': ['お', 'う'],
        'かい': ['か', 'い'], 'かう': ['か', 'う'], 'けい': ['け', 'い'],
        'こい': ['こ', 'い'], 'こう': ['こ', 'う'], 'さい': ['さ', 'い'],
        'さう': ['さ', 'う'], 'せい': ['せ', 'い'], 'そい': ['そ', 'い'],
        'そう': ['そ', 'う'], 'たい': ['た', 'い'], 'たう': ['た', 'う'],
        'てい': ['て', 'い'], 'とい': ['と', 'い'], 'とう': ['と', 'う'],
        'ない': ['な', 'い'], 'なう': ['な', 'う'], 'ねい': ['ね', 'い'],
        'のい': ['の', 'い'], 'のう': ['の', 'う'], 'はい': ['は', 'い'],
        'はう': ['は', 'う'], 'へい': ['へ', 'い'], 'ほい': ['ほ', 'い'],
        'ほう': ['ほ', 'う'], 'まい': ['ま', 'い'], 'まう': ['ま', 'う'],
        'めい': ['め', 'い'], 'もい': ['も', 'い'], 'もう': ['も', 'う'],
        'らい': ['ら', 'い'], 'らう': ['ら', 'う'], 'れい': ['れ', 'い'],
        'ろい': ['ろ', 'い'], 'ろう': ['ろ', 'う'], 'わい': ['わ', 'い'],
        'わう': ['わ', 'う'],
    }

    JAPANESE_YOON_MAP = {
        'きゃ': ['き', 'ゃ'], 'きゅ': ['き', 'ゅ'], 'きょ': ['き', 'ょ'],
        'しゃ': ['し', 'ゃ'], 'しゅ': ['し', 'ゅ'], 'しょ': ['し', 'ょ'],
        'ちゃ': ['ち', 'ゃ'], 'ちゅ': ['ち', 'ゅ'], 'ちょ': ['ち', 'ょ'],
        'にゃ': ['に', 'ゃ'], 'にゅ': ['に', 'ゅ'], 'にょ': ['に', 'ょ'],
        'ひゃ': ['ひ', 'ゃ'], 'ひゅ': ['ひ', 'ゅ'], 'ひょ': ['ひ', 'ょ'],
        'みゃ': ['み', 'ゃ'], 'みゅ': ['み', 'ゅ'], 'みょ': ['み', 'ょ'],
        'りゃ': ['り', 'ゃ'], 'りゅ': ['り', 'ゅ'], 'りょ': ['り', 'ょ'],
        'ぎゃ': ['ぎ', 'ゃ'], 'ぎゅ': ['ぎ', 'ゅ'], 'ぎょ': ['ぎ', 'ょ'],
        'じゃ': ['じ', 'ゃ'], 'じゅ': ['じ', 'ゅ'], 'じょ': ['じ', 'ょ'],
        'ぢゃ': ['ぢ', 'ゃ'], 'ぢゅ': ['ぢ', 'ゅ'], 'ぢょ': ['ぢ', 'ょ'],
        'びゃ': ['び', 'ゃ'], 'びゅ': ['び', 'ゅ'], 'びょ': ['び', 'ょ'],
        'ぴゃ': ['ぴ', 'ゃ'], 'ぴゅ': ['ぴ', 'ゅ'], 'ぴょ': ['ぴ', 'ょ'],
    }

    def __init__(self, singer_path: str):
        self.singer_path = singer_path
        self._available_phonemes = None
        self._compensation_cache = {}
        self._adjustment_cache = {}

    def _scan_available_phonemes(self) -> List[str]:
        """扫描声库中所有可用的音素"""
        if self._available_phonemes is not None:
            return self._available_phonemes

        phonemes = []
        if os.path.isdir(self.singer_path):
            for root, dirs, files in os.walk(self.singer_path):
                for f in files:
                    if f.lower().endswith('.wav'):
                        phoneme_name = os.path.splitext(f)[0].replace('_', '')
                        phonemes.append(phoneme_name)

        self._available_phonemes = phonemes
        return phonemes

    def _get_phoneme_features(self, phoneme: str) -> Dict:
        """获取音素的发音特征
        
        对于复合音素（如 'らぶ'），提取首字符作为主要特征。
        """
        p = phoneme.lower().strip()
        
        if p in self.VOWEL_FEATURES:
            return self.VOWEL_FEATURES[p]
        if p in self.CONSONANT_FEATURES:
            return self.CONSONANT_FEATURES[p]
        
        for char in p:
            if char in self.VOWEL_FEATURES:
                return self.VOWEL_FEATURES[char]
            if char in self.CONSONANT_FEATURES:
                return self.CONSONANT_FEATURES[char]
        
        return {'type': 'unknown'}

    def _compute_feature_similarity(self, phoneme1: str, phoneme2: str) -> float:
        """计算两个音素基于发音特征的相似度"""
        feat1 = self._get_phoneme_features(phoneme1)
        feat2 = self._get_phoneme_features(phoneme2)

        score = 0.0
        max_score = 0.0

        if 'type' in feat1 and 'type' in feat2:
            max_score += 1
            if feat1['type'] == feat2['type']:
                score += 1

        if 'height' in feat1 and 'height' in feat2:
            max_score += 1
            if feat1['height'] == feat2['height']:
                score += 1

        if 'backness' in feat1 and 'backness' in feat2:
            max_score += 1
            if feat1['backness'] == feat2['backness']:
                score += 1

        if 'rounding' in feat1 and 'rounding' in feat2:
            max_score += 1
            if feat1['rounding'] == feat2['rounding']:
                score += 1

        if 'place' in feat1 and 'place' in feat2:
            max_score += 1
            if feat1['place'] == feat2['place']:
                score += 0.8
            else:
                related_places = {
                    'bilabial': ['labiodental'],
                    'labiodental': ['bilabial'],
                    'alveolar': ['dental', 'retroflex'],
                    'dental': ['alveolar'],
                    'retroflex': ['alveolar'],
                    'palatal': ['velar'],
                    'velar': ['palatal'],
                }
                if feat1['place'] in related_places and feat2['place'] in related_places[feat1['place']]:
                    score += 0.4

        if 'manner' in feat1 and 'manner' in feat2:
            max_score += 1
            if feat1['manner'] == feat2['manner']:
                score += 1

        if 'voicing' in feat1 and 'voicing' in feat2:
            max_score += 1
            if feat1['voicing'] == feat2['voicing']:
                score += 0.5

        return score / max_score if max_score > 0 else 0.0

    def _compute_text_similarity(self, phoneme1: str, phoneme2: str) -> float:
        """计算两个音素的文本相似度"""
        p1 = phoneme1.lower()
        p2 = phoneme2.lower()

        if p1 == p2:
            return 1.0

        if p1 in p2 or p2 in p1:
            return 0.7

        vowels = set('aeiouあいうえお')
        p1_vowels = set(c for c in p1 if c in vowels)
        p2_vowels = set(c for c in p2 if c in vowels)

        if p1_vowels and p2_vowels:
            vowel_overlap = len(p1_vowels & p2_vowels) / max(len(p1_vowels), len(p2_vowels))
            return 0.3 + vowel_overlap * 0.4

        return 0.0

    def _decompose_phoneme(self, phoneme: str) -> Optional[List[str]]:
        """尝试分解复合音素"""
        if phoneme in self.JAPANESE_DIPHTHONGS:
            return self.JAPANESE_DIPHTHONGS[phoneme]
        if phoneme in self.JAPANESE_YOON_MAP:
            return self.JAPANESE_YOON_MAP[phoneme]
        
        for diphthong, components in self.JAPANESE_DIPHTHONGS.items():
            if diphthong in phoneme:
                return components
        
        for yoon, components in self.JAPANESE_YOON_MAP.items():
            if yoon in phoneme:
                return components

        return None

    def find_best_match(self, target_phoneme: str) -> Optional[Tuple[str, float]]:
        """查找最匹配的可用音素

        Args:
            target_phoneme: 目标音素

        Returns:
            (匹配的音素名, 相似度分数)，如果找不到返回 None
        """
        cache_key = target_phoneme
        if cache_key in self._compensation_cache:
            return self._compensation_cache[cache_key]

        available = self._scan_available_phonemes()
        if not available:
            return None

        if target_phoneme in available:
            self._compensation_cache[cache_key] = (target_phoneme, 1.0)
            return (target_phoneme, 1.0)

        best_match = None
        best_score = 0.0

        for candidate in available:
            feature_score = self._compute_feature_similarity(target_phoneme, candidate)
            text_score = self._compute_text_similarity(target_phoneme, candidate)
            
            combined_score = feature_score * 0.6 + text_score * 0.4

            if combined_score > best_score:
                best_score = combined_score
                best_match = candidate

        if best_match and best_score > 0.3:
            logger.info(f"音素补偿: {target_phoneme} -> {best_match} (相似度: {best_score:.2f})")
            self._compensation_cache[cache_key] = (best_match, best_score)
            return (best_match, best_score)

        decomposed = self._decompose_phoneme(target_phoneme)
        if decomposed:
            first_part = decomposed[0]
            for candidate in available:
                feature_score = self._compute_feature_similarity(first_part, candidate)
                text_score = self._compute_text_similarity(first_part, candidate)
                combined_score = feature_score * 0.6 + text_score * 0.4
                
                if combined_score > best_score:
                    best_score = combined_score
                    best_match = candidate

            if best_match and best_score > 0.3:
                logger.info(f"音素补偿(分解): {target_phoneme} -> {best_match} (相似度: {best_score:.2f})")
                self._compensation_cache[cache_key] = (best_match, best_score)
                return (best_match, best_score)

        return None

    def find_compensation_file(self, target_phoneme: str, relative_path: str) -> Optional[str]:
        """查找补偿音素的音频文件路径

        Args:
            target_phoneme: 目标音素
            relative_path: 原始相对路径

        Returns:
            补偿音素的文件路径，如果找不到返回 None
        """
        match = self.find_best_match(target_phoneme)
        if not match:
            return None

        matched_phoneme, _ = match
        dir_path = os.path.dirname(relative_path)
        
        if dir_path:
            new_relative_path = os.path.join(dir_path, f"_{matched_phoneme}.wav")
        else:
            new_relative_path = f"_{matched_phoneme}.wav"

        full_path = os.path.join(self.singer_path, new_relative_path)
        if os.path.exists(full_path):
            return new_relative_path

        # 精确文件名匹配（_<phoneme>.wav 或 <phoneme>.wav），不用子串跨库乱找，
        # 避免加载到错误子声库/相似音素（多音色混合、静音）
        for root, dirs, files in os.walk(self.singer_path):
            for f in files:
                low = f.lower()
                if low.endswith('.wav') and low in (f"_{matched_phoneme}.wav", f"{matched_phoneme}.wav"):
                    rel_path = os.path.relpath(os.path.join(root, f), self.singer_path)
                    return rel_path

        return None

    def compute_adjustment_params(self, target_phoneme: str, matched_phoneme: str) -> Dict[str, float]:
        """计算发音特征调整参数

        根据目标音素和补偿音素的发音特征差异，计算需要调整的参数：
        - formant_shift: 共振峰偏移（频率缩放因子）
        - pitch_shift_cent: 基频偏移（cent）
        - brightness: 明亮度调整
        - roundness: 圆唇度调整

        Args:
            target_phoneme: 目标音素
            matched_phoneme: 补偿音素

        Returns:
            调整参数字典
        """
        # 2026-08-04: 跨语种补偿已归档暂停。此前该逻辑会在正常 VCV/単独音渲染中
        # 误触发（如 oto 路径经 find_wav_file/补偿分支解析时，target=文件名
        # "おれるりら" vs matched 不一致），给每个音素算出非默认 pitch_shift/
        # 且逐音素计算开销大（8 音素句 64s）。统一返回默认（无调整）。
        return {
            'formant_shift': 1.0,
            'pitch_shift_cent': 0.0,
            'brightness': 1.0,
            'roundness': 0.0,
        }
        # —— 以下为归档前的实现（不再执行）——

        if target_feat['type'] == 'vowel' and matched_feat['type'] == 'vowel':
            height_map = {'high': 1.0, 'mid': 0.5, 'low': 0.0}
            backness_map = {'front': -1.0, 'central': 0.0, 'back': 1.0}
            rounding_map = {'rounded': 1.0, 'unrounded': 0.0}

            target_height = height_map.get(target_feat.get('height', 'mid'), 0.5)
            matched_height = height_map.get(matched_feat.get('height', 'mid'), 0.5)
            height_diff = target_height - matched_height

            target_back = backness_map.get(target_feat.get('backness', 'central'), 0.0)
            matched_back = backness_map.get(matched_feat.get('backness', 'central'), 0.0)
            back_diff = target_back - matched_back

            target_round = rounding_map.get(target_feat.get('rounding', 'unrounded'), 0.0)
            matched_round = rounding_map.get(matched_feat.get('rounding', 'unrounded'), 0.0)
            round_diff = target_round - matched_round

            params['formant_shift'] = 1.0 + height_diff * 0.15 + back_diff * 0.05
            params['pitch_shift_cent'] = height_diff * 20 + back_diff * 10
            params['brightness'] = 1.0 + (1 - target_round) * 0.2 - (1 - matched_round) * 0.2
            params['roundness'] = round_diff

        elif target_feat['type'] == 'consonant' and matched_feat['type'] == 'consonant':
            place_map = {
                'bilabial': 0.0, 'labiodental': 0.1,
                'dental': 0.2, 'alveolar': 0.3,
                'retroflex': 0.4, 'palatal': 0.5,
                'velar': 0.6, 'glottal': 0.7,
            }

            target_place = place_map.get(target_feat.get('place', 'alveolar'), 0.3)
            matched_place = place_map.get(matched_feat.get('place', 'alveolar'), 0.3)
            place_diff = target_place - matched_place

            target_voiced = target_feat.get('voicing', 'voiceless') == 'voiced'
            matched_voiced = matched_feat.get('voicing', 'voiceless') == 'voiced'

            params['formant_shift'] = 1.0 + place_diff * 0.1
            params['pitch_shift_cent'] = 0
            params['brightness'] = 1.0 + (1 if target_voiced else 0) * 0.1 - (1 if matched_voiced else 0) * 0.1

        return params

    def apply_adjustment(self, audio: np.ndarray, sr: int, params: Dict[str, float]) -> np.ndarray:
        """应用发音特征调整到音频

        Args:
            audio: 原始音频
            sr: 采样率
            params: 调整参数

        Returns:
            调整后的音频
        """
        if len(audio) == 0:
            return audio

        formant_shift = params.get('formant_shift', 1.0)
        pitch_shift_cent = params.get('pitch_shift_cent', 0.0)
        brightness = params.get('brightness', 1.0)

        result = audio.copy().astype(np.float32)

        if formant_shift != 1.0 and abs(formant_shift - 1.0) > 0.01:
            result = self._apply_formant_shift(result, sr, formant_shift)

        if pitch_shift_cent != 0.0 and abs(pitch_shift_cent) > 1.0:
            result = self._apply_pitch_shift(result, sr, pitch_shift_cent)

        if brightness != 1.0 and abs(brightness - 1.0) > 0.01:
            result = self._apply_brightness(result, sr, brightness)

        return result

    def _apply_formant_shift(self, audio: np.ndarray, sr: int, shift_factor: float) -> np.ndarray:
        """应用共振峰偏移

        通过频率轴缩放实现共振峰位置调整。

        Args:
            audio: 原始音频
            sr: 采样率
            shift_factor: 频率缩放因子 (>1 升高共振峰, <1 降低共振峰)

        Returns:
            调整后的音频
        """
        n_fft = 2048
        hop_length = 256

        from scipy import signal

        f, t, Zxx = signal.stft(audio, fs=sr, nperseg=n_fft, noverlap=n_fft - hop_length)
        mag = np.abs(Zxx)
        phase = np.angle(Zxx)

        n_bins = len(f)
        shifted_mag = np.zeros_like(mag)

        for i in range(len(t)):
            if shift_factor > 1:
                orig_bins = np.linspace(0, int(n_bins / shift_factor) - 1, int(n_bins / shift_factor))
                shifted_mag[:int(n_bins / shift_factor), i] = np.interp(
                    orig_bins, np.arange(int(n_bins / shift_factor)),
                    mag[:int(n_bins / shift_factor), i]
                )
            else:
                target_bins = np.linspace(0, n_bins - 1, int(n_bins * shift_factor))
                shifted_mag[:int(n_bins * shift_factor), i] = np.interp(
                    np.arange(int(n_bins * shift_factor)), target_bins,
                    mag[:int(n_bins * shift_factor), i], left=0, right=0
                )

        _, result = signal.istft(shifted_mag * np.exp(1j * phase), fs=sr, nperseg=n_fft, noverlap=n_fft - hop_length)
        return result[:len(audio)].astype(np.float32)

    def _apply_pitch_shift(self, audio: np.ndarray, sr: int, cent: float) -> np.ndarray:
        """应用基频偏移

        Args:
            audio: 原始音频
            sr: 采样率
            cent: 基频偏移（cent）

        Returns:
            调整后的音频
        """
        from scipy import signal

        shift_ratio = 2.0 ** (cent / 1200.0)
        n_fft = 2048
        hop_length = 256

        f, t, Zxx = signal.stft(audio, fs=sr, nperseg=n_fft, noverlap=n_fft - hop_length)
        mag = np.abs(Zxx)
        phase = np.angle(Zxx)

        n_bins = len(f)
        shifted_mag = np.zeros_like(mag)

        if shift_ratio > 1:
            orig_bins = np.linspace(0, int(n_bins / shift_ratio) - 1, int(n_bins / shift_ratio))
            shifted_mag[:int(n_bins / shift_ratio), :] = np.interp(
                orig_bins, np.arange(int(n_bins / shift_ratio)), mag[:int(n_bins / shift_ratio), :],
                axis=0
            )
        else:
            target_bins = np.linspace(0, n_bins - 1, int(n_bins * shift_ratio))
            shifted_mag[:int(n_bins * shift_ratio), :] = np.interp(
                np.arange(int(n_bins * shift_ratio)), target_bins,
                mag[:int(n_bins * shift_ratio), :], axis=0, left=0, right=0
            )

        _, result = signal.istft(shifted_mag * np.exp(1j * phase), fs=sr, nperseg=n_fft, noverlap=n_fft - hop_length)
        return result[:len(audio)].astype(np.float32)

    def _apply_brightness(self, audio: np.ndarray, sr: int, factor: float) -> np.ndarray:
        """应用明亮度调整

        Args:
            audio: 原始音频
            sr: 采样率
            factor: 明亮度因子 (>1 更亮, <1 更暗)

        Returns:
            调整后的音频
        """
        from scipy import signal

        n_fft = 2048
        hop_length = 256

        f, t, Zxx = signal.stft(audio, fs=sr, nperseg=n_fft, noverlap=n_fft - hop_length)
        mag = np.abs(Zxx)
        phase = np.angle(Zxx)

        freq_bins = len(f)
        high_freq_start = int(freq_bins * 0.3)

        mag[high_freq_start:] = mag[high_freq_start:] * factor

        _, result = signal.istft(mag * np.exp(1j * phase), fs=sr, nperseg=n_fft, noverlap=n_fft - hop_length)
        return result[:len(audio)].astype(np.float32)

    def apply_adjustment_to_mel(self, mel: np.ndarray, params: Dict[str, float]) -> np.ndarray:
        """在 Mel 层面应用发音特征调整

        这是更有效的方式，因为声码器直接使用 Mel 频谱生成音频。

        Args:
            mel: Mel 频谱 (n_mels, n_frames)
            params: 调整参数

        Returns:
            调整后的 Mel 频谱
        """
        if mel.ndim != 2 or mel.shape[0] == 0 or mel.shape[1] == 0:
            return mel

        formant_shift = params.get('formant_shift', 1.0)
        brightness = params.get('brightness', 1.0)

        result = mel.copy().astype(np.float32)

        if formant_shift != 1.0 and abs(formant_shift - 1.0) > 0.01:
            result = self._apply_formant_shift_to_mel(result, formant_shift)

        if brightness != 1.0 and abs(brightness - 1.0) > 0.01:
            result = self._apply_brightness_to_mel(result, brightness)

        return result

    def _apply_formant_shift_to_mel(self, mel: np.ndarray, shift_factor: float) -> np.ndarray:
        """在 Mel 频谱层面应用共振峰偏移

        通过频率轴缩放实现共振峰位置调整。
        使用平滑插值避免伪影。

        Args:
            mel: Mel 频谱 (n_mels, n_frames)
            shift_factor: 频率缩放因子 (>1 升高共振峰, <1 降低共振峰)

        Returns:
            调整后的 Mel 频谱
        """
        n_mels, n_frames = mel.shape

        result = mel.copy().astype(np.float32)

        if shift_factor > 1.0:
            shift_ratio = 1.0 / shift_factor
            for t in range(n_frames):
                source_bins = np.arange(n_mels) * shift_ratio
                result[:, t] = np.interp(
                    np.arange(n_mels), source_bins,
                    mel[:, t], left=-20.0, right=-20.0
                )
        else:
            shift_ratio = 1.0 / shift_factor
            for t in range(n_frames):
                source_bins = np.arange(n_mels) * shift_ratio
                result[:, t] = np.interp(
                    np.arange(n_mels), source_bins,
                    mel[:, t], left=-20.0, right=-20.0
                )

        return result

    def _apply_brightness_to_mel(self, mel: np.ndarray, factor: float) -> np.ndarray:
        """在 Mel 频谱层面应用明亮度调整

        Args:
            mel: Mel 频谱 (n_mels, n_frames)
            factor: 明亮度因子 (>1 更亮, <1 更暗)

        Returns:
            调整后的 Mel 频谱
        """
        n_mels, n_frames = mel.shape
        high_freq_start = int(n_mels * 0.3)

        result = mel.copy().astype(np.float32)
        result[high_freq_start:, :] = result[high_freq_start:, :] + np.log(factor)

        return result


def find_compensation_phoneme(singer_path: str, target_phoneme: str) -> Optional[str]:
    """便捷函数：查找补偿音素"""
    compensator = PhonemeCompensator(singer_path)
    result = compensator.find_best_match(target_phoneme)
    return result[0] if result else None


def find_compensation_file(singer_path: str, target_phoneme: str, relative_path: str) -> Optional[str]:
    """便捷函数：查找补偿音素的音频文件路径"""
    compensator = PhonemeCompensator(singer_path)
    return compensator.find_compensation_file(target_phoneme, relative_path)
