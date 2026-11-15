"""音频文件查找工具

提供给后端使用的简单音频文件查找功能。
注意：声库管理、oto.ini 解析是 OpenUTAU 的职责，后端不重复实现。
"""

import os
import logging

logger = logging.getLogger(__name__)


from typing import Optional


def find_wav_file(vb_path: str, rel_path: str) -> Optional[str]:
    """查找 wav 文件

    根据相对路径查找实际的音频文件。
    OpenUTAU 负责解析 oto.ini，后端只负责读取文件。

    Args:
        vb_path: 声库根目录
        rel_path: OpenUTAU 传来的文件路径（可能不含扩展名）

    Returns:
        完整文件路径，或 None 如果找不到
    """
    full_path = os.path.join(vb_path, rel_path)

    # 直接存在
    if os.path.exists(full_path):
        return full_path

    # 尝试添加 .wav
    wav_path = full_path + '.wav'
    if os.path.exists(wav_path):
        return wav_path

    # 尝试去掉扩展名再加 .wav
    base, _ = os.path.splitext(full_path)
    wav_path2 = base + '.wav'
    if os.path.exists(wav_path2):
        return wav_path2

    # 搜索文件（大小写不敏感）
    dir_path = os.path.dirname(full_path)
    file_name = os.path.basename(full_path).lower()

    if os.path.isdir(dir_path):
        for f in os.listdir(dir_path):
            if f.lower() == file_name or f.lower().startswith(file_name + '.'):
                return os.path.join(dir_path, f)

    logger.warning(f"音频文件找不到: {rel_path}")
    return None
