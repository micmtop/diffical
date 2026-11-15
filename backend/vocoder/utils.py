"""通用工具函数

复用自参考实现 (util/utils.py)。
"""

import os
import json


class AttrDict(dict):
    """字典属性访问"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self


def init_weights(m, mean=0.0, std=0.01):
    """初始化权重"""
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def get_padding(kernel_size, dilation=1):
    """计算 padding 大小"""
    return int((kernel_size * dilation - dilation) / 2)


class DotDict(dict):
    """点访问字典"""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f"'DotDict' object has no attribute '{name}'")

    def __setattr__(self, name, value):
        self[name] = value
