# -*- coding: utf-8 -*-
"""torchvision 0.26 兼容补丁：让 transforms.v2.Transform.forward 兼容旧式 _transform 接口。

DEIM (2024) 用 _transform/params 旧接口编写自定义 Transform（ConvertBoxes 等），
torchvision 0.26 的 Transform.__call__ 只调用 self.transform() → NotImplementedError。
本补丁注入兼容逻辑：若子类实现 _transform 而非 transform，则用 _transform 包装。
在 import deim 前先 import 本模块即可。

用法: 在训练脚本启动处:
    import patch_tw  (或 from patch_tw import apply; apply())
"""
import sys
import io
import os
import types


def apply():
    try:
        import torchvision.transforms.v2 as T
    except Exception:
        return
    base = T.Transform

    # 记录原 forward/transform
    orig_forward = base.forward
    orig_transform = getattr(base, 'transform', None)

    # 若子类有 _transform 属性（旧接口）且自身没重写 transform，则用 _transform 提供 transform
    if not hasattr(base, '_patch_applied'):
        def _make_transform(self, inpt, params):
            if hasattr(self, '_transform') and not isinstance(getattr(base, 'transform', None) and base.__dict__.get('transform'), types.FunctionType):
                # 只有子类定义了 _transform 时才用旧接口
                if type(self).__dict__.get('_transform'):
                    return type(self)._transform(self, inpt, params)
            # 默认：调用新接口（如果有）
            if orig_transform:
                return orig_transform(self, inpt, params)
            return inpt

        # 用 transform 属性注入（仅当基类没定义 transform 时）
        if 'transform' not in base.__dict__:
            base.transform = _make_transform
        base._patch_applied = True
        print('[patch_tw] torchvision transforms.v2 兼容补丁已应用', flush=True)


if __name__ == '__main__':
    apply()
