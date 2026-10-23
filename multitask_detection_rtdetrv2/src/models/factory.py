# 算法工厂：按 model.arch 查找模型和损失的构造函数，检查二者属于同一算法族。
"""Algorithm registry, independent of business task registry."""
_ALGORITHMS = {}

# 把算法名称与模型、损失构造器一起登记；重复名称会报错，避免悄悄覆盖已有实现。
def register_algorithm(name, model_builder, criterion_builder, family):
    if name in _ALGORITHMS:
        raise KeyError(name)
    _ALGORITHMS[name] = (model_builder, criterion_builder, family)

# 延迟导入以触发注册，再校验所选模型和 loss.family 的配套关系。
def components(cfg):
    from src.models import rtdetrv2  # register lazily
    name = cfg['model']['arch']
    if name not in _ALGORITHMS:
        raise NotImplementedError('Algorithm is not integrated: '+name)
    builders = _ALGORITHMS[name]
    if builders[2] != cfg['loss']['family']:
        raise ValueError('Model and loss must belong to the same algorithm family')
    return builders

def build_model(cfg):
    return components(cfg)[0](cfg)

def build_criterion(cfg):
    return components(cfg)[1](cfg)
