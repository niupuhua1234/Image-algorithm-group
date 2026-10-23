"""No global engine registry: components are constructed explicitly."""
def register(cls=None, **kwargs):
    return cls if cls is not None else lambda component: component
