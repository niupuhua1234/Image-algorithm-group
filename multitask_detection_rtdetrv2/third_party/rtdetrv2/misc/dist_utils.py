import torch.distributed as dist
def is_dist_available_and_initialized():
    return dist.is_available() and dist.is_initialized()
def get_world_size():
    return dist.get_world_size() if is_dist_available_and_initialized() else 1
