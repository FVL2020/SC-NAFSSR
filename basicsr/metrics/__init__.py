from copy import deepcopy

from basicsr.utils.registry import METRIC_REGISTRY
from .niqe import calculate_niqe
from .lpips import calculate_lpips
from .psnr_ssim import calculate_psnr, calculate_ssim, calculate_skimage_ssim

__all__ = ['calculate_psnr', 'calculate_ssim', 'calculate_niqe', 'calculate_lpips', 'calculate_skimage_ssim']


def calculate_metric(data, opt):
    """Calculate metric from data and options.

    Args:
        opt (dict): Configuration. It must contain:
            type (str): Model type.
    """
    opt = deepcopy(opt)
    metric_type = opt.pop('type')
    if 'lpips' in metric_type or 'score' in metric_type:
        metric = METRIC_REGISTRY.get('Ntire_score')(**data, **opt)
    else:
        metric = METRIC_REGISTRY.get(metric_type)(**data, **opt)
    return metric
