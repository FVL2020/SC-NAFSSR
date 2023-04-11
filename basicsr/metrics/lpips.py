import lpips
import torch.nn as nn
from basicsr.utils.registry import METRIC_REGISTRY

@METRIC_REGISTRY.register()
def calculate_lpips(img1, img2, lpips_vgg):

    def _lpips(img1, img2):
        return lpips_vgg(img1, img2, normalize=True)
    l1, r1 = img1[:,:3].cuda(), img1[:,3:].cuda()
    l2, r2 = img2[:,:3].cuda(), img2[:,3:].cuda()
    return (_lpips(l1, l2) + _lpips(r1, r2))/2
