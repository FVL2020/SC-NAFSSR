CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=4328 \
basicsr/train.py -opt options/train/NAFSSR.yml --launcher pytorch

# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=4328 \
# basicsr/train.py -opt options/train/NAFSSR_FT.yml --launcher pytorch

# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=4328 \
# basicsr/train.py -opt options/train/NAFSSR_FT_GAN.yml --launcher pytorch