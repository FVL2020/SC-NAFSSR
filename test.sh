CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --master_port=4323 \
basicsr/test.py -opt options/test/NAFSSR.yml --launcher pytorch
