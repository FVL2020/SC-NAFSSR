# SC-NAFSSR
Official PyTorch implement for SC-NAFSSR (Perceptual-Oriented Stereo Image Super-Resolution Using Stereo Consistency Guided NAFSSR)

## Requirements

The codes are based on [BasicSR](https://github.com/xinntao/BasicSR).

```
pip install -r requirements.txt
python setup.py develop
```

## Train
### 1. Prepare training data 
Modify the training set and validation set path in `NAFSSR.yml`, `NAFSSR_FT.yml`, `NAFSSR_FT_GAN.yml`.

### 2. Begin to train
We apply a two-stage training strategy by commenting out the code in test.sh or uncommenting for different training configs. We also provide the GAN-based model's config in `NAFSSR_FT_GAN.yml`.

```
sh train.sh
```

## Test
### 1. Prepare test data 
Modify the test set path and pre-training model path in `NAFSSR.yml`.

### 2. Begin to test
```
sh test.sh
```
