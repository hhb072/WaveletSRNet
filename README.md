# WaveletSRNet
A pytorch implementation of Paper ["Wavelet-srnet: A wavelet-based cnn for multi-scale face super resolution"](http://openaccess.thecvf.com/content_iccv_2017/html/Huang_Wavelet-SRNet_A_Wavelet-Based_ICCV_2017_paper.html)

## Prerequisites
* Python 2.7
* PyTorch

## Run

Use the default hyparameters except changing the parameter "upscale" according to the expected upscaling factor.

CUDA_VISIBLE_DEVICES=1 python main.py --ngpu=1 --test --start_epoch=0  --test_iter=1000  --batchSize=64 --test_batchSize=32 --nrow=4  --upscale=3 --input_height=128 --output_height=128 --crop_height=128 --lr=2e-4  --nEpochs=500 --cuda

## Results
