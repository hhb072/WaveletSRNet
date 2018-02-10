# WaveletSRNet
A pytorch implementation of Paper ["Wavelet-srnet: A wavelet-based cnn for multi-scale face super resolution"](http://openaccess.thecvf.com/content_iccv_2017/html/Huang_Wavelet-SRNet_A_Wavelet-Based_ICCV_2017_paper.html)

## Prerequisites
* Python 2.7
* PyTorch

## Run

Use the default hyparameters except changing the parameter "upscale" according to the expected upscaling factor(2, 3, 4 for 4, 8, 16 upcaling factors, respectively).

>CUDA_VISIBLE_DEVICES=1 python main.py --ngpu=1 --test --start_epoch=0  --test_iter=1000  --batchSize=64 --test_batchSize=32 --nrow=4  --upscale=3 --input_height=128 --output_height=128 --crop_height=128 --lr=2e-4  --nEpochs=500 --cuda

## Results

![](https://github.com/hhb072/WaveletSRNet/blob/master/results.png)

## Citation

If you use our codes, please cite the following paper:

>@inproceedings{huang2017wavelet,<br>
   title={Wavelet-srnet: A wavelet-based cnn for multi-scale face super resolution},<br>
   author={Huang, Huaibo and He, Ran and Sun, Zhenan and Tan, Tieniu},<br>
   booktitle={IEEE International Conference on Computer Vision },<br>
   pages={1689--1697},<br>
   year={2017}<br>
}

** The released codes are only allowed for non-commercial use. **
