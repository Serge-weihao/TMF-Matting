# TMFNet
The offical repo for Trimap-guided feature mining and fusion network for natural image matting.
### Install
````bash
pip install -r requirement_new.txt
````
### Train command
````bash
python3 -m torch.distributed.launch --nproc_per_node=4 --master_port=$PORT \
    tools/train.py configs/mattors/gradloss/tmflaploss020.py --launcher pytorch --work-dir $WORKDIR --ckpt-least 190000 --eval-least 500000 --eval-interval 2000 --ckpt-interval 2000 --total-iters 200000 --per-gpu 16
````
### Results and models

|                              Model                               |    Training set     |  Test set | TTA |   SAD    |    MSE     |   GRAD    |   CONN    |                              Download                               |
| :--------------------------------------------------------------: | :------------: | :-------: | :--------: | :-------: | :-------: | :----------------: |:-------: | :-----------------------------------------------------------------: |
|      TMF_comp1k       | Composition-1K train|  Composition-1K test          | No |   23.0   |   4.0   |   7.5   |   18.7  |       [BaiduYun(Access Code:gjjr)](https://pan.baidu.com/s/1sy7wOFI8vEs1AJVG_2Icag)|
|      TMF_comp1k       | Composition-1K train| Composition-1K test          | Yes |   22.1   |   3.6   |   6.7   |   17.6  |         [BaiduYun(Access Code:gjjr)](https://pan.baidu.com/s/1sy7wOFI8vEs1AJVG_2Icag)|
|           TMF_ciom            | CIOM train |          CIOM test           | No | 20.2 | 1.8 | 4.8 | 13.6 | [BaiduYun(Access Code:zcww)](https://pan.baidu.com/s/1-ID40tkH8YUHz_PsWyLvLA)|
| TMF_ciom | CIOM train |        Composition-1K test          | No |  21.6   |   4.0   |   7.6   |   17.1   |   [BaiduYun(Access Code:zcww)](https://pan.baidu.com/s/1-ID40tkH8YUHz_PsWyLvLA)|
| TMF_ciom | CIOM train |   Composition-1K test          | Yes | 20.8   |   3.8   |   6.7   |   16.0   |         [BaiduYun(Access Code:zcww)](https://pan.baidu.com/s/1-ID40tkH8YUHz_PsWyLvLA)|
### Test command
````bash
./tools/dist_test.sh configs/mattors/gradloss/tmflaploss020.py comp1k.pth 2
###or with TTA
./tools/dist_test.sh configs/mattors/gradloss/tmflaploss020tta8.py comp1k.pth 2
````


### Citing
If you find TMFNet useful in your research, please consider citing:
```BibTex
@article{jiang2023trimap,
  title={Trimap-guided feature mining and fusion network for natural image matting},
  author={Jiang, Weihao and Yu, Dongdong and Xie, Zhaozhi and Li, Yaoyi and Yuan, Zehuan and Lu, Hongtao},
  journal={Computer Vision and Image Understanding},
  volume={230},
  pages={103645},
  year={2023},
  publisher={Elsevier}
}
```    
