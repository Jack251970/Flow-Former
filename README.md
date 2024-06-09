# FlowFormer: A Transformer Architecture for Optical Flow
### [Project Page](https://drinkingcoder.github.io/publication/flowformer/) 

> FlowFormer: A Transformer Architecture for Optical Flow    
> [Zhaoyang Huang](https://drinkingcoder.github.io)<sup>\*</sup>, Xiaoyu Shi<sup>\*</sup>, Chao Zhang, Qiang Wang, Ka Chun Cheung, [Hongwei Qin](http://qinhongwei.com/academic/), [Jifeng Dai](https://jifengdai.org/), [Hongsheng Li](https://www.ee.cuhk.edu.hk/~hsli/)  
> ECCV 2022  


<img src="assets/teaser.png">

## News
Our FlowFormer++ and VideoFlow are accepted by CVPR and ICCV, which ranks 2nd and 1st on the Sintel benchmark!
Please also refer to our [FlowFormer++](https://github.com/XiaoyuShi97/FlowFormerPlusPlus) and [VideoFlow](https://github.com/XiaoyuShi97/VideoFlow).

## Data Preparation
Similar to RAFT, to evaluate/train FlowFormer, you will need to download the required datasets. 
* [FlyingChairs](https://lmb.informatik.uni-freiburg.de/resources/datasets/FlyingChairs.en.html#flyingchairs)
* [FlyingThings3D](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html)
* [Sintel](http://sintel.is.tue.mpg.de/)
* [KITTI2015](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=flow)
* [TubCrowdFlow](https://github.com/tsenst/CrowdFlow)
* [HD1K](http://hci-benchmark.iwr.uni-heidelberg.de/) (optional)

By default `datasets.py` will search for the datasets in these locations. You can create symbolic links to wherever the datasets were downloaded in the `datasets` folder

```Shell
├── data
    ├── Sintel
        ├── test
        ├── training
    ├── KITTI
        ├── testing
        ├── training
        ├── devkit
    ├── FlyingChairs_release
        ├── data
    ├── FlyingThings3D
        ├── frames_cleanpass
        ├── frames_finalpass
        ├── optical_flow
    ├── TUBCrowdFlow
        ├── gt_flow
        ├── gt_trajectories
        ├── images
        ├── masks
```

## Requirements
```shell
conda create --name flowformer
conda activate flowformer
conda install pytorch=1.6.0 torchvision=0.7.0 cudatoolkit=10.1 -c pytorch
pip install yacs loguru einops timm==0.4.12 imageio tqdm matplotlib tensorboard scipy opencv-python
```

## Training
The script will load the config according to the training stage. The trained model will be saved in a directory in `logs` and `checkpoints`. For example, the following script will load the config `configs/default.py`. The trained model will be saved as `logs/xxxx/final` and `checkpoints/chairs.pth`.
```shell
python -u train_FlowFormer.py --name chairs --stage chairs --validation chairs
```
To finish the entire training schedule, you can run:
```shell
./run_train.sh
```

## Models
We provide [models](https://drive.google.com/drive/folders/1K2dcWxaqOLiQ3PoqRdokrgWsGIf3yBA_?usp=sharing) trained in the four stages. The default path of the models for evaluation is:
```Shell
├── checkpoints
    ├── chairs.pth
    ├── sintel.pth
    ├── things.pth
    ├── kitti.pth
    ├── flowformer-small 
        ├── chairs.pth
        ├── sintel.pth
        ├── things.pth
    ├── things_kitti.pth
```
The models under flowformer-small is a small version of our flowformer. things_kitti.pth is the FlowFormer# introduced in our [supplementary](https://drinkingcoder.github.io/publication/flowformer/images/FlowFormer-supp.pdf), used for KITTI training set evaluation.

## Evaluation
The model to be evaluated is assigned by the `_CN.model` in the config file.

Evaluating the model on the Sintel training set and the KITTI training set. The corresponding config file is `configs/things_eval.py`.
```Shell
# with tiling technique
python evaluate_FlowFormer_tile.py --eval sintel_validation
python evaluate_FlowFormer_tile.py --eval kitti_validation --model checkpoints/things_kitti.pth
# without tiling technique
python evaluate_FlowFormer.py --dataset sintel
```
||with tile|w/o tile|
|----|-----|--------|
|clean|0.94|1.01|
|final|2.33|2.40|

Evaluating the small version model. The corresponding config file is `configs/small_things_eval.py`.
```Shell
# with tiling technique
python evaluate_FlowFormer_tile.py --eval sintel_validation --small
# without tiling technique
python evaluate_FlowFormer.py --dataset sintel --small
```
||with tile|w/o tile|
|----|-----|--------|
|clean|1.21|1.32|
|final|2.61|2.68|


Generating the submission for the Sintel and KITTI benchmarks. The corresponding config file is `configs/submission.py`.
```Shell
python evaluate_FlowFormer_tile.py --eval sintel_submission
python evaluate_FlowFormer_tile.py --eval kitti_submission
```
Visualizing the sintel dataset:
```Shell
python visualize_flow_image.py --eval_type sintel --keep_size
```
Visualizing an image sequence extracted from a video:
```Shell
python visualize_flow_image.py --eval_type seq
```
The default image sequence format is:
```Shell
├── demo_data
    ├── mihoyo
        ├── 000001.png
        ├── 000002.png
        ├── 000003.png
            .
            .
            .
        ├── 001000.png
```

## Results
The validation results with tile of FlowFormer on the TubCrowdFlow dataset are shown below. 
The results are generated by the model trained on the Sintel dataset.

|           | EPE      | AE       | IE        |
|-----------|----------|----------|-----------|
| IM01      | 0.421972 | 0.212379 | 51.086972 |
| IM02      | 0.305167 | 0.145175 | 43.489626 |
| IM03      | 0.238332 | 0.165726 | 66.957967 |
| IM04      | 0.223487 | 0.106554 | 35.952300 |
| IM05      | 1.112371 | 0.470768 | 74.283647 |
| IM01_hDyn | 2.626816 | 1.079218 | 65.503151 |
| IM02_hDyn | 2.346542 | 1.013571 | 59.352042 |
| IM03_hDyn | 2.455814 | 1.012770 | 56.903450 |
| IM04_hDyn | 3.359264 | 1.145701 | 56.903450 |
| IM05_hDyn | 6.204594 | 1.168258 | 89.512292 |

This is the demo video of IM01 dataset.

<https://github.com/Jack251970/Flow-Former/assets/53996452/0583c246-31c9-438e-bd85-e8254dadd981>

## License
FlowFormer is released under the Apache License

## Citation
```bibtex
@article{huang2022flowformer,
  title={{FlowFormer}: A Transformer Architecture for Optical Flow},
  author={Huang, Zhaoyang and Shi, Xiaoyu and Zhang, Chao and Wang, Qiang and Cheung, Ka Chun and Qin, Hongwei and Dai, Jifeng and Li, Hongsheng},
  journal={{ECCV}},
  year={2022}
}
@inproceedings{shi2023flowformer++,
  title={Flowformer++: Masked cost volume autoencoding for pretraining optical flow estimation},
  author={Shi, Xiaoyu and Huang, Zhaoyang and Li, Dasong and Zhang, Manyuan and Cheung, Ka Chun and See, Simon and Qin, Hongwei and Dai, Jifeng and Li, Hongsheng},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={1599--1610},
  year={2023}
}
@article{huang2023flowformer,
  title={FlowFormer: A Transformer Architecture and Its Masked Cost Volume Autoencoding for Optical Flow},
  author={Huang, Zhaoyang and Shi, Xiaoyu and Zhang, Chao and Wang, Qiang and Li, Yijin and Qin, Hongwei and Dai, Jifeng and Wang, Xiaogang and Li, Hongsheng},
  journal={arXiv preprint arXiv:2306.05442},
  year={2023}
}
```

## Acknowledgement

In this project, we use parts of codes in:
- [RAFT](https://github.com/princeton-vl/RAFT)
- [GMA](https://github.com/zacjiang/GMA)
- [timm](https://github.com/rwightman/pytorch-image-models)
