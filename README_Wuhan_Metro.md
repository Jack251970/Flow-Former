# FlowFormer: A Transformer Architecture for Optical Flow

## Data Preparation
To evaluate FlowFormer, you will need to prepare the required datasets with `utils/label_reader.py`. \
By default `datasets.py` will search for the datasets in these locations. You can create symbolic links to wherever the datasets were downloaded in the `data` folder

```Shell
├── data
    ├── Wuhan_Metro
        ├── transfer1-1-20231231170000-20231231203000-100992192
            ├── frame_0001.json
            ├── frame_0001.png
            ├── frame_0001.pkl
            ├── frame_0002.json
            ├── frame_0002.png
            ├── frame_0002.pkl
            ├── ...
        ├── 江汉路-D口地面扶梯-2-20231231170000-20231231203000-30896731
            ├── frame_0001.json
            ├── frame_0001.png
            ├── frame_0001.pkl
            ├── frame_0002.json
            ├── frame_0002.png
            ├── frame_0002.pkl
            ├── ...
        ├── 江汉路-付费区B端扶梯-3-20231231170000-20231231203000-28677086
            ├── frame_0001.json
            ├── frame_0001.png
            ├── frame_0001.pkl
            ├── frame_0002.json
            ├── frame_0002.png
            ├── frame_0002.pkl
            ├── ...
        ├── 江汉路-厅东闸机1-3-20231231170000-20231231203000-29300357
            ├── frame_0001.json
            ├── frame_0001.png
            ├── frame_0001.pkl
            ├── frame_0002.json
            ├── frame_0002.png
            ├── frame_0002.pkl
            ├── ...
        ├── 江汉路-厅东闸机2-2-20231231170000-20231231203000-29254227
            ├── frame_0001.json
            ├── frame_0001.png
            ├── frame_0001.pkl
            ├── frame_0002.json
            ├── frame_0002.png
            ├── frame_0002.pkl
            ├── ...
```

## Requirements
Install the required packages with the following commands:

```shell
conda create --name flowformer
conda activate flowformer
conda install pytorch=1.6.0 torchvision=0.7.0 cudatoolkit=10.1 -c pytorch
pip install yacs loguru einops timm==0.4.12 imageio tqdm matplotlib tensorboard scipy opencv-python
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

## Test
Test the model on the Wuhan_Metro dataset. The corresponding config file is `configs/sintel.py`.
```Shell
python visualize_flow_image.py --eval_type wuhan --cfg sintel --data_dir data/Wuhan_Metro/ --scenes transfer1-1-20231231170000-20231231203000-100992192 江汉路-D口地面扶梯-2-20231231170000-20231231203000-30896731
```
