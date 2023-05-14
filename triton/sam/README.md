# Segment Anything Model (SAM)

[SAM](https://github.com/facebookresearch/segment-anything) is a versatile foundation model for vision. Using prompts such as points, bounding boxes, masks and/or texts, SAM generates masks which can be useful for downstream tasks. 

### Install

Before using the pytriton for SAM, install SAM and download its model weights.

```
pip install git+https://github.com/facebookresearch/segment-anything.git
pip install opencv-python pycocotools matplotlib onnxruntime onnx
```

```
cd triton/sam
mkdir checkpoints
cd checkpoints 
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth .
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth .
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth .

```

Other model weights can also be downloaded but for this example, `vit_h` SAM is used.

### Run the server

```
cd ..
python server.py
```

### Run the client

Open a new terminal, then run:

```
python client.py
```
