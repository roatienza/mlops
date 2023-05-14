'''
PyTriton SAM server

Usage:
    python3 server.py

Author:
    Rowel Atienza
    rowel@eee.upd.edu.ph

'''


import torch
import numpy as np
import os
import logging
from pytriton.decorators import batch
from pytriton.model_config import ModelConfig, Tensor
from pytriton.triton import Triton
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s: %(message)s")
logger = logging.getLogger("SAM logger")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
sam_checkpoint = os.path.join("checkpoints", "sam_vit_h_4b8939.pth")
model_type = "vit_h"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
sam_masks_gen = SamAutomaticMaskGenerator(sam)

@batch
def infer_sam_masks(**image):
    image = image["image"][0]
    logger.debug(f"Image data: {image.shape} ({image.size})") 
    masks = sam_masks_gen.generate(image)
    segmentation = []
    area = []
    for i, mask in enumerate(masks):
        for k,v in mask.items():
            if k == "segmentation":
                segmentation.append(v)
            elif k == "area":
                area.append(v)

    segmentation = np.array(segmentation).astype(np.bool_)
    segmentation = np.expand_dims(segmentation, axis=0)
    area = np.array(area).astype(np.intc)
    area = np.expand_dims(area, axis=0)
    return { "segmentation" : segmentation, "area" : area }


# Connecting inference callback with Triton Inference Server
with Triton() as triton:
    # Load model into Triton Inference Server
    logger.debug("Loading SAM.")
    triton.bind(
        model_name="SAM",
        infer_func=infer_sam_masks,
        inputs=[
            Tensor(name="image", dtype=np.uint8, shape=(-1,-1,3)),
        ],
        outputs=[
            Tensor(name="segmentation", dtype=np.bool_, shape=(-1,-1,-1)),
            Tensor(name="area", dtype=np.intc, shape=(-1,)),
        ],
        config=ModelConfig(max_batch_size=1)
    )
    triton.serve()
