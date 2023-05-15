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
import open_clip
import urllib
from PIL import Image
from pytriton.decorators import batch
from pytriton.model_config import ModelConfig, Tensor
from pytriton.triton import Triton
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s: %(message)s")
logger = logging.getLogger("PyTriton")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

sam_checkpoint_h = os.path.join("sam/checkpoints", "sam_vit_h_4b8939.pth")
model_type_h = "vit_h"
sam_h = sam_model_registry[model_type_h](checkpoint=sam_checkpoint_h)
sam_h.to(device=device)
sam_masks_gen_h = SamAutomaticMaskGenerator(sam_h)

sam_checkpoint_l = os.path.join("sam/checkpoints", "sam_vit_l_0b3195.pth")
model_type_l = "vit_l"
sam_l = sam_model_registry[model_type_l](checkpoint=sam_checkpoint_l)
sam_l.to(device=device)
sam_masks_gen_l = SamAutomaticMaskGenerator(sam_l)

sam_checkpoint_b = os.path.join("sam/checkpoints", "sam_vit_b_01ec64.pth")
model_type_b = "vit_b"
sam_b = sam_model_registry[model_type_b](checkpoint=sam_checkpoint_b)
sam_b.to(device=device)
sam_masks_gen_b = SamAutomaticMaskGenerator(sam_b)

openclip_b32, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
tokenizer = open_clip.get_tokenizer('ViT-B-32')
openclip_b32.to(device)

filename = "imagenet1000_labels.txt"
url = "https://gist.githubusercontent.com/yrevar/942d3a0ac09ec9e5eb3a/raw/238f720ff059c1f82f368259d1ca4ffa5dd8f9f5/imagenet1000_clsidx_to_labels.txt"

# Download the file if it does not exist
if not os.path.isfile(filename):
    urllib.request.urlretrieve(url, filename)

with open(filename) as f:
    idx2label = eval(f.read())

imagenet_labels = list(idx2label.values())
text = tokenizer(imagenet_labels)
text = text.to(device)

coca_l14, _, transform = open_clip.create_model_and_transforms(
  model_name="coca_ViT-L-14",
  pretrained="mscoco_finetuned_laion2B-s13B-b90k"
)

coca_l14.to(device)

@batch
def infer_sam_masks_h(**image):
    image = image["image"][0]
    logger.debug(f"Image data: {image.shape} ({image.size})") 
    masks = sam_masks_gen_h.generate(image)
    return format_outputs(masks)

@batch
def infer_sam_masks_l(**image):
    image = image["image"][0]
    logger.debug(f"Image data: {image.shape} ({image.size})") 
    masks = sam_masks_gen_l.generate(image)
    return format_outputs(masks)

@batch
def infer_sam_masks_b(**image):
    image = image["image"][0]
    logger.debug(f"Image data: {image.shape} ({image.size})") 
    masks = sam_masks_gen_b.generate(image)
    return format_outputs(masks)

def format_outputs(masks):
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

@batch
def infer_openclip_b32(**image):
    image = image["image"][0]
    image = Image.fromarray(image)
    image = preprocess(image).unsqueeze(0)
    image = image.to(device)
    with torch.no_grad(), torch.cuda.amp.autocast():
        image_features = openclip_b32.encode_image(image)
        text_features = openclip_b32.encode_text(text)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

    index = np.argmax(text_probs.cpu().numpy())
    label = imagenet_labels[index]

    index = np.array([index]).astype(np.int32)
    index = np.expand_dims(index, axis=0)

    label = np.frombuffer(label.encode('utf-32'), dtype=np.uint32)
    label = np.expand_dims(label, axis=0)

    return { "index": index , "label": label }

@batch
def infer_coca_l14(**image):
    image = image["image"][0]
    image = Image.fromarray(image)
    image = image.convert("RGB")
    image = transform(image).unsqueeze(0)
    image = image.to(device)

    with torch.no_grad(), torch.cuda.amp.autocast():
        generated = coca_l14.generate(image)

    label = open_clip.decode(generated[0]).split("<end_of_text>")[0].replace("<start_of_text>", "")

    label = np.frombuffer(label.encode('utf-32'), dtype=np.uint32)
    label = np.expand_dims(label, axis=0)

    return { "label": label }


# Connecting inference callback with Triton Inference Server
with Triton() as triton:
    # Load model into Triton Inference Server
    logger.debug("Loading SAM_h.")
    triton.bind(
        model_name="SAM_h",
        infer_func=infer_sam_masks_h,
        inputs=[
            Tensor(name="image", dtype=np.uint8, shape=(-1,-1,3)),
        ],
        outputs=[
            Tensor(name="segmentation", dtype=np.bool_, shape=(-1,-1,-1)),
            Tensor(name="area", dtype=np.intc, shape=(-1,)),
        ],
        config=ModelConfig(max_batch_size=1)
    )
    triton.bind(
        model_name="SAM_l",
        infer_func=infer_sam_masks_l,
        inputs=[
            Tensor(name="image", dtype=np.uint8, shape=(-1,-1,3)),
        ],
        outputs=[
            Tensor(name="segmentation", dtype=np.bool_, shape=(-1,-1,-1)),
            Tensor(name="area", dtype=np.intc, shape=(-1,)),
        ],
        config=ModelConfig(max_batch_size=1)
    )
    triton.bind(
        model_name="SAM_b",
        infer_func=infer_sam_masks_b,
        inputs=[
            Tensor(name="image", dtype=np.uint8, shape=(-1,-1,3)),
        ],
        outputs=[
            Tensor(name="segmentation", dtype=np.bool_, shape=(-1,-1,-1)),
            Tensor(name="area", dtype=np.intc, shape=(-1,)),
        ],
        config=ModelConfig(max_batch_size=1)
    )
    logger.debug("Loading OpenClip.")
    triton.bind(
        model_name="OpenClip_b32",
        infer_func=infer_openclip_b32,
        inputs=[
            Tensor(name="image", dtype=np.uint8, shape=(-1,-1,3)),
        ],
        outputs=[
            Tensor(name="index", dtype=np.int32, shape=(-1,)),
            Tensor(name="label", dtype=np.uint32, shape=(-1,)),
        ],
        config=ModelConfig(max_batch_size=1)
    )
    logger.debug("Loading Coca.")
    triton.bind(
        model_name="CoCa_l14",
        infer_func=infer_coca_l14,
        inputs=[
            Tensor(name="image", dtype=np.uint8, shape=(-1,-1,3)),
        ],
        outputs=[
            Tensor(name="label", dtype=np.uint32, shape=(-1,)),
        ],
        config=ModelConfig(max_batch_size=1)
    )
    triton.serve()
