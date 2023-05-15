'''
PyTriton OpenClip server

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
from pytriton.triton import Triton, TritonConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s: %(message)s")
logger = logging.getLogger("SAM logger")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
#print(imagenet_labels)
#print(len(imagenet_labels))
text = tokenizer(imagenet_labels)
text = text.to(device)


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
    print("index", index)
    label = imagenet_labels[index]
    print("label", label)

    index = np.array([index]).astype(np.int32)
    index = np.expand_dims(index, axis=0)

    label = np.frombuffer(label.encode('utf-32'), dtype=np.uint32)
    label = np.expand_dims(label, axis=0)

    return { "index": index , "label": label }


# Connecting inference callback with Triton Inference Server
config = TritonConfig(http_port=8010, grpc_port=8011, metrics_port=8012)
with Triton(config=config) as triton:
    # Load model into Triton Inference Server
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
    triton.serve()
