# https://github.com/huggingface/notebooks/blob/main/examples/automatic_mask_generation.ipynb

import numpy as np
import matplotlib.pyplot as plt
import gc
import os
import tqdm
from transformers import pipeline
from PIL import Image
import glob

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    del mask
    gc.collect()

def show_masks_on_image(raw_image, masks, title=None, save=False):
  plt.imshow(np.array(raw_image))
  ax = plt.gca()
  ax.set_autoscale_on(False)
  for mask in masks:
      show_mask(mask, ax=ax, random_color=True)
  plt.axis("off")
  ax.set_title(title, fontsize=16, fontweight='bold', fontfamily='monospace', color='r')
  if save:
      plt.savefig(im_path.replace(".png", "_masked.png").replace("data", "masks"), bbox_inches='tight')
  plt.show()
  del mask
  gc.collect()

generator = pipeline("mask-generation", model="facebook/sam-vit-huge", device="cpu")

imgs = glob.glob("./data/south_bay_surf_frame_image*.png")
for i in tqdm.tqdm(range(0, len(imgs), 400)):
    im_path = imgs[i]
    raw_image = Image.open(im_path).convert("RGB")
    plt.imshow(raw_image)
    outputs = generator(raw_image, points_per_batch=32)
    masks = outputs["masks"]
    show_masks_on_image(raw_image, masks, title=os.path.basename(im_path), save=True)
