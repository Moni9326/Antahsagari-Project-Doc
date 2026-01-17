#!/usr/bin/env python
"""
# > SESR Inference on a Single Local Image
"""
import os
import time
import numpy as np
from os.path import join, exists, basename
from PIL import Image
from keras.models import model_from_json

# Local utils
from utils.data_utils import preprocess, deprocess_uint8, deprocess_mask

# === CONFIG ===
input_img_path = r"/Users/polakiramesh/Desktop/4x images/set_u113_2SESR.png"  # <<<<<< UPDATE THIS
output_dir = "filtered images/"
ckpt_name = "deep_sesr_2x_1d"
model_json_path = join("models", ckpt_name + ".json")
model_weights_path = join("models", ckpt_name + ".h5")

# Image dimensions
scale = 2
hr_w, hr_h = 640, 480  # HR
lr_w, lr_h = hr_w // scale, hr_h // scale
lr_res, hr_res = (lr_w, lr_h), (hr_w, hr_h)
lr_shape, hr_shape = (lr_h, lr_w, 3), (hr_h, hr_w, 3)

# === Load Model ===
assert exists(model_json_path) and exists(model_weights_path), "Model files not found."
with open(model_json_path, "r") as f:
    model = model_from_json(f.read())
model.load_weights(model_weights_path)
print("✔ Model loaded.")

# === Load and Prepare Input Image ===
assert exists(input_img_path), "Input image not found."
img_name = basename(input_img_path).split('.')[0]
img_lrd = np.array(Image.open(input_img_path).resize(lr_res))
input_tensor = np.expand_dims(preprocess(img_lrd), axis=0)

# === Inference ===
start = time.time()
gen_lr = model.predict(input_tensor)[0]
elapsed = time.time() - start
print(f"✔ Inference done in {elapsed:.3f}s")

# === Postprocess and Save Outputs ===
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

Image.fromarray(img_lrd).save(join(output_dir, f"{img_name}_Input.png"))
Image.fromarray(deprocess_uint8(gen_lr).reshape(lr_shape)).save(join(output_dir, f"{img_name}_EnhancedLR.png"))
Image.fromarray(deprocess_uint8(gen_hr).reshape(hr_shape)).save(join(output_dir, f"{img_name}_SESR.png"))
Image.fromarray(deprocess_mask(gen_mask).reshape(lr_h, lr_w)).save(join(output_dir, f"{img_name}_Saliency.png"))

print(f"✔ Outputs saved to: {output_dir}")
