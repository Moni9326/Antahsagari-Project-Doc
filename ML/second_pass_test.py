import os
import time
import numpy as np
from glob import glob
from ntpath import basename
from os.path import join, exists
from PIL import Image
from keras.models import model_from_json

# local utilities
from utils.data_utils import preprocess, deprocess_uint8, deprocess_mask

# Resolution settings
scale = 2
hr_w, hr_h = 640, 480
lr_w, lr_h = 320, 240
lr_res, lr_shape = (lr_w, lr_h), (lr_h, lr_w, 3)
hr_res, hr_shape = (hr_w, hr_h), (hr_h, hr_w, 3)

# Previously generated SESR output directory
first_pass_dir = "data/output/keras_out"
# Load only SESR outputs from previous run
sesr_paths = sorted(glob(join(first_pass_dir, "*_SESR.png")))
print("{0} SESR images loaded for second pass.".format(len(sesr_paths)))

# Load the same model
ckpt_name = "deep_sesr_2x_1d"
model_h5 = join("models/", ckpt_name + ".h5")
model_json = join("models/", ckpt_name + ".json")
assert (exists(model_h5) and exists(model_json))

with open(model_json, "r") as json_file:
    loaded_model_json = json_file.read()
generator = model_from_json(loaded_model_json)
generator.load_weights(model_h5)
print("\nModel loaded successfully.")

# Output directory for second pass
second_pass_dir = join("data/output/", "second_pass")
if not exists(second_pass_dir):
    os.makedirs(second_pass_dir)

# Inference loop
times = []
for sesr_path in sesr_paths:
    img_name = basename(sesr_path).replace('_SESR.png', '')

    # Load the previous SESR image and downscale it to LR resolution
    img_sesr = np.array(Image.open(sesr_path).resize(lr_res))
    im = np.expand_dims(preprocess(img_sesr), axis=0)

    # Predict
    s = time.time()
    gen_op = generator.predict(im)
    gen_lr, gen_hr, gen_mask = gen_op[0], gen_op[1], gen_op[2]
    tot = time.time() - s
    times.append(tot)

    # Post-process
    gen_lr = deprocess_uint8(gen_lr).reshape(lr_shape)
    gen_hr = deprocess_uint8(gen_hr).reshape(hr_shape)
    gen_mask = deprocess_mask(gen_mask).reshape(lr_h, lr_w)

    # Save second-pass outputs
    Image.fromarray(img_sesr).save(join(second_pass_dir, img_name + '_2in.png'))       # Second input
    Image.fromarray(gen_lr).save(join(second_pass_dir, img_name + '_2En.png'))        # Second-pass enhanced LR
    Image.fromarray(gen_mask).save(join(second_pass_dir, img_name + '_2Sal.png'))     # Saliency map
    Image.fromarray(gen_hr).save(join(second_pass_dir, img_name + '_2SESR.png'))      # Second-pass SESR
    print("Second pass completed for:", sesr_path)

# Print summary
if len(sesr_paths) == 0:
    print("\nNo images found for second pass.")
else:
    Ttime, Mtime = np.sum(times[1:]), np.mean(times[1:])
    print("\nTotal second-pass images: {0}".format(len(sesr_paths)))
    print("Time taken: {0:.4f} sec at {1:.2f} fps".format(Ttime, 1. / Mtime))
    print("Second-pass images saved in:", second_pass_dir)
