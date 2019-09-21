import os
import time
import numpy as np
import cv2
import imageio
from PIL import Image

image_dir = "images/test_images"
frame_ids = [img for img in sorted(os.listdir(image_dir)) if os.path.isfile(os.path.join(image_dir, img))]
mined_images = [np.array(Image.open(os.path.join(image_dir, frame_id))) for frame_id in frame_ids]

for i, mined_image in enumerate(mined_images):
	image_resized = cv2.resize(mined_image, (224, 224), interpolation=cv2.INTER_AREA)
	image_resized = cv2.cvtColor(image_resized, cv2.COLOR_BGRA2BGR)
	imageio.imwrite("images/test_images_resized/{0}".format(frame_ids[i]), image_resized)
