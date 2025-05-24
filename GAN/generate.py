import tensorflow as tf
from PIL import Image
import numpy as np
import os
from gan import Generator


z_dim = 100
num_images = 2000
save_dir = 'generated_images'
ckpt_path = 'generator.ckpt-1600'


generator = Generator()

_ = generator(tf.random.normal([1, 100]))

generator.load_weights('generator.ckpt-1600')



z = tf.random.normal([num_images, z_dim])
generated_imgs = generator(z, training=False).numpy()
os.makedirs(save_dir, exist_ok=True)

for i, img in enumerate(generated_imgs):
    img = ((img + 1.0) * 127.5).astype(np.uint8)
    if img.shape[-1] == 1:
        img = np.squeeze(img, axis=-1)
    im = Image.fromarray(img, mode='L')
    im.save(os.path.join(save_dir, f"gen_{i}.jpg"))


