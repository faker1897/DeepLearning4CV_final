import cv2
import os
import numpy as np



os.makedirs("dataSet/train/disgust_augmented", exist_ok=True)


def augment_image(img):
    augmented = []
    img = cv2.resize(img, (48, 48))

    augmented.append(cv2.flip(img, 1))

    augmented.append(cv2.GaussianBlur(img, (3, 3), 0))

    augmented.append(cv2.equalizeHist(img))
    bright = cv2.convertScaleAbs(img, alpha=1.2, beta=10)
    augmented.append(bright)

    for angle in [20, -20]:
        M = cv2.getRotationMatrix2D((24, 24), angle, 1.0)
        rotated = cv2.warpAffine(img, M, (48, 48))
        augmented.append(rotated)

    return augmented

img_count = 0

for fname in os.listdir("dataSet/train/disgust"):
    path = os.path.join("dataSet/train/disgust", fname)
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    augmented_imgs = augment_image(img)

    for i, aug in enumerate(augmented_imgs):
        save_path = os.path.join("dataSet/train/disgust_augmented", f"aug_{img_count}_{i}.png")
        cv2.imwrite(save_path, aug)

    img_count += 1

print('accomplishment')
