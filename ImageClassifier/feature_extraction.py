import cv2 as cv
import matplotlib.pyplot as plt
from skimage import data, exposure
from skimage import transform
from skimage.color import rgb2gray
from skimage.feature import match_descriptors, plot_matches, SIFT, hog
import numpy as np
from sklearn import svm
from sklearn import cluster
import pickle
import os


def sift_feature_extraction(data_loader):
    """_summary_

    Args:
        data_loader (_type_): data loader for images to extract features from
    """
    for img in data_loader:
        # Converting image to grayscale
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # Applying SIFT detector
        sift = cv.SIFT_create()
        kp, d = sift.detectAndCompute(gray, None)
        return kp, d


img1 = cv.imread("./data/amazon/cifar10/train/airplane/0001.png")
img2 = cv.imread("./data/amazon/cifar10/train/airplane/0001.png")

fd1, hog_image1 = hog(
    img1,
    orientations=8,
    pixels_per_cell=(16, 16),
    cells_per_block=(1, 1),
    visualize=True,  # to return the image
    channel_axis=-1,
)

print(type(fd1))
print(fd1)
print(fd1.shape)
print(type(hog_image1))
print(hog_image1.shape)

fd2, hog_image2 = hog(
    img2,
    orientations=8,
    pixels_per_cell=(16, 16),
    cells_per_block=(1, 1),
    visualize=True,
    channel_axis=-1,
    feature_vector=True,
)

print(type(fd2))
print(fd2.shape)
print(fd2)
print(type(hog_image2))
print(hog_image2.shape)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

ax1.axis("off")
ax1.imshow(img1, cmap=plt.cm.gray)
ax1.set_title("Input image")

# Rescale histogram for better display
hog_image_rescaled = exposure.rescale_intensity(hog_image1, in_range=(0, 10))

ax2.axis("off")
ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
ax2.set_title("Histogram of Oriented Gradients")
# plt.show()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

ax1.axis("off")
ax1.imshow(img2, cmap=plt.cm.gray)
ax1.set_title("Input image")

# Rescale histogram for better display
hog_image_rescaled = exposure.rescale_intensity(hog_image2, in_range=(0, 10))

ax2.axis("off")
ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
ax2.set_title("Histogram of Oriented Gradients")
plt.show()

# my_data = data.astronaut()
# print(type(my_data))
# img1 = rgb2gray(my_data)
# Loading the image
# img1 = cv.imread("./data/amazon/cifar10/train/airplane/0001.png")
# img2 = cv.imread("./data/amazon/cifar10/train/airplane/0002.png")
# print(type(img1))
# print(img1.shape)
# # Converting image to grayscale
# gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
# gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
# print(type(gray1))
# print(gray1.shape)
# print(max(gray1.flatten()))
# print(min(gray1.flatten()))
# # Applying SIFT detector
# sift = cv.SIFT_create(nfeatures=256)
# kp1, d1 = sift.detectAndCompute(gray1, None)
# kp2, d2 = sift.detectAndCompute(gray2, None)
# print(type(kp1), type(d1))
# print(len(kp1), len(d1))
# print(type(kp2), type(d2))
# print(len(kp2), len(d2))
# # Marking the keypoint on the image using circles
# img1 = cv.drawKeypoints(
#     gray1, kp1, img1, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
# )
# img2 = cv.drawKeypoints(
#     gray2, kp2, img2, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
# )
# cv.imwrite("image1.jpg", img1)
# cv.imwrite("image2.jpg", img2)
# print(type(img1))
# img2 = transform.rotate(img1, 180)
# tform = transform.AffineTransform(scale=(1.3, 1.1), rotation=0.5, translation=(0, -200))
# img3 = transform.warp(img1, tform)

# descriptor_extractor = SIFT()

# descriptor_extractor.detect_and_extract(img1)
# keypoints1 = descriptor_extractor.keypoints
# descriptors1 = descriptor_extractor.descriptors
# # print type and length of descriptors
# print(type(descriptors1), len(descriptors1))
# print(type(keypoints1), len(keypoints1))

# descriptor_extractor.detect_and_extract(img2)
# keypoints2 = descriptor_extractor.keypoints
# descriptors2 = descriptor_extractor.descriptors
# print(type(descriptors2), len(descriptors2))
# print(type(keypoints2), len(keypoints2))
# descriptor_extractor.detect_and_extract(img3)
# keypoints3 = descriptor_extractor.keypoints
# descriptors3 = descriptor_extractor.descriptors

# matches12 = match_descriptors(
#     descriptors1, descriptors2, max_ratio=0.6, cross_check=True
# )
# matches13 = match_descriptors(
#     descriptors1, descriptors3, max_ratio=0.6, cross_check=True
# )
# fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(11, 8))

# plt.gray()

# plot_matches(ax[0, 0], img1, img2, keypoints1, keypoints2, matches12)
# ax[0, 0].axis("off")
# ax[0, 0].set_title("Original Image vs. Flipped Image\n" "(all keypoints and matches)")

# plot_matches(ax[1, 0], img1, img3, keypoints1, keypoints3, matches13)
# ax[1, 0].axis("off")
# ax[1, 0].set_title(
#     "Original Image vs. Transformed Image\n" "(all keypoints and matches)"
# )

# plot_matches(
#     ax[0, 1], img1, img2, keypoints1, keypoints2, matches12[::15], only_matches=True
# )
# ax[0, 1].axis("off")
# ax[0, 1].set_title(
#     "Original Image vs. Flipped Image\n" "(subset of matches for visibility)"
# )

# plot_matches(
#     ax[1, 1], img1, img3, keypoints1, keypoints3, matches13[::15], only_matches=True
# )
# ax[1, 1].axis("off")
# ax[1, 1].set_title(
#     "Original Image vs. Transformed Image\n" "(subset of matches for visibility)"
# )

# plt.tight_layout()
# plt.show()
