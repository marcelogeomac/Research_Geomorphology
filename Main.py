import cv2
import numpy as np
import skimage
from skimage.feature import peak_local_max
from scipy import ndimage as ndi

import matplotlib.pyplot as plt

from google.colab import drive
drive.mount('/content/drive')

from google.colab.patches import cv2_imshow

!curl -o logo.png https://colab.research.google.com/img/colab_favicon_256px.png
import cv2
img = cv2.imread('/content/drive/MyDrive/DeepLearning/Reseaech_Morphology/Morphology/images/gur_exe_wb_vs2.jpg', cv2.IMREAD_UNCHANGED)
cv2_imshow(img)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh_img = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)
cv2_imshow(thresh_img)
print(type(thresh_img))

kernel = np.ones((3),np.uint8)
opening_img = cv2.morphologyEx(thresh_img, cv2.MORPH_OPEN, kernel, iterations = 9)

cv2_imshow(opening_img)

closing_img = cv2.morphologyEx(thresh_img, cv2.MORPH_CLOSE, kernel, iterations = 4)
cv2_imshow(closing_img)

dist_transform = cv2.distanceTransform(255 - closing_img, cv2.DIST_L2, 3)
cv2_imshow(dist_transform)

local_max_location = peak_local_max(dist_transform, min_distance=1, indices=True)

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=30)
kmeans.fit(local_max_location)
local_max_location = kmeans.cluster_centers_.copy()

local_max_location = local_max_location.astype(int)

local_max_location.shape
dist_transform_copy = dist_transform.copy()
for i in range(local_max_location.shape[0]):
  cv2.circle( dist_transform_copy, (local_max_location[i][1],local_max_location[i][0]  ), 5, 255 )
  
cv2_imshow(dist_transform_copy)

markers = np.zeros_like(dist_transform)
labels = np.arange(kmeans.n_clusters)
markers[local_max_location[:,0],local_max_location[:,1]   ] = labels + 1

markers = markers.astype(int)

markers_copy = markers.copy()
index_non_zero_markers = np.argwhere(markers != 0)

markers_copy = markers_copy.astype(np.uint8)

index_non_zero_markers
font = cv2.FONT_HERSHEY_SIMPLEX
for i in range(index_non_zero_markers.shape[0]):
  string_text = str(markers[index_non_zero_markers[i][0] ,index_non_zero_markers[i][1]    ])
  cv2.putText( markers_copy, string_text, (index_non_zero_markers[i][1], index_non_zero_markers[i][0]), font, 1, 255)

cv2_imshow(markers_copy)

## converti a imagem original(input) de tiff para png através do GIMP
markers = markers.astype(np.int32)
segmented = cv2.watershed(img, markers)
cv2_imshow(segmented)
print(segmented)

dpi = plt.rcParams['figure.dpi']
figsize = img.shape[1] / float(dpi), img.shape[0] / float(dpi)

plt.figure(figsize=figsize)
plt.imshow(segmented, cmap="jet")
filename = "markers.jpg"
plt.axis('off')
plt.savefig(filename, bbox_inches = 'tight', pad_inches = 0)

from PIL import Image

overlay = cv2.imread("markers.jpg")
overlay = np.asarray(overlay)

img_copy = img.copy()
overlay_copy = overlay.copy()

overlay_copy = cv2.resize(overlay_copy, (img_copy.shape[1], img_copy.shape[0]))
final_img = cv2.addWeighted(overlay_copy, 0.5, img_copy, 0.5,	0)

cv2_imshow(final_img)

## Output / Saída
img_c = img.copy() 
img_c[segmented == -1] = [255, 0, 0]
cv2_imshow(img_c)
