import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import time
from user_functions import *


# original image ref https://towardsdatascience.com/image-processing-with-python-alternative-histogram-equalization-methods-c9088110b0ef
img_orig = cv.imread('Cambodian_Bazaar.png',cv.IMREAD_GRAYSCALE)

# image equalization using OpenCV
time1 = time.time()
img_eq_cv = cv.equalizeHist(img_orig)
time2 = time.time()
print('OpenCV algorithm time dt = ', time2 - time1, ' (sec)')

# image equalization using classical algorithm
time1 = time.time()
img_eq_classical = image_equalize_classical(img_orig)
time2 = time.time()
print('classical algorithm time dt = ', time2 - time1, ' (sec)')

# image equalization using 1 custom algorithm
time1 = time.time()
img_eq_custom1 = img_equalize_flat_histogram_method1(img_orig, num_bins=10000)
time2 = time.time()
print('1 custom algorithm algorithm time dt = ', time2 - time1, ' (sec)')

# image equalization using 2 custom algorithm
time1 = time.time()
img_eq_custom2 = img_equalize_flat_histogram_method2(img_orig, precision=3)
time2 = time.time()
print('2 custom algorithm algorithm time dt = ', time2 - time1, ' (sec)')

img_titles = ['original',
          'equalize using OpenCV',
          'equalize using \n classical algorithm',
          'equalize using \n №1 custom algorithm',
          'equalize using \n №2 custom algorithm']
images = [img_orig,
          img_eq_cv,
          img_eq_classical,
          img_eq_custom1,
          img_eq_custom2]

fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(14,5))  # по столбцам пары: изображение и гистограмма.

# Формирование изображений
i = 0
for j in range(5):
    axes[i, j].imshow(images[j], cmap='gray')
    axes[i, j].set_xticks([])
    axes[i, j].set_yticks([])
    axes[i, j].set_title(img_titles[j])
# Формирование нормированных кривых CDF и PDF
i = 1
for j in range(5):
    bin_locations, img_pdf, img_cdf = calc_pdf_and_cdf_8bit(images[j])
    axes[i, j].set_xlim([0, 255])
    axes[i, j].set_ylim([0, 1.1])
    axes[i, j].step(bin_locations, img_pdf, color='blue',lw=.15)
    axes[i, j].step(bin_locations, img_cdf, color='red',lw=.5)

fig.tight_layout()
plt.show()
