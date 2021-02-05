import numpy as np
import os

from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as img


# path = './Image/'
# file_list = os.listdir(path)

image = []
# plt.imshow(image)
# plt.show()

# pix = np.array(image)
for x in os.listdir('./Image/'):
    image.append(img.imread('./Image/'+x))

plt.imshow(image[0])
plt.show()