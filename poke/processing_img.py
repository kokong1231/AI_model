import numpy as np
import os

from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as img


path = './Image/'
file_list = os.listdir(path)

image = img.imread(path+file_list[0])
plt.imshow(image)
plt.show()

pix = np.array(image)
print(image.size)