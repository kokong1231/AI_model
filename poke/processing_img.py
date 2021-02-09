import numpy as np
import os
# import shutil

from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as img


# path = './Image/'
# file_list = os.listdir(path)

image = []
# plt.imshow(image)
# plt.show()

# pix = np.array(image)
for x in os.listdir('./image/'):
    image.append(img.imread('./image/'+x))


print(len(image))



'''
for y in os.listdir('./jpg/'):
    a = Image.open('./jpg/'+y)
    a.save('./png/'+y.split('.')[0]+'.png')
'''

'''
temp = []

for z in os.listdir('./png/'):
    temp.append(img.imread('./png/'+z))
            

print(temp[0][50][60])
'''
# plt.imshow(image[0])
# plt.show()

# temp = []

'''
for y in os.listdir('./Image/'):
    if y.split('.')[1] != 'png':
        shutil.copyfile('./Image/' + y, './jpg/' + y)
'''