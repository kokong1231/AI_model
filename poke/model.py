from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint, EarlyStopping

import matplotlib.pyplot as plt
import matplotlib.image as img

import numpy
from numpy.core.records import array
import tensorflow as tf

import pandas as pd

import os
#import sys



seed = 0
numpy.random.seed(seed)
tf.random.set_seed(3)




# train = pd.read_csv('./pokemon_sort.csv')
# test = pd.read_csv('./temp.csv')

train = list([0]*809 for x in range(809))
test = list([0]*809 for x in range(14))

for x in range(809):
    train[x][x] = 1

for x in range(14):
    test[x][x] = 1

data = []
data_test = []

for x in os.listdir('./Image/'):
    data.append(img.imread('./Image/'+x))

for y in os.listdir('./test/'):
    data_test.append(img.imread('./test/'+y))

print(data[15][0][1], data[16][0][1])
data = array(data)
data_test = array(data_test)

X_train, Y_class_train = array(data, array(train))
X_test, Y_class_test = array(data_test, array(test))

print("train image : %d " % (X_train.shape[0]))
print("test image : %d " % (X_test.shape[0]))



#plt.imshow(X_train[0], cmap='Greys')
# plt.show()



X_train = X_train.reshape(X_train.shape[0], 120, 120, 3).astype('float64')/255
X_test = X_test.reshape(X_test.shape[0], 120, 120, 3).astype('float64')/255



# for x in X_train[0]:
#     for y in x:
#         sys.stdout.write('%0.1f\t' % y)
#     sys.stdout.write('\n')



Y_train = np_utils.to_categorical (Y_class_train, 809)
Y_test = np_utils.to_categorical (Y_class_test, 14)



model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), input_shape=(28, 28, 1), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(512, input_dim=784, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])



MODEL_DIR = './model/'
if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)

modelpath = "./model/{epoch:02d}-{val_loss:.4f}.hdf5"
checkpointer = ModelCheckpoint(filepath=modelpath, monitor='val_loss', verbose=1, save_best_only=True)
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10)

history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), \
    epochs=30, batch_size=200, verbose=0, \
        callbacks=[early_stopping_callback, checkpointer])

print("\n Test Accuracy: %.4f" % (model.evaluate(X_test, Y_test)[1]))



# y_vloss = history.history['val_loss']
# y_loss = history.history['loss']

# x_len = numpy.arange(len(y_loss))
# plt.plot(x_len, y_vloss, marker='.', c='red', label='Testset_loss')
# plt.plot(x_len, y_loss, marker='.', c='blue', label='Trainset_loss')

# plt.legend(loc='upper right')

# plt.grid()
# plt.xlabel('epoch')
# plt.ylabel('loss')
# plt.show()


