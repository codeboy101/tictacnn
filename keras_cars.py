import numpy as np
import loader
import time
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop
from keras.utils import np_utils

# init hyperparameters
batch_size = 50
nb_classes = 2
nb_epoch = 30

# prepare data
x_train, y_train, x_test, y_test = loader.load_images('TrainImages')

x_train = x_train.reshape(len(x_train), 4000)
y_train = y_train.reshape(len(y_train), 2)

x_test = x_test.reshape(len(x_test),4000)
y_test = y_test.reshape(len(y_test), 2)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# build model
model = Sequential()
# layer 1
model.add(Dense(92, input_shape=(4000, )))
model.add(Activation('sigmoid'))
model.add(Dropout(0.2))

# layer 2
model.add(Dense(91))
model.add(Activation('sigmoid'))
model.add(Dropout(0.2))

# layer 3
model.add(Dense(90))
model.add(Activation('sigmoid'))
model.add(Dropout(0.2))

# layer 4
model.add(Dense(2))
model.add(Activation('softmax'))

model.summary()

model.compile(loss='categorical_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])

train = model.fit(x_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1, validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)

print('Test Score: ', score[0])
print('Test Accuracy: ', score[1])

