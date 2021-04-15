import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.optimizers import SGD
import numpy as np
import mlflow
import mlflow.keras
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, TensorSpec

(train_X, train_Y), (test_X, test_Y) = tf.keras.datasets.mnist.load_data()
trainX = train_X.reshape((train_X.shape[0], 28, 28, 1))
testX = test_X.reshape((test_X.shape[0], 28, 28, 1))
trainY = tf.keras.utils.to_categorical(train_Y)
testY = tf.keras.utils.to_categorical(test_Y)

model = tf.keras.models.Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(10, activation='softmax'))
opt = SGD(lr=0.01, momentum=0.9)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(trainX, trainY, epochs=1, batch_size=32, validation_data=(testX, testY))

input_schema = Schema([
  TensorSpec(np.dtype(np.uint8), (-1, 28, 28, 1)),
])
output_schema = Schema([TensorSpec(np.dtype(np.float32), (-1, 10))])
signature = ModelSignature(inputs=input_schema, outputs=output_schema)
input_example = np.array([
   [[  0,   0,   0,   0],
    [  0, 134,  25,  56],
    [253, 242, 195,   6],
    [  0,  93,  82,  82]],
   [[  0,  23,  46,   0],
    [ 33,  13,  36, 166],
    [ 76,  75,   0, 255],
    [ 33,  44,  11,  82]]
], dtype=np.uint8)

mlflow.keras.log_model(model, "mnist_cnn", signature=signature, input_example=input_example)
