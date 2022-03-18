import os
os.environ['TFF_CPP_MIN_LOG_LEVEL'] = "2" 
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 28*28).astype("float32") / 255.0
x_test = x_test.reshape(-1, 28*28).astype("float32") / 255.0

model = keras.Sequential( 
    [
        layers.Dense(512, activation = 'relu'),
        layers.Dense(256, activation = 'relu'),
        layers.Dense(10)
    ]
)

model.compile(
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits= True),
    optimizer = keras.optimizers.Adam(lr = 0.001),
    metrics=[keras.metrics.SparseCategoricalAccuracy()]
    )

model.fit(x_train, y_train, batch_size = 32, epochs = 5, verbose = 2)

model.summary()
