import numpy as np
import tensorflow as tf
from tensorflow import keras

model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
model.compile(optimizer='sgd', loss='mean_squared_error')
xs = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=int)
ys = np.array([1, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5], dtype=float)
model.fit(xs, ys, epochs=700)
print(model.predict([7.0]))
