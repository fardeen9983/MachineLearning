import keras
import matplotlib.pyplot as plt
import tensorflow as tf

fashion_mnist = keras.datasets.fashion_mnist

(train_img, train_label), (test_img, test_label) = fashion_mnist.load_data()

train_img = train_img / 255.0
test_img = test_img / 255.0

plt.imshow(train_img[30])
print(train_label[30])
print(train_img[30])

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])


class MyCallBack(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        if logs.get('loss') < 0.3:
            print("desired loss achieved")
            self.model.stop_training = True


model.compile(optimizer=tf.train.AdamOptimizer(), loss="sparse_categorical_crossentropy")

model.fit(train_img, train_label, epochs=5, callbacks=[MyCallBack()])

model.evaluate(test_img, test_label)
