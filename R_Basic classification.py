# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 20:41:41 2018

@author: RAvi KAMble
"""
 
import tensorflow as tf
mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(512, activation=tf.nn.relu),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

#model.fit(x_train, y_train, epochs=5)
#model.evaluate(x_test, y_test)

history = model.fit(x_train,
                    y_train,
                    epochs=50,
                    verbose=1)

results = model.evaluate(x_test, y_test)
#print(results)

history_dict = history.history
history_dict.keys()
#dict_keys(['loss', 'acc'])

import matplotlib.pyplot as plt

acc = history.history['acc']
loss = history.history['loss']

epochs = range(1, len(acc) + 1)

# "bo" is for "blue dot"
plt.plot(epochs, loss, 'b', label='Training loss')
# b is for "solid blue line"
plt.title('Training loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


plt.clf()   # clear figure
acc_values = history_dict['acc']

plt.plot(epochs, acc, 'b', label='Training acc')
plt.title('Training ')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()