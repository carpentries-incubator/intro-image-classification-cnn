# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 12:56:21 2023

@author: bellf
"""

from tensorflow import keras
(train_images, train_labels), (val_images, val_labels) = keras.datasets.cifar10.load_data()

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import time

start = time.time()

#model.compile(optimizer = 'adam', 
#              **loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)**,               
#              metrics = ['accuracy'])

# Dropout

#model_ex.compile(loss = 'mse')

#history = model.fit(train_images, train_labels, epochs=10, validation_data=(val_images, val_labels))
# convert the history to a dataframe for plotting 

# NOTE this should already be in memory from previous episode

#history_pool = model_pool.fit(train_images, train_labels, epochs=10, validation_data=(val_images, val_labels))

history_pool_df = pd.DataFrame.from_dict(history_pool.history)

# plot the loss and accuracy from the training process
fig, axes = plt.subplots(1, 2)
fig.suptitle('cifar_model_pool')
sns.lineplot(ax=axes[0], data=history_pool_df[['loss', 'val_loss']])
sns.lineplot(ax=axes[1], data=history_pool_df[['accuracy', 'val_accuracy']])

# Dropout

inputs_dropout = keras.Input(shape=train_images.shape[1:])
x_dropout = keras.layers.Conv2D(50, (3, 3), activation='relu')(inputs_dropout)
x_dropout = keras.layers.MaxPooling2D((2, 2))(x_dropout)
x_dropout = keras.layers.Conv2D(50, (3, 3), activation='relu')(x_dropout)
x_dropout = keras.layers.MaxPooling2D((2, 2))(x_dropout)
x_dropout = keras.layers.Conv2D(50, (3, 3), activation='relu')(x_dropout)
x_dropout = keras.layers.Dropout(0.8)(x_dropout) # This is new!
x_dropout = keras.layers.Flatten()(x_dropout)
x_dropout = keras.layers.Dense(50, activation='relu')(x_dropout)
outputs_dropout = keras.layers.Dense(10)(x_dropout)

model_dropout = keras.Model(inputs=inputs_dropout, outputs=outputs_dropout, name="cifar_model_dropout")

model_dropout.summary()


model_dropout.compile(optimizer='adam',
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history_dropout = model_dropout.fit(train_images, train_labels, epochs=20,
                    validation_data=(val_images, val_labels))

# save dropout model
model_dropout.save('fit_outputs/model_dropout.h5')

history_dropout_df = pd.DataFrame.from_dict(history_dropout.history)

fig, axes = plt.subplots(1, 2)
fig.suptitle('cifar_model_dropout')
sns.lineplot(ax=axes[0], data=history_dropout_df[['loss', 'val_loss']])
sns.lineplot(ax=axes[1], data=history_dropout_df[['accuracy', 'val_accuracy']])

val_loss, val_acc = model_dropout.evaluate(val_images,  val_labels, verbose=2)

plt.show() #Force a new plot to be created

# Challenge Vary Dropout Rate

dropout_rates = [0.15, 0.3, 0.45, 0.6, 0.75]
val_losses_vary = []
for dropout_rate in dropout_rates:
    inputs_vary = keras.Input(shape=train_images.shape[1:])
    x_vary = keras.layers.Conv2D(50, (3, 3), activation='relu')(inputs_vary)
    x_vary = keras.layers.MaxPooling2D((2, 2))(x_vary)
    x_vary = keras.layers.Conv2D(50, (3, 3), activation='relu')(x_vary)
    x_vary = keras.layers.MaxPooling2D((2, 2))(x_vary)
    x_vary = keras.layers.Conv2D(50, (3, 3), activation='relu')(x_vary)
    x_vary = keras.layers.Dropout(dropout_rate)(x_vary)
    x_vary = keras.layers.Flatten()(x_vary)
    x_vary = keras.layers.Dense(50, activation='relu')(x_vary)
    outputs_vary = keras.layers.Dense(10)(x_vary)

    model_vary = keras.Model(inputs=inputs_vary, outputs=outputs_vary, name="cifar_model_vary")

    model_vary.compile(optimizer = 'adam',
              loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics = ['accuracy'])

    model_vary.fit(train_images, train_labels, epochs=20,
                    validation_data=(val_images, val_labels))

    val_loss_vary, val_acc_vary = model_vary.evaluate(val_images,  val_labels)
    val_losses_vary.append(val_loss_vary)

loss_df = pd.DataFrame({'dropout_rate': dropout_rates, 'val_loss_vary': val_losses_vary})

sns.lineplot(data=loss_df, x='dropout_rate', y='val_loss_vary')

model_vary.save('fit_outputs/model_vary.h5')

end = time.time()

print()
print()
print("Time taken to run program was:", end - start, "seconds")