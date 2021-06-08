# PROJECT

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# Qui viene caricato il dataset che sono 60.000 immagini 28x28 pixel con degli oggetti di moda
# come per esempio delle scarpe
(train_images, train_labels), (test_images, test_labels) = keras.datasets.fashion_mnist.load_data()
# train_images.shape
# train_images
# print(train_images)

# print(train_labels)

# visto che le immagini sono degli array che vanno da 0 a 255, vogliamo riportarli a che siano da 0 a 1 e quindi
# si divede per 255 (per normalizzarli, senza si hanno dei risultati pessimi nel training, dal 0.87 normalizzato a 0.20
# senza normalizzazione)
train_images = train_images / 255.0
test_images = test_images / 255.0

# Set input shape
sample_shape = train_images[0].shape
img_width, img_height = sample_shape[0], sample_shape[1]
input_shape = (img_width, img_height, 1)

# Reshape data 
# l'ho dovuto fare seguendo https://www.machinecurve.com/index.php/question/valueerror-input-0-of-layer-sequential-is-incompatible-with-the-layer-expected-min_ndim4-found-ndim3-full-shape-received-250-28-28/
# altrimenti il primo conv2d mi dava errore
train_images = train_images.reshape(len(train_images), input_shape[0], input_shape[1], input_shape[2])
test_images  = test_images.reshape(len(test_images), input_shape[0], input_shape[1], input_shape[2])

# # Qui si crea la rete neurale sequenziale
# model = keras.Sequential([
#     # Questo e' l'input layer e visto che le immagini sono 28x28, anche l'input sara' 28x28
#     keras.layers.Flatten(input_shape=(28, 28)),
#     # Questo e' l'hidden layer (128 nodi) con la funzione di attivazione relu
#     # Dense e' dense connected, ovvero completamente interconnesso con il livello prima
#     keras.layers.Dense(128, activation=tf.nn.relu),
#     # Questo e' il dropout al 50 %
# #     keras.layers.Dropout(0.5),
#     # Questo e' il dropout al 25 %
#     keras.layers.Dropout(0.25),
#     # Questo e' l'output layer, anch'esso dense connected, con funziona attivazione softmax
#     # ed e' fatto di 10 nodi, perchè ci sono 10 oggetti da trovare    
#     keras.layers.Dense(10, activation=tf.nn.softmax)
# ])

model = keras.Sequential()
# Primo layer convolutional
model.add(keras.layers.Conv2D(
            32, 
            kernel_size=(3,3), 
            activation='relu', 
#             input_shape=(4, 28, 28, 1)
            input_shape=input_shape
            )
         )

# Secondo layer convolutional
model.add(keras.layers.Conv2D(64, kernel_size=(3,3), activation='relu'))

# Terzo layer maxpooling
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))

# Aggiungo un dropout anche qui
model.add(keras.layers.Dropout(0.25))

# Layer classici
# model.add(keras.layers.Flatten(input_shape=(28, 28)))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(128, activation=tf.nn.relu))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(10, activation=tf.nn.softmax))

# Per vedere un riepilogo della rete neurale
# model.summary()

# Qui viene compilata la rete neurale scegliendo optimizer e loss
model.compile(
#     optimizer=keras.optimizers.Adadelta(),
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])

# # Qui viene esercitata la macchina per 5 volte (epochs=5)
# # Validation data e' usata per controllare loss e metrics alla fine di ogni epoch
# # batch size sono il numero di immagini da sottoporre prima di passare alla prossima epoca
history = model.fit(train_images, train_labels, epochs=5, validation_data=(test_images, test_labels), batch_size=128)

# Evaluate the model on the test data using `evaluate`
print("Evaluate on test data")
results = model.evaluate(test_images, test_labels, batch_size=128)
print("test loss, test acc:", results)

# print(history.history)

# # Subflow 1 riga, 2 colonne, al posto 1
# plt.subplot(1,2,1)
# plt.plot(history.history['accuracy'], 'r-')

# # Subflow 1 riga, 2 colonne, al posto 2
# # plt.subplot(2,2,2)
# plt.plot(history.history['val_accuracy'], 'b-')

# # Subflow 1 riga, 2 colonne, al posto 3
# plt.subplot(1,2,2)
# plt.plot(history.history['loss'], 'r-')

# # Subflow 1 riga, 2 colonne, al posto 4
# # plt.subplot(2,2,4)
# plt.plot(history.history['val_loss'], 'b-')
# plt.show()