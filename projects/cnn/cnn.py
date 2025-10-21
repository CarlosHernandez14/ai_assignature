import numpy as np
import os
import re
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import keras
import tensorflow as tf
from keras.utils import to_categorical
from keras.models import Sequential,Model
from keras.layers import Input
from keras.layers import Dense, Dropout, Flatten

from keras.layers import (
    BatchNormalization, SeparableConv2D, MaxPooling2D, Activation, Flatten, Dropout, Dense, Conv2D
)
from keras.layers import LeakyReLU
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from typing import Any
import time, os

datasetPath = '/Users/carloskvpchc/Documents/tec_projects/ai_assignature/projects/cnn/datasets/sportimages'
destinationModelPath = '/Users/carloskvpchc/Documents/tec_projects/ai_assignature/projects/cnn/models'

dirname = os.path.join(os.getcwd(), datasetPath)
imgpath = dirname + os.sep 

images = []
directories = []
dircount = []
prevRoot=''
cant=0

print("leyendo imagenes de ",imgpath)

# READ ALL THE IMAGES OF THE DATASET RECURSIVELY
for root, dirnames, filenames in os.walk(imgpath):
    for filename in filenames:
        if re.search(r'\.(jpg|jpeg|png|bmp|tiff)$', filename):
            cant=cant+1
            filepath = os.path.join(root, filename)
            image = plt.imread(filepath)
            if(len(image.shape)==3):
                
                images.append(image)
            b = "Leyendo..." + str(cant)
            if prevRoot !=root:
                prevRoot=root
                directories.append(root)
                dircount.append(cant)
                cant=0
dircount.append(cant)

dircount = dircount[1:]
dircount[0]=dircount[0]+1
print('Directorios leidos:',len(directories))
print("Imagenes en cada directorio", dircount)
print('suma Total de imagenes en subdirs:',sum(dircount))

# TARGET GENERATION
labels=[]
indice=0
for cantidad in dircount:
    for i in range(cantidad):
        labels.append(indice)
    indice=indice+1
print("Cantidad etiquetas creadas: ",len(labels))

# MAP LABEL INDEX TO NAMES
sriesgos=[]
indice=0
for directorio in directories:
    name = directorio.split(os.sep)
    print(indice , name[len(name)-1])
    sriesgos.append(name[len(name)-1])
    indice=indice+1

# CONVERT TO NUMPY ARRAY
y = np.array(labels)
X = np.array(images, dtype=np.uint8) #convierto de lista a numpy

classes = np.unique(y)
nClasses = len(classes)
print('Total number of outputs : ', nClasses)
print('Output classes : C', classes)

# Split the data set to have 80% of data for training and 20% for validation

train_X,test_X,train_Y,test_Y = train_test_split(X, y, test_size=0.2, stratify=y, random_state=13)
print('Training data shape : ', train_X.shape, train_Y.shape)
print('Testing data shape : ', test_X.shape, test_Y.shape)

# Shows the first image of each class with their target names 

plt.figure(1, figsize=(5,5))

# Display the first image in training data
plt.subplot(121)
plt.imshow(train_X[0,:,:], cmap='gray')
plt.title("Ground Truth : {}".format(train_Y[0]))

# Display the first image in testing data
plt.subplot(122)
plt.imshow(test_X[0,:,:], cmap='gray')
plt.title("Ground Truth : {}".format(test_Y[0]))

# Convert image data to float and normalize values to a range between 0 and 1
train_X = train_X.astype('float32')
test_X = test_X.astype('float32')
train_X = train_X/255.
test_X = test_X/255.
plt.figure(2, figsize=(5,5))
plt.imshow(test_X[0,:,:])
plt.title('Sample test image')


# Convert labels to one-hot format

train_Y_one_hot = to_categorical(train_Y)
test_Y_one_hot = to_categorical(test_Y)
print('Original label:', train_Y[0])
print('After conversion to one-hot:', train_Y_one_hot[0])


# Split data for traning and validatino using one-hot format

#Mezclar todo y crear los grupos de entrenamiento y testing
train_X,valid_X,train_label,valid_label = train_test_split(train_X, train_Y_one_hot, test_size=0.2, random_state=13, stratify=train_Y)


# Verify that the set of data has been divided correctly
print(train_X.shape,valid_X.shape,train_label.shape,valid_label.shape)


# Define learning rate, Epoch

#declaramos variables con los parámetros de configuración de la red
INIT_LR = 1e-3 # Valor inicial de learning rate. El valor 1e-3 corresponde con 0.001
epochs = 38 # Cantidad de iteraciones completas al conjunto de imagenes de entrenamiento
batch_size = 64 # cantidad de imágenes que se toman a la vez en memoria

# Define the CNN model using KERAS Api
# This may include convolutional layers, activation, pooling, normalization (Dropout) and Full Connected

riesgo_model = Sequential()
riesgo_model.add(Conv2D(32, kernel_size=(3, 3), activation='linear', padding='same', input_shape=(21,28,3)))
riesgo_model.add(MaxPooling2D((2, 2), padding='same'))
riesgo_model.add(Dropout(0.3))

riesgo_model.add(Conv2D(64, kernel_size=(3, 3), activation='linear', padding='same'))
riesgo_model.add(MaxPooling2D((2, 2), padding='same'))
riesgo_model.add(Dropout(0.3))

# riesgo_model.add(Conv2D(128, kernel_size=(3, 3), activation='linear', padding='same'))
# riesgo_model.add(MaxPooling2D((2, 2), padding='same'))
# riesgo_model.add(Dropout(0.3))

riesgo_model.add(Flatten())
riesgo_model.add(Dense(64, activation='linear'))
riesgo_model.add(Dense(nClasses, activation='softmax'))

riesgo_model.summary()


# Configure lose fucntion, optimizer and the metrics usedn on the model training
# This prepares the model to be trained based on the data
riesgo_model.compile(
    loss=keras.losses.categorical_crossentropy, 
    optimizer='sgd',
    metrics=['accuracy']
)


# Performs the real training of CNN using the prepared data
# The model auto adjust the to the data, minimizing the lose

riesgo_train = riesgo_model.fit(
    train_X, 
    train_label, 
    batch_size=batch_size,
    epochs=epochs,
    verbose=1, # type:ignore
    validation_data=(valid_X, valid_label)
)

riesgo_model.save(os.path.join(destinationModelPath, 'riesgo_2layers.h5'))


test_eval = riesgo_model.evaluate(test_X, test_Y_one_hot, verbose=1) # type:ignore

print('Test loss:', test_eval[0])
print('Test accuracy:', test_eval[1])


# Make the graphic baesd on the accurracy of the model (Using validation data)

accuracy = riesgo_train.history['accuracy']
val_accuracy = riesgo_train.history['val_accuracy']
loss = riesgo_train.history['loss']
val_loss = riesgo_train.history['val_loss']
epochs = range(len(accuracy))

# Combine metrics into a single figure with two subplots
fig_metrics, (ax_acc, ax_loss) = plt.subplots(1, 2, figsize=(12, 4))
ax_acc.plot(epochs, accuracy, 'bo', label='Training accuracy')
ax_acc.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
ax_acc.set_title('Training and validation accuracy')
ax_acc.legend()

ax_loss.plot(epochs, loss, 'ro', label='Training loss')
ax_loss.plot(epochs, val_loss, 'r', label='Validation loss')
ax_loss.set_title('Training and validation loss')
ax_loss.legend()

fig_metrics.tight_layout()


predicted_classes2 = riesgo_model.predict(test_X)

predicted_classes=[]
for predicted_riesgo in predicted_classes2:
    predicted_classes.append(predicted_riesgo.tolist().index(max(predicted_riesgo)))
predicted_classes=np.array(predicted_classes)

print(predicted_classes.shape, test_Y.shape)


# Show the correct predictions based on the validation data (single 3x3 grid)
correct = np.where(predicted_classes==test_Y)[0]
print("Found %d correct labels" % len(correct))
num_correct = min(9, len(correct))
fig_corr, axes_corr = plt.subplots(3, 3, figsize=(9, 9))
for i in range(9):
    r, c = divmod(i, 3)
    ax = axes_corr[r, c]
    if i < num_correct:
        idx = correct[i]
        ax.imshow(test_X[idx].reshape(21,28,3), cmap='gray', interpolation='none')
        ax.set_title("{}, {}".format(sriesgos[predicted_classes[idx]], sriesgos[test_Y[idx]]))
        ax.axis('off')
    else:
        ax.axis('off')
fig_corr.suptitle('Correct predictions', y=0.98)
fig_corr.tight_layout(rect=(0, 0, 1, 0.97))



# Show the wrong predictions of the model (single 3x3 grid)
incorrect = np.where(predicted_classes!=test_Y)[0]
print("Found %d incorrect labels" % len(incorrect))
num_incorrect = min(9, len(incorrect))
fig_inc, axes_inc = plt.subplots(3, 3, figsize=(9, 9))
for i in range(9):
    r, c = divmod(i, 3)
    ax = axes_inc[r, c]
    if i < num_incorrect:
        idx = incorrect[i]
        ax.imshow(test_X[idx].reshape(21,28,3), cmap='gray', interpolation='none')
        ax.set_title("{}, {}".format(sriesgos[predicted_classes[idx]], sriesgos[test_Y[idx]]))
        ax.axis('off')
    else:
        ax.axis('off')
fig_inc.suptitle('Incorrect predictions', y=0.98)
fig_inc.tight_layout(rect=(0, 0, 1, 0.97))

# Show all figures at the end
plt.show()


# CREATE THE CLASIFICATION REPORT To know the accurray of the model
target_names = [sriesgos[i] for i in classes]
print(classification_report(test_Y, predicted_classes, labels=classes, target_names=target_names, zero_division=0))
