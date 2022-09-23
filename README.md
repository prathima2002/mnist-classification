# Convolutional Deep Neural Network for Digit Classification

## AIM

To Develop a convolutional deep neural network for digit classification and to verify the response for scanned handwritten images.

## PROBLEM STATEMENT AND DATASET

## NEURAL NETWORK MODEL
![1c94e938-1321-4b3e-8944-cffdd4b3c3fa](https://user-images.githubusercontent.com/108709865/191960937-dd834671-720c-4401-a0a7-114776b2ff7e.png)


## DESIGN STEPS

### Step 1:
Start by importing all the necessary libraries. And load the Data into Test sets and Training sets.

### Step 2:
Then we move to normalization and encoding of the data.

### Step 3:
The Model is then built using a Conv2D layer, MaxPool2D layer, Flatten layer, and 2 Dense layers of 16 and 10 neurons respectively.

### Step 4:
The necessary Validating parameters are visualized for inspection.

### Step 5:
Finally, we pass handwritten digits to the model for prediction.

## PROGRAM
```python
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import utils
import pandas as pd
from sklearn.metrics import classification_report,confusion_matrix
from tensorflow.keras.preprocessing import image

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train.shape

X_test.shape

single_image= X_train[100]

single_image.shape

plt.imshow(single_image,cmap='gray')

y_train.shape

X_train.min()

X_train.max()

X_train_scaled = X_train/255.0
X_test_scaled = X_test/255.0

X_train_scaled.min()

X_train_scaled.max()

y_train[0]

y_train_onehot = utils.to_categorical(y_train,10)
y_test_onehot = utils.to_categorical(y_test,10)

type(y_train_onehot)

y_train_onehot.shape

single_image = X_train[500]
plt.imshow(single_image,cmap='gray')

y_train_onehot[500]

X_train_scaled = X_train_scaled.reshape(-1,28,28,1)
X_test_scaled = X_test_scaled.reshape(-1,28,28,1)

model = keras.Sequential()
model.add(layers.Input(shape=(28,28,1)))
model.add(layers.Conv2D(filters=32,kernel_size=(3,3),activation='relu'))
model.add(layers.MaxPool2D(pool_size=(2,2)))
model.add(layers.Flatten())
model.add(layers.Dense(16,activation='tanh'))
model.add(layers.Dense(10,activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics='accuracy')

model.fit(X_train_scaled ,y_train_onehot, epochs=5,
          batch_size=64, 
          validation_data=(X_test_scaled,y_test_onehot))

metrics = pd.DataFrame(model.history.history)

metrics.head()

metrics[['accuracy','val_accuracy']].plot()

metrics[['loss','val_loss']].plot()

x_test_predictions = np.argmax(model.predict(X_test_scaled), axis=1)

print(confusion_matrix(y_test,x_test_predictions))

print(classification_report(y_test,x_test_predictions))

img = image.load_img('eight.png')

type(img)

img = image.load_img('eight.png')
img_tensor = tf.convert_to_tensor(np.asarray(img))
img_28 = tf.image.resize(img_tensor,(28,28))
img_28_gray = tf.image.rgb_to_grayscale(img_28)
img_28_gray_scaled = img_28_gray.numpy()/255.0


x_single_prediction = np.argmax(
    model.predict(img_28_gray_scaled.reshape(1,28,28,1)),
     axis=1)

print(x_single_prediction)

plt.imshow(img_28_gray_scaled.reshape(28,28),cmap='gray')

img_28_gray_inverted = 255.0-img_28_gray
img_28_gray_inverted_scaled = img_28_gray_inverted.numpy()/255.0


plt.imshow(img_28_gray_inverted_scaled.reshape(28,28),cmap='gray')

x_single_prediction = np.argmax(
    model.predict(img_28_gray_inverted_scaled.reshape(1,28,28,1)),
     axis=1)

print(x_single_prediction)
```

### Classification Report
![WhatsApp Image 2022-09-22 at 22 10 28 (1)](https://user-images.githubusercontent.com/108709865/191959456-dbb293a9-d084-4717-ab56-0c2f758f2876.jpeg)


### Confusion Matrix

![Screenshot 2022-09-23 173620](https://user-images.githubusercontent.com/108709865/191959860-f9fcda8e-2ac4-4814-adf5-e4a7887fc722.png)



### New Sample Data Prediction
#### Sample Input
![image](https://user-images.githubusercontent.com/108709865/191959966-e9cb7f20-1fe8-4658-ae39-335689a141b6.png)


#### Sample Output

![image](https://user-images.githubusercontent.com/108709865/191960042-24bb34aa-f691-4da1-916d-b22da9a10181.png)

## RESULT
Thus, a Convolutional Neural Model has been built to predict the handwritten digits.
