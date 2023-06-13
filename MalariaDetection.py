import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout, BatchNormalization
from keras.callbacks import ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.models import load_model
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import os

# read the path of the infected and uninfected blood cell images
infected = os.listdir("cell_images/Parasitized")
uninfected = os.listdir("cell_images/Uninfected")

# load the images into data array and load the their lables into labels array
data = []
labels = []

for i in infected:
    try:
        img = cv2.imread("cell_images/Parasitized/" + i)
        img_array = Image.fromarray(img, "RGB")
        resized_img = img_array.resize((50, 50))
        data.append(np.array(resized_img))
        labels.append(1)

    except AttributeError:
        print("")

for u in uninfected:
    try:
        img = cv2.imread("cell_images/Uninfected/" + u)
        img_array = Image.fromarray(img, "RGB")
        resized_img = img_array.resize((50, 50))
        data.append(np.array(resized_img))
        labels.append(0)

    except AttributeError:
        print("")

# convert data and labels array into numpy arrays
cells = np.array(data)
labels = np.array(labels)

# save as .npy file
np.save("Cells", cells)
np.save("Labels", labels)

# print shape of the cells and labels numpy arrays
print("Cells: {} | Labels: {}".format(cells.shape, labels.shape))

# visualization of some samples for both infected and uninfected blood cells
plt.figure(1, figsize=(15, 9))
n = 0
for i in range(49):
    n += 1
    r = np.random.randint(0, cells.shape[0], 1)
    plt.subplot(7, 7, n)
    plt.subplots_adjust(hspace=0.5, wspace=0.5)
    plt.imshow(cells[r[0]])
    plt.title("{}: {}".format("Infected" if labels[r[0]] == 1 else "Uninfected", labels[r[0]]))
    plt.xticks([])
    plt.yticks([])

plt.show()

# visualization of a sample for infected blood cell and for a uninfected blood cell
plt.figure(1, figsize=(17, 7))
plt.subplot(1, 2, 1)
plt.imshow(cells[0])
plt.title("Infected Cell")
plt.xticks([])
plt.yticks([])

plt.subplot(1, 2, 2)
plt.imshow(cells[27557])
plt.title("Uninfected Cell")
plt.xticks([])
plt.yticks([])

plt.show()

# normalization
n = np.arange(cells.shape[0])
np.random.shuffle(n)
cells = cells[n]
labels = labels[n]

cells = cells.astype(np.float32)
labels = labels.astype(np.float32)
cells = cells / 255

x_train, x, y_train, y = train_test_split(cells, labels, test_size=0.2, random_state=111)
x_eval, x_test, y_eval, y_test = train_test_split(x, y, test_size=0.5, random_state=111)

# visualization of the amount of train, test, and evaluation labels
plt.figure(1, figsize=(15, 5))
n = 0
for z, j in zip([y_train, y_eval, y_test], ["train labels", "evaluation labels", "test labels"]):
    n += 1
    plt.subplot(1, 3, n)
    sns.countplot(x=z)
    plt.title(j)

plt.show()

# print the shape of the x_train, x_eval, x_test
print("train data shape: {}, evaluation data shape: {}, test data shape: {}".format(x_train.shape, x_eval.shape, x_test.shape))

"""
For the data augmentation:
Randomly rotate some training images by 30 degrees
Randomly Zoom by 20% some training images
Randomly shift images horizontally by 10% of the width
Randomly shift images vertically by 10% of the height
Randomly flip images horizontally. Once our model is ready, we fit the training dataset.
"""
data_generator = ImageDataGenerator(
    featurewise_center=False, 
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    rotation_range=30,
    zoom_range=0.2,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    vertical_flip=False)

data_generator.fit(x_train)

# creating the model
model = Sequential([
    Conv2D(16, (2, 2), activation='relu', padding='same', input_shape=(50, 50, 3)),
    MaxPool2D((2, 2)),
    Conv2D(32, (2, 2), padding='same', activation='relu'),
    MaxPool2D((2, 2)),
    Conv2D(64, (2, 2), padding='same', activation='relu'),
    MaxPool2D((2, 2)),
    Flatten(),
    Dropout(0.2),
    Dense(500, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])
model.summary()

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# learning_rate_reduction provides the decreasing of the learning rate according to the valdation accuracy
learning_rate_reduction = ReduceLROnPlateau(monitor="val_accuracy", patience=2, verbose=1, factor=0.3, min_lr=0.000001)

# training the model
history = model.fit(data_generator.flow(x_train, y_train, batch_size=32), epochs=12, validation_data=data_generator.flow(x_eval, y_eval), callbacks=[learning_rate_reduction])

# print the loss and accuracy of the training
print("Loss of the model:", round(model.evaluate(x_test, y_test)[0], 2))
print("Accuracy of the model:", round(model.evaluate(x_test, y_test)[1] * 100, 2), "%")

# saving the model
model.save("final_model.h5")

# plot the accuracy and the loss to understand whether there is a overfitting or underfitting
epochs = [i for i in range(12)]
fig, ax = plt.subplots(1,2)
train_accuracy = history.history["accuracy"]
train_loss = history.history["loss"]
validation_accuracy = history.history["val_accuracy"]
validation_loss = history.history["val_loss"]
fig.set_size_inches(20,10)

ax[0].plot(epochs, train_accuracy, "go-", label="Training Accuracy")
ax[0].plot(epochs, validation_accuracy, "ro-", label="Validation Accuracy")
ax[0].set_title("Training & Validation Accuracy")
ax[0].legend()
ax[0].set_xlabel("Epochs")
ax[0].set_ylabel("Accuracy")

ax[1].plot(epochs, train_loss, "g-o", label="Training Loss")
ax[1].plot(epochs, validation_loss, "r-o", label="Validation Loss")
ax[1].set_title("Testing Accuracy & Loss")
ax[1].legend()
ax[1].set_xlabel("Epochs")
ax[1].set_ylabel("Training & Validation Loss")
plt.show()

# loading the model and making predictions
model_1 = load_model("final_model.h5")

probabilities = model_1.predict(x_test)
predictions = (probabilities > 0.5).astype(int)
predictions = predictions.reshape(1, -1)[0]

# print precision, recall, f1-score scores
print(classification_report(y_test, predictions, target_names = ["Uninfected (Class 0)", "Infected (Class 1)"], zero_division=1))

# making confusion matrix for the predictions
cm = confusion_matrix(y_test, predictions)
print(cm)

# convert cm to pandas dataframe
cm = pd.DataFrame(cm , index = ["0", "1"] , columns = ["0", "1"])

# visualization of cm
plt.figure(figsize=(10,10))
sns.heatmap(cm, cmap="Blues", linecolor="black", linewidth=1, annot=True, fmt="", xticklabels=["Uninfected", "Infected"], yticklabels=["Uninfected", "Infected"])
plt.show()

# seperate correct and incorrect predictions into different arrays
correct = np.nonzero(predictions == y_test)[0]
incorrect = np.nonzero(predictions != y_test)[0]

# visualization of correct predictions and incorrect predictions
i = 0
for c in correct[:6]:
    plt.subplot(4,2,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(x_test[c], cmap="gray", interpolation="none")
    plt.title("Predicted Class {}, Actual Class {}".format(predictions[c], y_test[c]))
    plt.tight_layout()
    i += 1
    
plt.show()

i = 0
for c in incorrect[:6]:
    plt.subplot(4,2,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(x_test[c], cmap="gray", interpolation="none")
    plt.title("Predicted Class {}, Actual Class {}".format(predictions[c], y_test[c]))
    plt.tight_layout()
    i += 1
    
plt.show()
