import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # Data preprocesing, Augmentation 
from tensorflow.keras.applications import MobileNetV2                # Base Network for our model 
#Importing different layers for our model
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input    # Scaling pixel intensities
from tensorflow.keras.preprocessing.image import img_to_array              # Conversion of image to array format
from tensorflow.keras.preprocessing.image import load_img                  # Resizing
# One Hot Encoding
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
# Spliting the testing and trainging data
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import os

# initializing the initial learning rate, number of epochs to train for and Batch size
INIT_LR = 1e-4
EPOCHS = 20
BS = 32

DIRECTORY = r"C:\one\dataset"
CATEGORIES = ["with_mask", "without_mask"]

print(" Please Wait till the Images are being loaded ... ")

# Data list to store the images and the Labels list to store the corresponding labels (mask ,unmasked)
data = []
labels = []

#Image Preprocessing before feeding the images to the model
for category in CATEGORIES:
	path = os.path.join(DIRECTORY, category)                 # Creating full pathe to the directory of the images
	for img in os.listdir(path):
		img_path = os.path.join(path, img)                   # Resizing the Images to 224x224 pixels ensuring all images have same sizes and for faster calculations
		image = load_img(img_path, target_size=(224, 224))   # Conversion to numpy array format
		image = img_to_array(image)                          # Scaling the pixel intensities in the input image to range [-1,1] for better convergence during training 
		image = preprocess_input(image)

data.append(image)          # Appending the images to the Data List
labels.append(category)     # Appending the corresponding label to the Labels List

# perform one-hot encoding on the labels
lb = LabelBinarizer()                # converting categorical data to binar vectors 
labels = lb.fit_transform(labels)    # fits the label binarizar to labels and transforms the labels 
labels = to_categorical(labels)      # convert vinary vectors to one hot encoded labels

# Numpy Array
data = np.array(data, dtype="float32")
labels = np.array(labels)

# Spliting the dataset into training and testing dataset
# Stratify ensures that each class in the target variable is proportionally represented in the training and testing sets. 
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.20, stratify=labels, random_state=42)

# construct the training image generator for data augmentation
aug = ImageDataGenerator(
	rotation_range=20,      # Rotation
	zoom_range=0.15,        # Magnification
	width_shift_range=0.2,  # Horizontal Shift
	height_shift_range=0.2, # Vertical Shift
	shear_range=0.15,       # Shear
	horizontal_flip=True,   # Flip
	fill_mode="nearest")    # Fill empty pixels with the nearest pixels on shifting

# load the MobileNetV2 network, Excluding the fully connected layer to nuild our own binary classification

baseModel = MobileNetV2(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))

# construct the head of the model , to be placed on the head of the base model 
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)

# The fully Connected model is placed on the top of the base model to get our actual model
model = Model(inputs=baseModel.input, outputs=headModel)

# loop over all layers in the base model and freeze them so they wont be updated during the first training process
# because the pre-trained MobileNetV2 model has already learned important features from the ImageNet dataset, and we do not want to modify these weights too much.
for layer in baseModel.layers:
	layer.trainable = False

# compile our model with Accuracy metrics for evaluation of model and binary crossentropy
print("[INFO] compiling model...")
opt = tf.keras.optimizers.legacy.Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)     #Adam optimizer to be used with specified learning rate and decay
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

# train the head of the network using the augmented data , providing steps per epoch and batch size
print("[INFO] training head...")
H = model.fit(
	aug.flow(trainX, trainY, batch_size=BS),
	steps_per_epoch=len(trainX) // BS,
	validation_data=(testX, testY),
	validation_steps=len(testX) // BS,
	epochs=EPOCHS)

# make predictions on the testing set
print("[INFO] evaluating network...")
predIdxs = model.predict(testX, batch_size=BS)

# for each image in the testing set we need to find the index of the
# label with corresponding largest predicted probability
predIdxs = np.argmax(predIdxs, axis=1)

# show a nicely formatted classification report
print(classification_report(testY.argmax(axis=1), predIdxs, target_names=lb.classes_))

# serialize the model to disk
print("[INFO] saving mask detector model...")
model.save("mask_detector.model", save_format="h5")

# plot the training loss and accuracy
N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("plot.png")