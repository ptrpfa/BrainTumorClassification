from config import *
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report, matthews_corrcoef, cohen_kappa_score, hamming_loss, mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
import cv2

# Function to preprocess an image before feeding it to the model
def preprocess_image(image_path, img_size):
    img = keras.preprocessing.image.load_img(image_path, target_size=img_size)
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch

    # Preprocess the image (normalize pixel values)
    processed_img = keras.applications.mobilenet_v3.preprocess_input(img_array)
    return processed_img

# Function to predict the class of an image
def predict_class(image_path, model, class_names, img_size, verbose_output=1):
    processed_img = preprocess_image(image_path, img_size)
    predictions = model.predict(processed_img, verbose=verbose_output)
    predicted_class_index = tf.argmax(predictions[0]).numpy()
    
    return class_names[predicted_class_index]

# Function to load images and labels
def load_data(dataDir):
    imagePaths = []
    labels = []
    classNames = sorted(os.listdir(dataDir))

    for label in classNames:
        classDir = os.path.join(dataDir, label)
        if os.path.isdir(classDir):
            for imgName in os.listdir(classDir):
                imgPath = os.path.join(classDir, imgName)
                imagePaths.append(imgPath)
                labels.append(label)

    return np.array(imagePaths), np.array(labels), classNames

# Function to return the performance for a classifier
def classifier_metrics(list_y, list_pred, print_results=False):
    # Obtain metrics
    results = {
        "accuracy": accuracy_score(list_y, list_pred),
        "precision": precision_score(list_y, list_pred, average='macro'),
        "recall": recall_score(list_y, list_pred, average='macro'),
        "f1": f1_score(list_y, list_pred, average='macro'),
        "mcc": matthews_corrcoef(list_y, list_pred),
        "kappa": cohen_kappa_score(list_y, list_pred),
        "hamming_loss_val": hamming_loss(list_y, list_pred),
        "cm": confusion_matrix(list_y, list_pred),
        "class_report": classification_report(list_y, list_pred),
    }
    
    if(print_results):
        print("Accuracy:", results['accuracy'])                                    # Model Accuracy: How often is the classifier correct
        print("Precision:", results['precision'])                                  # Model Precision: what percentage of positive tuples are labeled as such?
        print("Recall:", results['recall'])                                        # Model Recall: what percentage of positive tuples are labelled as such?
        print("F1 Score:", results['f1'])                                          # F1 Score: The weighted average of Precision and Recall
        print("Matthews Correlation Coefficient (MCC):", results['mcc'])           # Matthews Correlation Coefficient (MCC): Measures the quality of binary classifications
        print("Cohen's Kappa:", results['kappa'])                                  # Cohen's Kappa: Measures inter-rater agreement for categorical items    
        print("Hamming Loss:", results['hamming_loss_val'], end='\n\n')            # Hamming Loss: The fraction of labels that are incorrectly predicted
        print("Confusion Matrix:\n", results['cm'], end="\n\n")
        print("Classification Report:\n", results['class_report'], end="\n\n\n")
        
    return results

# Function to compute performance of model
def get_model_metrics(dataset, model, num_batches):
    labels = []
    predictions = []
    for batch_index in range(num_batches):
        images, lbls = next(dataset)
        preds = model.predict(images, verbose=0)
        labels.extend(lbls)
        predictions.extend(np.argmax(preds, axis=1))
    
    y_true = np.argmax(labels, axis=1)
    classifier_metrics(y_true, predictions, print_results=True)