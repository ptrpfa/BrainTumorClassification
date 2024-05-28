from config import *
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report, matthews_corrcoef, cohen_kappa_score, hamming_loss, mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import os
import pickle

# Function to serialise an object into a pickle file
def save_to_pickle(file_name, save_data, complete_path=True):
    file_name_with_extension = file_name + ".pkl"
    complete_file_path = f'{EXPORT_FOLDER}/{file_name_with_extension}' if(complete_path) else file_name
    with open(complete_file_path, 'wb') as file:
        pickle.dump(save_data, file)

# Function to deserialise a pickle file
def load_from_pickle(file_name, complete_path=True):
    if(".pkl" not in file_name):
        file_name_with_extension = file_name + ".pkl"
    else:
        file_name_with_extension = file_name
    complete_file_path = f'{EXPORT_FOLDER}/{file_name_with_extension}' if(complete_path) else file_name
    with open(complete_file_path, 'rb') as file:
        loaded_data = pickle.load(file)
    return loaded_data

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

# Function to preprocess an image before feeding it to the model
def preprocess_image(image_path, img_size):
    img = keras.preprocessing.image.load_img(image_path, target_size=img_size)
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch

    # Preprocess the image (normalize pixel values)
    processed_img = keras.applications.mobilenet_v3.preprocess_input(img_array)
    return processed_img

# Function to predict the class of an image
def predict_class(image_path, model, class_names, img_size):
    processed_img = preprocess_image(image_path, img_size)
    predictions = model.predict(processed_img)
    predicted_class_index = tf.argmax(predictions[0]).numpy()
    return class_names[predicted_class_index]

# Load images and labels
def load_data(data_dir):
    image_paths = []
    labels = []
    class_names = sorted(os.listdir(data_dir))
    for label in class_names:
        class_dir = os.path.join(data_dir, label)
        if os.path.isdir(class_dir):
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                image_paths.append(img_path)
                labels.append(label)
    return np.array(image_paths), np.array(labels), class_names