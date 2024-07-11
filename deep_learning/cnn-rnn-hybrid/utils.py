from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report, matthews_corrcoef, cohen_kappa_score, hamming_loss
from tensorflow.keras.utils import to_categorical
import numpy as np


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
        print("Classification Report:\n", results['class_report'], end="\n\n")
        

    
# Calculate statistics
def calculate_statistics(data):
    return {
        'max': np.max(data),
        'min': np.min(data),
        'median': np.median(data),
        'mean': np.mean(data)
    }