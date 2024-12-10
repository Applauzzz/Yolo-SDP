import os
import torch
from pathlib import Path
from PIL import Image
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import csv
import json

Class_labels = {0: 'Nonmine', 1: 'abandon_Mining-Building', 2: 'abandon_Mining-Costean_Trench', 3: 'abandon_Mining-Dam', 4: 'abandon_Mining-Rubbish Dump', 5: 'abandon_Mining-Shaft', 6: 'activate_Mining-Building', 7: 'activate_Mining-Costean_Trench', 8: 'activate_Mining-Dam', 9: 'activate_Mining-Rubbish Dump', 10: 'activate_Mining-Shaft'}
Class_labels_index = { 'Nonmine': 0, 'abandon_Mining-Building': 1, 'abandon_Mining-Costean_Trench': 2, 'abandon_Mining-Dam': 3, 'abandon_Mining-Rubbish Dump': 4, 'abandon_Mining-Shaft': 5, 'activate_Mining-Building': 6, 'activate_Mining-Costean_Trench': 7, 'activate_Mining-Dam': 8, 'activate_Mining-Rubbish Dump': 9, 'activate_Mining-Shaft': 10}

def evaluate_model_on_subsets(model, validation_sets_dir: str, stream=False, save_to_file=False):
    """
    Evaluate the model on multiple sub-datasets (subclasses) and calculate the accuracy.
    
    Args:
        model: The YOLO model or any other model with the `predict()` method.
        validation_sets_dir (str): Directory containing sub-folders for each subclass.
        stream (bool): Whether to stream predictions or not.
        save_to_file (bool): Whether to save the results to a file.
    
    Returns:
        dict: A dictionary with the accuracy (precision, recall, F1 score) for each subclass.
    """
    results_per_class = {}
    
    # Walk through the validation dataset directory and process each subclass
    for subfolder in os.listdir(validation_sets_dir):
        subfolder_path = Path(validation_sets_dir) / subfolder
        if os.path.isdir(subfolder_path):
            print(f"Evaluating {subfolder}...")
            
            true_labels = []
            predictions = []
            
            # Loop through each image in the current subclass folder
            for img_name in os.listdir(subfolder_path):
                img_path = subfolder_path / img_name
                if img_path.suffix.lower() in [".jpg", ".png", ".jpeg"]:
                    # Load image
                    image = Image.open(img_path)

                    # Perform prediction
                    result = model.predict(source=image, stream=stream)[0].probs.top1
                    
                    # Use the subfolder name as the true class
                    true_class = Class_labels_index[subfolder]  # Assuming folder name is the class label
                    print(result, true_class)
                    # exit()
                    predictions.append(result)
                    true_labels.append(true_class)
            
            # Calculate the evaluation metrics
            if len(true_labels) > 0 and len(predictions) > 0:
                precision = precision_score(true_labels, predictions, average='macro', zero_division=0)
                recall = recall_score(true_labels, predictions, average='macro', zero_division=0)
                f1 = f1_score(true_labels, predictions, average='macro', zero_division=0)
                accuracy = accuracy_score(true_labels, predictions)  # Add accuracy calculation
            else:
                precision = recall = f1 = accuracy = 0  # If no predictions, set scores to 0
            
            # Store the results
            results_per_class[subfolder] = {
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'accuracy': accuracy  # Store accuracy
            }
            print(f"Precision: {precision}, Recall: {recall}, F1 Score: {f1}, Accuracy: {accuracy}")
    
    # Save results to file (CSV or JSON)
    if save_to_file:
        save_results(results_per_class)

    return results_per_class

def save_results(results, output_file='evaluation_results.json'):
    """
    Save the evaluation results to a JSON file.
    
    Args:
        results (dict): The evaluation results to save.
        output_file (str): The file path to save the results.
    """
    with open(output_file, 'w') as json_file:
        json.dump(results, json_file, indent=4)
    print(f"Results saved to {output_file}")

    # Optionally save as CSV
    with open('evaluation_results.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Class', 'Precision', 'Recall', 'F1 Score', 'Accuracy'])
        for class_name, metrics in results.items():
            writer.writerow([class_name, metrics['precision'], metrics['recall'], metrics['f1_score'], metrics['accuracy']])
    print("Results saved as evaluation_results.csv")

from ultralytics import YOLO

# Load a model
model = YOLO('/mnt/nvme_storage/Zehao/ultralytics_improve - SPD-Conv/plot-save/yolo-SPD-0.025/weights/best.pt')
# model = YOLO('/mnt/nvme_storage/Zehao/ultralytics_improve - SPD-Conv/ultralytics/cfg/models/v8/yolov8-cls.yaml')
# model.load('/mnt/nvme_storage/Zehao/ultralytics_improve - SPD-Conv/plot-save/yolo-SPD-0.025/weights/best.pt') # loading pretrain weights

# Example usage:
validation_sets_dir = '/mnt/nvme_storage/database/Nonmine_site/val-v2'  # Directory containing the sub-classes as folders
results = evaluate_model_on_subsets(model, validation_sets_dir, save_to_file=True)
print("Evaluation results per class:", results)
