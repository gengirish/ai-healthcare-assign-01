"""
Model Training and Evaluation Module
Includes training functions, evaluation metrics, and Grad-CAM implementation
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
from itertools import cycle
import pandas as pd

class ModelTrainer:
    """
    Class for training and evaluating CNN models
    """
    
    def __init__(self, project):
        self.project = project
        self.class_weights = None
        
    def calculate_class_weights(self):
        """
        Calculate class weights to handle class imbalance
        """
        if hasattr(self.project, 'class_imbalance') and self.project.class_imbalance:
            # Get original labels for class weight calculation
            train_dataset = self.project.DataClass(split='train', download=False)
            train_labels = train_dataset.labels.flatten()
            
            # Calculate class weights
            classes = np.unique(train_labels)
            class_weights = compute_class_weight('balanced', classes=classes, y=train_labels)
            self.class_weights = dict(zip(classes, class_weights))
            
            print("Class weights calculated:")
            for class_idx, weight in self.class_weights.items():
                class_name = self.project.info['label'][str(class_idx)]
                print(f"Class {class_idx} ({class_name}): {weight:.3f}")
        else:
            self.class_weights = None
            print("No class weights needed - dataset is balanced")
    
    def create_data_augmentation(self):
        """
        Create data augmentation generator for training
        """
        datagen = ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            zoom_range=0.1,
            fill_mode='nearest'
        )
        return datagen
    
    def train_model(self, model, model_name, epochs=50, use_augmentation=True):
        """
        Train a model with callbacks and optional data augmentation
        """
        print(f"\n{'='*60}")
        print(f"TRAINING {model_name.upper()}")
        print(f"{'='*60}")
        
        # Calculate class weights if needed
        self.calculate_class_weights()
        
        # Define callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            ModelCheckpoint(
                f'{model_name.lower()}_best_model.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Training with or without data augmentation
        if use_augmentation and hasattr(self.project, 'class_imbalance') and self.project.class_imbalance:
            print("Training with data augmentation...")
            datagen = self.create_data_augmentation()
            datagen.fit(self.project.x_train)
            
            history = model.fit(
                datagen.flow(self.project.x_train, self.project.y_train, batch_size=32),
                steps_per_epoch=len(self.project.x_train) // 32,
                epochs=epochs,
                validation_data=(self.project.x_val, self.project.y_val),
                callbacks=callbacks,
                class_weight=self.class_weights,
                verbose=1
            )
        else:
            print("Training without data augmentation...")
            history = model.fit(
                self.project.x_train, self.project.y_train,
                batch_size=32,
                epochs=epochs,
                validation_data=(self.project.x_val, self.project.y_val),
                callbacks=callbacks,
                class_weight=self.class_weights,
                verbose=1
            )
        
        print(f"\n✅ {model_name} training completed!")
        return history
    
    def plot_training_history(self, history, model_name):
        """
        Plot training history (loss and accuracy)
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot training & validation accuracy
        ax1.plot(history.history['accuracy'], label='Training Accuracy')
        ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title(f'{model_name} - Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # Plot training & validation loss
        ax2.plot(history.history['loss'], label='Training Loss')
        ax2.plot(history.history['val_loss'], label='Validation Loss')
        ax2.set_title(f'{model_name} - Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(f'{model_name.lower()}_training_history.png', dpi=300, bbox_inches='tight')
        plt.show()

class ModelEvaluator:
    """
    Class for comprehensive model evaluation
    """
    
    def __init__(self, project):
        self.project = project
    
    def evaluate_model(self, model, model_name):
        """
        Comprehensive model evaluation with all required metrics
        """
        print(f"\n{'='*60}")
        print(f"EVALUATING {model_name.upper()}")
        print(f"{'='*60}")
        
        # Make predictions
        y_pred_proba = model.predict(self.project.x_test)
        y_pred = np.argmax(y_pred_proba, axis=1)
        y_true = np.argmax(self.project.y_test, axis=1)
        
        # Basic metrics
        test_loss, test_accuracy = model.evaluate(self.project.x_test, self.project.y_test, verbose=0)
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"Test Loss: {test_loss:.4f}")
        
        # Classification report
        class_names = [self.project.info['label'][str(i)] for i in range(self.project.n_classes)]
        report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
        
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=class_names))
        
        # Confusion Matrix
        self.plot_confusion_matrix(y_true, y_pred, class_names, model_name)
        
        # ROC Curves and AUC for multiclass
        self.plot_roc_curves(y_true, y_pred_proba, class_names, model_name)
        
        # Create metrics summary
        metrics_summary = {
            'Model': model_name,
            'Test Accuracy': test_accuracy,
            'Test Loss': test_loss,
            'Macro Avg Precision': report['macro avg']['precision'],
            'Macro Avg Recall': report['macro avg']['recall'],
            'Macro Avg F1-Score': report['macro avg']['f1-score'],
            'Weighted Avg Precision': report['weighted avg']['precision'],
            'Weighted Avg Recall': report['weighted avg']['recall'],
            'Weighted Avg F1-Score': report['weighted avg']['f1-score']
        }
        
        return metrics_summary, y_pred_proba
    
    def plot_confusion_matrix(self, y_true, y_pred, class_names, model_name):
        """
        Plot confusion matrix
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title(f'{model_name} - Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(f'{model_name.lower()}_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_roc_curves(self, y_true, y_pred_proba, class_names, model_name):
        """
        Plot ROC curves for multiclass classification
        """
        # Binarize the output
        y_true_bin = label_binarize(y_true, classes=range(self.project.n_classes))
        
        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        for i in range(self.project.n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_pred_proba.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        
        # Plot ROC curves
        plt.figure(figsize=(12, 8))
        
        # Plot ROC curve for each class
        colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'red', 'green'])
        for i, color in zip(range(self.project.n_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=2,
                    label=f'{class_names[i]} (AUC = {roc_auc[i]:.3f})')
        
        # Plot micro-average ROC curve
        plt.plot(fpr["micro"], tpr["micro"], color='deeppink', linestyle='--', lw=2,
                label=f'Micro-average (AUC = {roc_auc["micro"]:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'{model_name} - ROC Curves (Multi-class)')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'{model_name.lower()}_roc_curves.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print AUC scores
        print(f"\nAUC Scores for {model_name}:")
        for i in range(self.project.n_classes):
            print(f"{class_names[i]}: {roc_auc[i]:.4f}")
        print(f"Micro-average AUC: {roc_auc['micro']:.4f}")

class GradCAMVisualizer:
    """
    Class for Grad-CAM visualization and explainability
    """
    
    def __init__(self, project):
        self.project = project
    
    def make_gradcam_heatmap(self, img_array, model, last_conv_layer_name, pred_index=None):
        """
        Generate Grad-CAM heatmap
        """
        # Create a model that maps the input image to the activations of the last conv layer
        # as well as the output predictions
        grad_model = tf.keras.models.Model(
            [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
        )
        
        # Compute the gradient of the top predicted class for our input image
        # with respect to the activations of the last conv layer
        with tf.GradientTape() as tape:
            last_conv_layer_output, preds = grad_model(img_array)
            if pred_index is None:
                pred_index = tf.argmax(preds[0])
            class_channel = preds[:, pred_index]
        
        # Gradient of the output neuron (top predicted or chosen)
        # with regard to the output feature map of the last conv layer
        grads = tape.gradient(class_channel, last_conv_layer_output)
        
        # Vector of mean intensity of the gradient over a specific feature map channel
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Multiply each channel in the feature map array
        # by "how important this channel is" with regard to the top predicted class
        last_conv_layer_output = last_conv_layer_output[0]
        heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        
        # Normalize the heatmap between 0 & 1 for visualization
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        return heatmap.numpy()
    
    def find_last_conv_layer(self, model):
        """
        Find the last convolutional layer in the model
        """
        for layer in reversed(model.layers):
            if len(layer.output_shape) == 4:  # Conv2D layers have 4D output
                return layer.name
        return None
    
    def visualize_gradcam(self, model, model_name, num_samples=8):
        """
        Visualize Grad-CAM for sample images
        """
        print(f"\n{'='*60}")
        print(f"GRAD-CAM VISUALIZATION FOR {model_name.upper()}")
        print(f"{'='*60}")
        
        # Find the last convolutional layer
        last_conv_layer_name = self.find_last_conv_layer(model)
        if last_conv_layer_name is None:
            print("No convolutional layer found in the model!")
            return
        
        print(f"Using last conv layer: {last_conv_layer_name}")
        
        # Select random samples from test set
        indices = np.random.choice(len(self.project.x_test), num_samples, replace=False)
        
        fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4*num_samples))
        if num_samples == 1:
            axes = axes.reshape(1, -1)
        
        class_names = [self.project.info['label'][str(i)] for i in range(self.project.n_classes)]
        
        for idx, test_idx in enumerate(indices):
            # Get image and true label
            img = self.project.x_test[test_idx:test_idx+1]
            true_label = np.argmax(self.project.y_test[test_idx])
            true_class = class_names[true_label]
            
            # Make prediction
            preds = model.predict(img, verbose=0)
            pred_label = np.argmax(preds[0])
            pred_class = class_names[pred_label]
            confidence = preds[0][pred_label]
            
            # Generate Grad-CAM heatmap
            heatmap = self.make_gradcam_heatmap(img, model, last_conv_layer_name)
            
            # Original image
            original_img = img[0]
            if self.project.n_channels == 1:
                original_img = original_img.squeeze()
                axes[idx, 0].imshow(original_img, cmap='gray')
            else:
                axes[idx, 0].imshow(original_img)
            axes[idx, 0].set_title(f'Original\nTrue: {true_class}')
            axes[idx, 0].axis('off')
            
            # Heatmap
            axes[idx, 1].imshow(heatmap, cmap='jet')
            axes[idx, 1].set_title('Grad-CAM Heatmap')
            axes[idx, 1].axis('off')
            
            # Superimposed
            if self.project.n_channels == 1:
                # Convert grayscale to RGB for superimposition
                img_rgb = cv2.cvtColor((original_img * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
            else:
                img_rgb = (original_img * 255).astype(np.uint8)
            
            # Resize heatmap to match image size
            heatmap_resized = cv2.resize(heatmap, (img_rgb.shape[1], img_rgb.shape[0]))
            heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
            
            # Superimpose
            superimposed = heatmap_colored * 0.4 + img_rgb * 0.6
            axes[idx, 2].imshow(superimposed.astype(np.uint8))
            axes[idx, 2].set_title('Superimposed')
            axes[idx, 2].axis('off')
            
            # Prediction info
            axes[idx, 3].text(0.1, 0.8, f'Predicted: {pred_class}', fontsize=12, transform=axes[idx, 3].transAxes)
            axes[idx, 3].text(0.1, 0.6, f'Confidence: {confidence:.3f}', fontsize=12, transform=axes[idx, 3].transAxes)
            axes[idx, 3].text(0.1, 0.4, f'True: {true_class}', fontsize=12, transform=axes[idx, 3].transAxes)
            
            # Color code: green for correct, red for incorrect
            color = 'green' if pred_label == true_label else 'red'
            axes[idx, 3].text(0.1, 0.2, '✓ Correct' if pred_label == true_label else '✗ Incorrect', 
                            fontsize=12, color=color, transform=axes[idx, 3].transAxes)
            axes[idx, 3].set_title('Prediction Info')
            axes[idx, 3].axis('off')
        
        plt.tight_layout()
        plt.savefig(f'{model_name.lower()}_gradcam_visualization.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def analyze_model_focus(self, model, model_name):
        """
        Analyze what regions the model focuses on for each class
        """
        print(f"\nAnalyzing model focus patterns for {model_name}...")
        
        # Find the last convolutional layer
        last_conv_layer_name = self.find_last_conv_layer(model)
        if last_conv_layer_name is None:
            return
        
        class_names = [self.project.info['label'][str(i)] for i in range(self.project.n_classes)]
        
        # For each class, find correctly predicted samples and analyze focus
        for class_idx in range(self.project.n_classes):
            # Find correctly predicted samples for this class
            y_pred = model.predict(self.project.x_test, verbose=0)
            y_pred_labels = np.argmax(y_pred, axis=1)
            y_true_labels = np.argmax(self.project.y_test, axis=1)
            
            correct_indices = np.where((y_true_labels == class_idx) & (y_pred_labels == class_idx))[0]
            
            if len(correct_indices) > 0:
                # Sample a few correct predictions
                sample_indices = correct_indices[:min(3, len(correct_indices))]
                
                print(f"\nClass {class_idx} ({class_names[class_idx]}): {len(correct_indices)} correct predictions")
                
                # Generate average heatmap for this class
                heatmaps = []
                for idx in sample_indices:
                    img = self.project.x_test[idx:idx+1]
                    heatmap = self.make_gradcam_heatmap(img, model, last_conv_layer_name, class_idx)
                    heatmaps.append(heatmap)
                
                if heatmaps:
                    avg_heatmap = np.mean(heatmaps, axis=0)
                    
                    # Find the region with highest activation
                    max_activation = np.max(avg_heatmap)
                    max_location = np.unravel_index(np.argmax(avg_heatmap), avg_heatmap.shape)
                    
                    print(f"  - Max activation: {max_activation:.3f} at position {max_location}")
                    print(f"  - Focus area: {'Center' if abs(max_location[0] - avg_heatmap.shape[0]//2) < avg_heatmap.shape[0]//4 else 'Edge'}")

def compare_models(metrics1, metrics2):
    """
    Compare two models and create comparison visualization
    """
    print(f"\n{'='*60}")
    print("MODEL COMPARISON")
    print(f"{'='*60}")
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame([metrics1, metrics2])
    
    print("\nMetrics Comparison:")
    print(comparison_df.to_string(index=False))
    
    # Visualize comparison
    metrics_to_plot = ['Test Accuracy', 'Macro Avg Precision', 'Macro Avg Recall', 'Macro Avg F1-Score']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.ravel()
    
    for i, metric in enumerate(metrics_to_plot):
        values = [metrics1[metric], metrics2[metric]]
        models = [metrics1['Model'], metrics2['Model']]
        
        bars = axes[i].bar(models, values, color=['skyblue', 'lightcoral'])
        axes[i].set_title(f'{metric} Comparison')
        axes[i].set_ylabel(metric)
        axes[i].set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{value:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Determine better model
    better_model = metrics1['Model'] if metrics1['Test Accuracy'] > metrics2['Test Accuracy'] else metrics2['Model']
    accuracy_diff = abs(metrics1['Test Accuracy'] - metrics2['Test Accuracy'])
    
    print(f"\n🏆 Better performing model: {better_model}")
    print(f"📊 Accuracy difference: {accuracy_diff:.4f}")
    
    if accuracy_diff < 0.01:
        print("📝 Note: The models have very similar performance")
    
    return comparison_df
