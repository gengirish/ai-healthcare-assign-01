"""
Deep Learning for Optical Coherence Tomography (OCT) Image Classification
IIIT Dharwad - AI in Healthcare Case Study Assignment 1

This project implements CNN models for classifying OCT images from the OCTMNIST dataset
into multiple retinal disease categories with explainability using Grad-CAM.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import label_binarize
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import ResNet50, VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import medmnist
from medmnist import INFO, Evaluator
import cv2
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class OCTClassificationProject:
    """
    Main class for OCT image classification project
    """
    
    def __init__(self):
        self.data_flag = 'octmnist'
        self.download = True
        self.info = INFO[self.data_flag]
        self.task = self.info['task']
        self.n_channels = self.info['n_channels']
        self.n_classes = len(self.info['label'])
        self.DataClass = getattr(medmnist, self.info['python_class'])
        
        # Initialize data containers
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
        # Model containers
        self.model1 = None  # Custom CNN
        self.model2 = None  # Pretrained model
        
        # Training history
        self.history1 = None
        self.history2 = None
        
        print(f"Dataset: {self.data_flag}")
        print(f"Task: {self.task}")
        print(f"Number of channels: {self.n_channels}")
        print(f"Number of classes: {self.n_classes}")
        print(f"Class labels: {self.info['label']}")
    
    def load_and_preprocess_data(self):
        """
        Load and preprocess the OCTMNIST dataset
        """
        print("Loading OCTMNIST dataset...")
        
        # Load datasets
        train_dataset = self.DataClass(split='train', download=self.download)
        val_dataset = self.DataClass(split='val', download=self.download)
        test_dataset = self.DataClass(split='test', download=self.download)
        
        # Extract data and labels
        x_train, y_train = train_dataset.imgs, train_dataset.labels
        x_val, y_val = val_dataset.imgs, val_dataset.labels
        x_test, y_test = test_dataset.imgs, test_dataset.labels
        
        # Normalize pixel values to [0, 1]
        x_train = x_train.astype('float32') / 255.0
        x_val = x_val.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0
        
        # Convert labels to categorical if multiclass
        if self.task == 'multi-class':
            y_train = keras.utils.to_categorical(y_train, self.n_classes)
            y_val = keras.utils.to_categorical(y_val, self.n_classes)
            y_test = keras.utils.to_categorical(y_test, self.n_classes)
        
        self.x_train, self.y_train = x_train, y_train
        self.x_val, self.y_val = x_val, y_val
        self.x_test, self.y_test = x_test, y_test
        
        print(f"Training set shape: {x_train.shape}")
        print(f"Validation set shape: {x_val.shape}")
        print(f"Test set shape: {x_test.shape}")
        
        # Analyze class distribution
        self.analyze_class_distribution()
        
        return x_train, y_train, x_val, y_val, x_test, y_test
    
    def analyze_class_distribution(self):
        """
        Analyze and visualize class distribution in the dataset
        """
        # Get original labels for analysis
        train_dataset = self.DataClass(split='train', download=False)
        train_labels = train_dataset.labels.flatten()
        
        # Count class distribution
        unique, counts = np.unique(train_labels, return_counts=True)
        class_distribution = dict(zip(unique, counts))
        
        print("\nClass Distribution in Training Set:")
        for class_idx, count in class_distribution.items():
            class_name = self.info['label'][str(class_idx)]
            print(f"Class {class_idx} ({class_name}): {count} samples")
        
        # Visualize class distribution
        plt.figure(figsize=(10, 6))
        class_names = [self.info['label'][str(i)] for i in unique]
        plt.bar(class_names, counts)
        plt.title('Class Distribution in OCTMNIST Training Set')
        plt.xlabel('Disease Categories')
        plt.ylabel('Number of Samples')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('class_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Check for class imbalance
        max_count = max(counts)
        min_count = min(counts)
        imbalance_ratio = max_count / min_count
        
        print(f"\nClass Imbalance Analysis:")
        print(f"Maximum class samples: {max_count}")
        print(f"Minimum class samples: {min_count}")
        print(f"Imbalance ratio: {imbalance_ratio:.2f}")
        
        if imbalance_ratio > 2.0:
            print("⚠️  Significant class imbalance detected. Consider using class weights or data augmentation.")
            self.class_imbalance = True
        else:
            print("✅ Class distribution is relatively balanced.")
            self.class_imbalance = False
    
    def visualize_sample_images(self, num_samples=16):
        """
        Visualize sample images from each class
        """
        train_dataset = self.DataClass(split='train', download=False)
        
        fig, axes = plt.subplots(4, 4, figsize=(12, 12))
        axes = axes.ravel()
        
        for i in range(min(num_samples, len(axes))):
            img = train_dataset.imgs[i]
            label = train_dataset.labels[i].item()
            class_name = self.info['label'][str(label)]
            
            if self.n_channels == 1:
                axes[i].imshow(img.squeeze(), cmap='gray')
            else:
                axes[i].imshow(img)
            
            axes[i].set_title(f'Class {label}: {class_name}')
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig('sample_images.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_custom_cnn(self):
        """
        Create a custom CNN architecture for OCT image classification
        """
        print("Creating Custom CNN Model...")
        
        model = models.Sequential([
            # First Convolutional Block
            layers.Conv2D(32, (3, 3), activation='relu', 
                         input_shape=(*self.x_train.shape[1:3], self.n_channels)),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Second Convolutional Block
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Third Convolutional Block
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Fourth Convolutional Block
            layers.Conv2D(256, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.5),
            
            # Dense layers
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.3),
            
            # Output layer
            layers.Dense(self.n_classes, activation='softmax' if self.task == 'multi-class' else 'sigmoid')
        ])
        
        self.model1 = model
        print("Custom CNN Model created successfully!")
        return model
    
    def create_pretrained_model(self, base_model_name='ResNet50'):
        """
        Create a pretrained model (ResNet50 or VGG16) for transfer learning
        """
        print(f"Creating Pretrained Model: {base_model_name}...")
        
        # Input shape adjustment for grayscale images
        if self.n_channels == 1:
            input_tensor = layers.Input(shape=(*self.x_train.shape[1:3], 1))
            # Convert grayscale to RGB for pretrained models
            x = layers.Conv2D(3, (1, 1), activation='linear')(input_tensor)
        else:
            input_tensor = layers.Input(shape=(*self.x_train.shape[1:3], self.n_channels))
            x = input_tensor
        
        # Load pretrained base model
        if base_model_name == 'ResNet50':
            base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=x)
        elif base_model_name == 'VGG16':
            base_model = VGG16(weights='imagenet', include_top=False, input_tensor=x)
        else:
            raise ValueError("Supported models: 'ResNet50', 'VGG16'")
        
        # Freeze base model layers initially
        base_model.trainable = False
        
        # Add custom classification head
        x = base_model.output
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        
        # Output layer
        predictions = layers.Dense(self.n_classes, 
                                 activation='softmax' if self.task == 'multi-class' else 'sigmoid')(x)
        
        model = models.Model(inputs=input_tensor, outputs=predictions)
        self.model2 = model
        
        print(f"Pretrained {base_model_name} Model created successfully!")
        return model
    
    def compile_models(self):
        """
        Compile both models with appropriate loss functions and optimizers
        """
        print("Compiling models...")
        
        # Determine loss function and metrics based on task
        if self.task == 'multi-class':
            loss = 'categorical_crossentropy'
            metrics = ['accuracy']
        else:
            loss = 'binary_crossentropy'
            metrics = ['accuracy']
        
        # Compile Custom CNN
        self.model1.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss=loss,
            metrics=metrics
        )
        
        # Compile Pretrained Model
        self.model2.compile(
            optimizer=optimizers.Adam(learning_rate=0.0001),  # Lower learning rate for pretrained
            loss=loss,
            metrics=metrics
        )
        
        print("Models compiled successfully!")
        
        # Print model summaries
        print("\n" + "="*50)
        print("CUSTOM CNN MODEL SUMMARY")
        print("="*50)
        self.model1.summary()
        
        print("\n" + "="*50)
        print("PRETRAINED MODEL SUMMARY")
        print("="*50)
        self.model2.summary()

if __name__ == "__main__":
    # Initialize project
    project = OCTClassificationProject()
    
    # Load and preprocess data
    x_train, y_train, x_val, y_val, x_test, y_test = project.load_and_preprocess_data()
    
    # Visualize sample images
    project.visualize_sample_images()
    
    # Create models
    custom_model = project.create_custom_cnn()
    pretrained_model = project.create_pretrained_model('ResNet50')
    
    # Compile models
    project.compile_models()
    
    print("\n🎉 Project setup completed successfully!")
    print("Next steps: Train models, evaluate performance, and implement Grad-CAM")
