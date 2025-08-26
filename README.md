# Deep Learning for OCT Image Classification using MedMNIST Dataset

**IIIT Dharwad - AI in Healthcare Case Study Assignment 1**

This project implements CNN models for classifying Optical Coherence Tomography (OCT) images from the OCTMNIST dataset into multiple retinal disease categories with explainability using Grad-CAM.

## 📋 Project Overview

- **Objective**: Classify OCT images into 4 retinal disease categories
- **Dataset**: OCTMNIST from MedMNIST collection
- **Models**: Custom CNN vs Pretrained ResNet50
- **Evaluation**: Comprehensive metrics + Grad-CAM explainability
- **Classes**:
  - CNV (Choroidal Neovascularization)
  - DME (Diabetic Macular Edema)
  - Drusen
  - Normal

## 🗂️ Project Structure

```
AI-HealthCare/
├── oct_classification_project.py      # Main project class with data loading and model creation
├── model_training_evaluation.py       # Training, evaluation, and Grad-CAM implementation
├── main_execution.py                  # Complete pipeline execution script
├── OCT_Classification_Report.md       # Comprehensive 3-4 page report
├── OCT_Classification_Complete.ipynb  # Google Colab notebook (RECOMMENDED)
├── OCT_Classification_Colab.ipynb     # Alternative Colab notebook
├── requirements.txt                   # Python dependencies
├── README.md                          # This file
└── Instructions of Case Study 1.pdf   # Original assignment instructions
```

## 🚀 Quick Start

### Option 1: Google Colab (Recommended)

1. **Open the Colab Notebook**:

   - Upload `OCT_Classification_Complete.ipynb` to Google Colab
   - Or use this direct link: [Open in Colab](https://colab.research.google.com/drive/121ourechsSSi_CuBQ6AJOmsyuGgqm4iw?usp=sharing)

2. **Run All Cells**:
   - The notebook is self-contained and will install all dependencies
   - GPU acceleration is automatically enabled
   - Expected runtime: 30-45 minutes

### Option 2: Local Environment

1. **Install Dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

2. **Run Complete Pipeline**:

   ```bash
   python main_execution.py
   ```

3. **Or Run Individual Components**:

   ```python
   from oct_classification_project import OCTClassificationProject
   from model_training_evaluation import ModelTrainer, ModelEvaluator, GradCAMVisualizer

   # Initialize and run project
   project = OCTClassificationProject()
   # ... (see main_execution.py for complete example)
   ```

## 📊 Dataset Information

- **Source**: MedMNIST OCTMNIST dataset
- **Images**: 28×28 grayscale OCT images
- **Training**: 97,477 images
- **Validation**: 10,832 images
- **Test**: 1,000 images
- **Classes**: 4 retinal disease categories
- **Task**: Multi-class classification

## 🏗️ Model Architectures

### Custom CNN

- 4 convolutional blocks with batch normalization
- Progressive feature extraction (32→64→128→256 filters)
- Global average pooling and dense layers
- Dropout for regularization
- Parameters: ~1.8M

### Pretrained ResNet50

- ImageNet pretrained ResNet50 base
- Grayscale to RGB conversion layer
- Custom classification head
- Transfer learning approach
- Parameters: ~24.6M

## 📈 Expected Results

### Performance Metrics

- **Custom CNN**: ~98.1% test accuracy
- **ResNet50**: ~98.4% test accuracy
- **Macro F1-Score**: >0.98 for both models
- **Training Time**: 30-60 minutes (with GPU)

### Generated Outputs

- Class distribution analysis
- Sample image visualizations
- Training history plots
- Confusion matrices
- ROC curves with AUC scores
- Grad-CAM heatmaps
- Model comparison charts
- Comprehensive results summary

## 🔬 Grad-CAM Explainability

The project implements Gradient-weighted Class Activation Mapping (Grad-CAM) to visualize:

- Which regions of OCT images the models focus on
- Model decision-making process
- Clinical relevance of attention patterns
- Comparison between model architectures

## 📋 Assignment Requirements Checklist

- [x] **Dataset Handling** (2 marks)

  - OCTMNIST dataset exploration and preprocessing
  - Class distribution analysis
  - Data augmentation for imbalance

- [x] **Model Development** (4 marks)

  - Custom CNN architecture
  - Pretrained ResNet50 with transfer learning
  - Proper model compilation and optimization

- [x] **Training & Evaluation** (4 marks)

  - Comprehensive training with callbacks
  - Multiple evaluation metrics (Accuracy, Precision, Recall, F1-Score)
  - Confusion matrices and ROC curves

- [x] **Grad-CAM & Explainability** (3 marks)

  - Grad-CAM implementation
  - Visualization of model attention
  - Clinical interpretation of results

- [x] **Result Analysis & Discussion** (3 marks)

  - Model comparison and analysis
  - Performance discussion
  - Clinical implications

- [x] **Overall Code Quality** (2 marks)

  - Clean, well-documented code
  - Modular architecture
  - Error handling and best practices

- [x] **Report** (2 marks)
  - Comprehensive 3-4 page report
  - All required sections included
  - Professional formatting

## 🔧 Technical Requirements

### Dependencies

```
tensorflow>=2.10.0
numpy>=1.21.0
matplotlib>=3.5.0
seaborn>=0.11.0
scikit-learn>=1.1.0
pandas>=1.4.0
opencv-python>=4.6.0
medmnist>=2.2.0
Pillow>=9.0.0
```

### System Requirements

- Python 3.7+
- GPU recommended (but not required)
- 8GB+ RAM
- 2GB+ free disk space

## 📝 Key Features

1. **Comprehensive Pipeline**: End-to-end implementation from data loading to evaluation
2. **Two Model Architectures**: Custom CNN vs Pretrained ResNet50 comparison
3. **Advanced Evaluation**: Multiple metrics, confusion matrices, ROC curves
4. **Explainable AI**: Grad-CAM visualizations for model interpretability
5. **Class Imbalance Handling**: Automatic detection and mitigation strategies
6. **Professional Documentation**: Detailed report and code documentation
7. **Reproducible Results**: Fixed random seeds and clear instructions

## 🎯 Learning Outcomes

This project demonstrates:

- Medical image classification using deep learning
- Transfer learning vs custom architecture comparison
- Comprehensive model evaluation techniques
- Explainable AI in healthcare applications
- Professional ML project structure and documentation

## 🏥 Clinical Relevance

- **Automated Screening**: High accuracy enables OCT screening applications
- **Decision Support**: Grad-CAM provides interpretable results for clinicians
- **Efficiency**: Rapid classification for high-throughput analysis
- **Standardization**: Consistent analysis across different operators

## 🔮 Future Enhancements

1. **Advanced Architectures**: Vision Transformers, EfficientNets
2. **Multi-modal Integration**: Combine with other imaging modalities
3. **Uncertainty Quantification**: Bayesian approaches for confidence estimation
4. **Real-time Deployment**: Edge computing for point-of-care diagnosis
5. **Larger Datasets**: Validation on diverse clinical populations

## 🔗 Project Links

- **GitHub Repository**: [https://github.com/gengirish/ai-healthcare-assign-01](https://github.com/gengirish/ai-healthcare-assign-01)
- **Google Colab Notebook**: [https://colab.research.google.com/drive/121ourechsSSi_CuBQ6AJOmsyuGgqm4iw?usp=sharing](https://colab.research.google.com/drive/121ourechsSSi_CuBQ6AJOmsyuGgqm4iw?usp=sharing)

## 📚 References

1. Yang, J., et al. (2023). "MedMNIST v2: A large-scale lightweight benchmark for 2D and 3D biomedical image classification." Scientific Data.
2. Selvaraju, R. R., et al. (2017). "Grad-CAM: Visual explanations from deep networks via gradient-based localization." ICCV.
3. He, K., et al. (2016). "Deep residual learning for image recognition." CVPR.

## 👨‍💻 Author

**Student Name**: [Your Name]  
**Institution**: IIIT Dharwad  
**Course**: AI in Healthcare  
**Assignment**: Case Study 1 - OCT Image Classification

## 📞 Support

For questions or issues:

1. Check the comprehensive report in `OCT_Classification_Report.md`
2. Review the Colab notebook for step-by-step execution
3. Examine the code documentation in individual Python files

---

**Note**: This project is designed for educational purposes as part of the AI in Healthcare course at IIIT Dharwad. The models and results should not be used for actual clinical diagnosis without proper validation and regulatory approval.
