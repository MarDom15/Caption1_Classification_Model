# 🌟 **Skin Lesion Analyzer**  
*Detection and classification of skin lesions using ResNet50, MLOps, and Flask.*

---

## 🗂️ **Table of Contents**  
1. [✨ Introduction](#introduction)  
2. [🔍 Data Exploration (EDA)](#data-exploration-eda)  
3. [🛠️ Data Preparation](#data-preparation)  
4. [🏋️‍♂️ Model Training](#model-training)  
5. [✅ Testing and Evaluation](#testing-and-evaluation)  
6. [🚀 Deployment with Flask](#deployment-with-flask)  
    - [📦 Creating a Docker Image](#creating-a-docker-image)  
    - [🌐 Server Deployment](#server-deployment)  
7. [📈 Monitoring and MLOps](#monitoring-and-mlops)  
    - [🔄 Building an MLOps Pipeline](#building-an-mlops-pipeline)  
8. [📂 Data](#data)  
9. [🙌 Contributors](#contributors)  

---

## ✨ **1. Introduction**  
The **Skin Lesion Analyzer** is an application that uses artificial intelligence to detect and classify skin lesions from images.  
This approach automates and improves the accuracy of dermatological diagnostics through deep learning algorithms.  

### Theoretical Context:
Image classification models rely on convolutional neural networks (CNNs), such as ResNet50. These models can detect complex patterns and differentiate lesion classes based on visual features.

🎯 **Main Objectives:**  
- Provide a fast and accurate model for analyzing skin lesions.  
- Deploy a user-friendly and accessible application.  
- Integrate a continuous training and deployment pipeline to ensure consistent performance improvements.

---

## 🔍 **2. Data Exploration (EDA)**  
Data Exploration is an essential step in data-driven projects. It helps understand the structure of the data and detect any anomalies or trends.  

### Key Steps:
1. Analyze the distribution of classes (types of lesions).  
2. Study metadata, such as age or anatomical location.  
3. Visualize correlations between different attributes.  

### Theoretical Context:
The quality and diversity of data directly influence model performance. An imbalanced class distribution may require techniques like sampling or weighting to ensure reliable predictions.  

📈 **Visualizations:**  
- Class distribution: Distribution of different lesion types:  
  ![Class Distribution](https://github.com/MarDom15/Caption1_Classification_Model/blob/main/prog_images/distrubution_Labels.png)  
- Age distribution: Analysis of age groups affected by each type of lesion:  
  ![Age Distribution](https://github.com/MarDom15/Caption1_Classification_Model/blob/main/prog_images/age_distribution.png)  
- Head Metadata: Analysis of metadata csv:
  ![Head Metadata](https://github.com/MarDom15/Caption1_Classification_Model/blob/main/prog_images/head_csv.png)

---

## 🛠️ **3. Data Preparation**  
Data preparation is crucial to ensure uniform and relevant input to the model.  

### Key Steps:
1. **Data Cleaning**: Remove missing or anomalous values in the metadata.  
2. **Image Preprocessing**: Resize images to 224x224 pixels to ensure compatibility with ResNet50.  
3. **Data Augmentation**: Generate variations of images (rotation, zoom, flipping) to enrich the dataset and prevent overfitting.  
4. **Normalization**: Adjust pixel values between 0 and 1 to stabilize training.  

### Theoretical Context:
Neural networks are sensitive to data scales. Normalization accelerates convergence and improves model robustness.

📷 **Examples of images:**  
![Example of a normalized image](https://github.com/MarDom15/Caption1_Classification_Model/blob/main/prog_images/image_exemple.png)  

---

## 🏋️‍♂️ **4. Model Training**  
The model used, **ResNet50**, is a convolutional neural network pretrained on ImageNet. It is fine-tuned to classify skin lesions into 7 categories.  

### Training Steps:
1. Load cleaned and normalized data.  
2. Perform data augmentation to improve robustness.  
3. Train by adjusting the model's final layers.  

### Theoretical Context:
- **Transfer Learning**: Using a pretrained model reduces data requirements and training time.  
- **Fine-Tuning**: Adapting a general-purpose model (like ResNet50) to a specific task.  

📈 **Visual Indicators:**  
- Accuracy curve: Evaluates performance on the validation set:  
  ![Accuracy Curve](https://github.com/MarDom15/Caption1_Classification_Model/blob/main/prog_images/accuracycurve.png)  
- Loss curve: Analyzes learning convergence:  
  ![Loss Curve](https://github.com/MarDom15/Caption1_Classification_Model/blob/main/prog_images/losscurve.png)  

---

## ✅ **5. Testing and Evaluation**  
Once trained, the model's performance is evaluated on an independent test set.  

### Metrics Used:
1. **Accuracy**: Proportion of correct predictions.  
2. **Recall**: Model's ability to correctly identify positive cases.  
3. **F1-score**: Harmonic mean of precision and recall, useful for imbalanced data.  
4. **Confusion Matrix**: Visualizes classification errors.  

📊 **Results and Confusion Matrix:**  
![Confusion Matrix](https://github.com/MarDom15/Caption1_Classification_Model/blob/main/prog_images/Labels_Numbers.png)  

### Theoretical Context:
Metrics like F1-score are particularly useful in medical contexts where classification errors have critical implications.  

---

## 🚀 **6. Deployment with Flask**  
Once validated, the model is integrated into a Flask application for interactive use.  

### 📦 **Creating a Docker Image**  
Docker is used to ensure application portability. A Docker image contains all necessary dependencies.  

**Dockerfile:**
```dockerfile
FROM python:3.9-slim

# Install dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

# Copy application files
COPY . /app
WORKDIR /app

# Command to start the Flask application
CMD ["python", "app.py"]
```

### 🌐 **Server Deployment**  
The Docker image is deployed on a server or cloud platform (AWS, Azure, Google Cloud).  

---

## 📈 **7. Monitoring and MLOps**  
Monitoring and MLOps practices ensure the model's maintenance and continuous improvement after deployment.  

### Monitoring with Prometheus and Grafana:
- Collect performance metrics (response time, error rate).  
- Real-time visualization on dashboards.  

### Building an MLOps Pipeline:
1. **CI/CD**: Continuous integration and deployment with GitHub Actions.  
2. **Kubernetes**: Container orchestration for scalable deployment.  

#### CI/CD Pipeline:
```yaml
name: CI/CD Pipeline

on: [push]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout code
        uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: 3.9

      - name: Install dependencies
        run: pip install -r requirements.txt

      - name: Run tests
        run: pytest

      - name: Build Docker image
        run: docker build -t skin-lesion-analyzer .
```

---

## 📂 **8. Data**  
The dataset **HAM10000** used in this project is a collection of 10,015 images of skin lesions, with detailed medical annotations.  

### Dataset Link:
- [HAM10000 on Kaggle](https://www.kaggle.com/kmader/skin-cancer-mnist-ham10000)  

---

## 🙌 **9. Contributors**  
This project was created by:  
- **[M. Domche]**  
For any questions or suggestions, contact me at [yemail@example.com](mailto:your-email@example.com).  

✨ Thank you for your interest in this project!  
