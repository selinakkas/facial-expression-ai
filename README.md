# Facial Expression Recognition (FER2013)

This project aims to detect facial expressions using both a **custom-built CNN model** and **transfer learning (MobileNetV2)**.

## 📁 Dataset
- Dataset used: [FER-2013 from Kaggle](https://www.kaggle.com/datasets/msambare/fer2013)
- 7 emotion classes: `angry`, `disgust`, `fear`, `happy`, `neutral`, `sad`, `surprise`

## 🧠 Models
### 1. Transfer Learning
- Pre-trained model: **MobileNetV2**
- Top layers customized for 7 emotion classes
- Some layers frozen (feature extraction), others fine-tuned

### 2. Custom CNN Model
- Multiple `Conv2D`, `MaxPooling`, and `Dense` layers
- Trained from scratch on the same dataset

## 📊 Performance
- Accuracy, loss, confusion matrix, and F1-scores reported
- Training & validation curves provided
- Sample predictions shown in the final report

## 📝 Team Members
- Selin Akkas
- Hayrun Nisa Celik
- Yasemin Ozer

## 📅 Deadline & Submission
- Report: Submitted on 11 May 2025
- Presentation: 12 May 2025, Classroom 1108
