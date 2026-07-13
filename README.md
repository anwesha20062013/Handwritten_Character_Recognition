# 🧠 AI Handwritten Character & Digit Recognition using CNN

## 📌 Overview

This project is an AI-powered handwritten recognition system developed using **TensorFlow**, **OpenCV**, and **Streamlit**. It can recognize handwritten **digits** as well as **characters** drawn by the user on an interactive canvas.

The application uses Convolutional Neural Networks (CNNs) trained on the **MNIST** and **EMNIST Balanced** datasets.

---

## 🚀 Features

- 🔢 Handwritten Digit Recognition
- 🔤 Handwritten Character Recognition
- ✍️ Interactive Drawing Canvas
- 🧠 CNN-based Prediction
- 📊 Confidence Score
- 🎨 Streamlit Web Interface
- ⚡ Real-time Prediction

---

## 🛠 Technologies Used

- Python
- TensorFlow / Keras
- OpenCV
- NumPy
- Pandas
- Streamlit
- streamlit-drawable-canvas

---

## 📂 Dataset

### MNIST
Used for handwritten digit recognition.

### EMNIST Balanced
Used for handwritten character recognition.

---

## 🧠 Model Architecture

The project uses a Convolutional Neural Network (CNN) consisting of:

- Convolution Layers
- Batch Normalization
- Max Pooling
- Dropout
- Fully Connected Dense Layers
- Softmax Output Layer

---

## 📊 Model Performance

| Model | Dataset | Accuracy |
|-------|---------|----------|
| Digit Recognition | MNIST | ~98–99% |
| Character Recognition | EMNIST Balanced | 82.28% |

---

## 📁 Project Structure

```text
Handwritten_Character_Recognition/

├── app.py
├── predictor.py
├── preprocess.py
├── segment.py
├── character_mapping.py
├── train_digit_model.py
├── train_character_model.py
├── models/
│   ├── handwritten_cnn_model.h5
│   └── character_model.h5
├── README.md
└── requirements.txt
```

---

## ▶️ Installation

Clone the repository:

```bash
git clone <your-github-link>
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

---

## 📷 Application

The user can:

- Select Digit or Character Recognition
- Draw on the canvas
- Click **Predict**
- View the prediction and confidence score

---

## 👩‍💻 Developed By

**Anwesha Barik**

B.Tech CSE (AI & ML)

---

## ⭐ Future Improvements

- Upload image support
- Top-3 predictions
- Dark mode
- Model optimization
- Deployment using Streamlit Cloud