# Email Spam Detection with Deep Learning

## 📝 Project Description

This project focuses on **Email Spam Detection**, which involves identifying and filtering out unsolicited, irrelevant, or harmful email messages (commonly known as spam) before they reach a user's inbox. The process uses machine learning algorithms, heuristic rules, and various filters to analyze email content, sender information, and metadata to classify emails as either spam or legitimate (ham).

## 🎯 Project Purpose

The primary goal is to develop an email spam detection system using an LSTM (Long Short-Term Memory) model to accurately classify emails as either spam or ham.

## 📂 Dataset

For this project, we utilize a custom dataset created specifically for training and evaluating the spam detection model.

## 🛠️ Libraries Used

- **tensorflow:** An open-source deep learning library used for building and training the LSTM model.
- **scikit-learn:** An open-source machine learning library used for various tasks such as data splitting, preprocessing, and model evaluation.
- **numpy (Numerical Python):** A library that provides support for multi-dimensional arrays and matrices, along with a collection of mathematical functions to operate on these arrays efficiently.
- **pandas:** A powerful data analysis and manipulation library used for handling and preprocessing the email dataset.
- **matplotlib:** A data visualization library used for creating plots and charts to analyze the model's performance.

## ⚙️ Project Steps

1. **Dataset Creation:**  
   - Create a custom dataset containing both spam and ham emails.

2. **Data Preprocessing:**  
   - Load the dataset and preprocess the email text data.
   - Clean the text data by removing irrelevant characters, stop words, and HTML tags.
   - Tokenize the text data and convert it into numerical sequences.
   - Pad the sequences to ensure uniform length for LSTM input.

3. **LSTM Model Creation:**  
   - Design and build an LSTM-based neural network architecture for spam detection.
   - Configure the model with appropriate layers, activation functions, and regularization techniques.

4. **Model Training:**  
   - Train the LSTM model using the preprocessed dataset.
   - Split the dataset into training and validation sets.
   - Monitor the model's performance during training and adjust hyperparameters as needed.

5. **Model Visualization and Evaluation:**  
   - Evaluate the trained model on a test dataset.
   - Visualize the model's performance using metrics such as accuracy, precision, recall, and F1-score.
   - Plot confusion matrices and ROC curves to analyze the model's classification performance.

## 🚀 Getting Started

1. **Clone the Repository**
    ```bash
    git clone https://github.com/elifkeskin/Email-Spam-Detection.git
    cd Email-Spam-Detection
    ```

2. **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3. **Run the Project**
    - Execute the main script or notebook to preprocess the data, build, train, and evaluate the LSTM model.

## 📊 Example Code Snippet

   ```bash
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

# Example: Creating a simple LSTM model
model = Sequential([
    Embedding(input_dim=10000, output_dim=128, input_length=100),
    LSTM(128),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
  ```

🤝 Contributing
Contributions are welcome!

1.**Fork the repository**

2.**Create a new branch** (git checkout -b feature-branch)

3.**Commit your changes** (git commit -m 'Add new feature')

4.**Push to the branch** (git push origin feature-branch)

5.**Open a Pull Request**

📄 License
This project is licensed under the MIT License.
