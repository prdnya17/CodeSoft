# CodeSoft
# Task 1: Titanic Survival Prediction: Machine Learning Dashboard

An end-to-end Data Science project utilizing the Titanic dataset to predict passenger survival using a **Random Forest Classifier**. This project includes comprehensive data preprocessing, feature encoding, and a quadrant-based visualization dashboard.

## 📊 1. Training Data Preview
The model was trained on a cleaned version of the dataset. Below is a snapshot of the processed training features (first 5 rows):

| Pclass | Sex | Age | SibSp | Parch | Fare | Embarked |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | 1 | 45.5 | 0 | 0 | 28.5000 | 2 |
| 2 | 1 | 23.0 | 0 | 0 | 13.0000 | 2 |
| 3 | 1 | 32.0 | 0 | 0 | 7.9250 | 2 |
| 3 | 1 | 26.0 | 1 | 0 | 7.8542 | 2 |
| 3 | 0 | 6.0 | 4 | 2 | 31.2750 | 2 |

> *Note: Sex (0=Female, 1=Male) | Embarked (0=C, 1=Q, 2=S)*

## 📈 2. Analysis Dashboard (Quadrant View)
The project generates a consolidated $2 \times 2$ quadrant visualization to analyze survival drivers and model performance:



1.  **Top-Left (Gender):** Highlights the "Women and Children first" protocol.
2.  **Top-Right (Class):** Visualizes the survival disparity between 1st, 2nd, and 3rd class.
3.  **Bottom-Left (Age):** A KDE distribution showing age peaks for survivors.
4.  **Bottom-Right (Confusion Matrix):** Displays the model's True Positives vs. False Positives.

## 🤖 3. Model Performance
The Random Forest model was evaluated on a 20% test split:

* **Overall Accuracy:** `82.12%`
* **Precision (Survived):** `0.81`
* **Recall (Survived):** `0.74`
* **F1-Score:** `0.77`

## 🛠️ 4. Tech Stack & Requirements
* **Language:** Python 3.x
* **Libraries:** `pandas`, `seaborn`, `matplotlib`, `scikit-learn`
* **Dataset:** Titanic-Dataset.csv

## 🚀 5. How to Run
1. Ensure `Titanic-Dataset.csv` is in the same folder as the script.
2. Run the analysis script:
   ```bash
   python titanic_analysis.py
