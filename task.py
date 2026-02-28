import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Load the dataset
# Ensure 'Titanic-Dataset.csv' is in the same directory as your script
df = pd.read_csv('Titanic-Dataset.csv')

# ==========================================
# 2. Exploratory Data Analysis & Visualizations
# ==========================================

# Set the visualization style
sns.set_theme(style="whitegrid")

# Visualization 1: Survival by Gender
plt.figure(figsize=(6, 4))
sns.countplot(x='Survived', hue='Sex', data=df, palette='Set2')
plt.title('Survival Count by Gender (0 = No, 1 = Yes)')
plt.show()

# Visualization 2: Survival by Passenger Class
plt.figure(figsize=(6, 4))
sns.countplot(x='Survived', hue='Pclass', data=df, palette='Set1')
plt.title('Survival Count by Passenger Class')
plt.show()

# Visualization 3: Age Distribution of Survivors vs Non-Survivors
plt.figure(figsize=(8, 5))
sns.histplot(data=df, x='Age', hue='Survived', kde=True, bins=30, palette='coolwarm')
plt.title('Age Distribution by Survival')
plt.show()


# ==========================================
# 3. Data Preprocessing
# ==========================================

# Fill missing 'Age' values with the median age
df['Age'] = df['Age'].fillna(df['Age'].median())

# Fill missing 'Embarked' values with the most common port
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# Drop columns that aren't useful for the base model or have too many missing values ('Cabin')
df.drop(['Cabin', 'Ticket', 'Name', 'PassengerId'], axis=1, inplace=True)

# Encode categorical variables ('Sex' and 'Embarked') into numerical formats
le_sex = LabelEncoder()
df['Sex'] = le_sex.fit_transform(df['Sex']) # male: 1, female: 0

le_embarked = LabelEncoder()
df['Embarked'] = le_embarked.fit_transform(df['Embarked'])


# ==========================================
# 4. Model Building
# ==========================================

# Split data into features (X) and target (y)
X = df.drop('Survived', axis=1)
y = df['Survived']

# Split into 80% training data and 20% testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Model (Random Forest)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


# ==========================================
# 5. Model Evaluation & Confusion Matrix
# ==========================================

# Make Predictions
y_pred = model.predict(X_test)

# Print metrics
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}\n")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Visualization 4: Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Not Survived', 'Survived'], 
            yticklabels=['Not Survived', 'Survived'])
plt.xlabel('Predicted Label')
plt.ylabel('Actual Label')
plt.title('Confusion Matrix')
plt.show()