# Titanic Survival Prediction – Exploratory Data Analysis & Logistic Regression

## Project Overview

This project analyzes the **Titanic dataset** to understand the factors that influenced passenger survival and builds a **machine learning model to predict survival**.

The project covers the complete **data science workflow**, including:

* Data loading
* Data cleaning
* Exploratory Data Analysis (EDA)
* Feature engineering
* Data visualization
* Model training using **Logistic Regression**
* Model evaluation

The goal is to identify patterns that influenced survival and demonstrate a **classification pipeline using Python and Scikit-learn**.

---

# Dataset

The dataset used in this project is the **Titanic dataset** available through the Seaborn library.

It contains information about passengers such as:

* Age
* Gender
* Passenger class
* Fare
* Embarkation port
* Family relationships
* Survival status

Target variable:

```
survived
```

* **0 → Did not survive**
* **1 → Survived**

---

# Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Jupyter Notebook

---

# Project Workflow

## 1. Data Loading

The dataset is loaded using Seaborn:

```python
df = sns.load_dataset("titanic")
```

This provides a structured dataset containing passenger details and survival information.

---

# 2. Data Cleaning

Several preprocessing steps were performed to handle missing and irrelevant data.

### Handling Missing Values

* Missing **Age values** were replaced with the **median age**.
* The **deck column** was removed due to excessive missing values.
* Remaining rows with missing values were dropped.

Example:

```python
df['age'] = df['age'].fillna(df['age'].median())
df.drop("deck", axis=1)
df.dropna(inplace=True)
```

---

# 3. Exploratory Data Analysis (EDA)

Multiple visualizations were created to understand patterns in the dataset.

### Visualizations Used

* Boxplots
* Scatterplots
* Countplots
* Heatmaps
* Histograms
* Distribution plots
* Pie charts

These visualizations help analyze relationships such as:

* Survival vs Age
* Survival vs Gender
* Survival vs Passenger Class
* Age distribution
* Fare distribution

Example visualization:

```python
sns.countplot(x='survived', hue='sex', data=df)
```

---

# 4. Correlation Analysis

A correlation heatmap was used to identify relationships between numerical features.

```python
sns.heatmap(df.corr(), annot=True)
```

This helps identify features that may influence survival probability.

---

# 5. Feature Engineering

Categorical variables were converted into numerical format using **one-hot encoding**.

Example:

```python
gender = pd.get_dummies(df['sex'], drop_first=True)
embarked = pd.get_dummies(df['embarked'], drop_first=True)
pclass = pd.get_dummies(df['pclass'], drop_first=True)
```

These encoded features were combined with the dataset to create the final modeling dataset.

Irrelevant columns were removed, including:

* sex
* pclass
* embark_town
* alive
* adult_male
* class
* alone
* who

---

# 6. Feature and Target Selection

The dataset was split into:

**Features (X)**
Passenger attributes used for prediction.

**Target (y)**
Survival status.

```python
X = df2.drop(['survived'], axis=1)
y = df2['survived']
```

---

# 7. Train-Test Split

The dataset was divided into **training and testing sets**.

* 80% training data
* 20% testing data

```python
train_test_split(X, y, train_size=0.8, random_state=42)
```

---

# 8. Model Training

A **Logistic Regression model** was used for binary classification.

```python
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
lr.fit(x_train, y_train)
```

Logistic regression is commonly used for **binary classification problems** such as survival prediction.

---

# 9. Model Prediction

Predictions were generated on the test dataset.

```python
prediction = lr.predict(x_test)
```

---

# 10. Model Evaluation

Model performance was evaluated using:

### Classification Report

Includes:

* Precision
* Recall
* F1-score
* Accuracy

```python
classification_report(y_test, prediction)
```

### Confusion Matrix

Shows correct and incorrect predictions.

```python
confusion_matrix(y_test, prediction)
```

---

# Key Insights from the Analysis

Some patterns observed during EDA include:

* **Female passengers had a higher survival rate than males**
* **Passengers in first class had better survival chances**
* **Younger passengers showed different survival trends**
* **Passenger class strongly influenced survival probability**

---

# Project Structure

```
Titanic-Survival-Analysis
│
├── Titanic dataset.ipynb
├── README.md
```

---

# How to Run the Project

1. Clone the repository

```
git clone https://github.com/yourusername/titanic-survival-analysis.git
```

2. Navigate to the project directory

```
cd titanic-survival-analysis
```

3. Install required libraries

```
pip install pandas numpy matplotlib seaborn scikit-learn
```

4. Run the Jupyter Notebook

```
jupyter notebook
```

Open the notebook and run all cells.

---

# Learning Outcomes

This project demonstrates:

* Practical **data cleaning techniques**
* **Exploratory data analysis**
* **Data visualization using Seaborn**
* **Feature engineering**
* **Binary classification using Logistic Regression**
* **Model evaluation using Scikit-learn**

---

# Future Improvements

Possible improvements include:

* Testing additional models such as:

  * Random Forest
  * Gradient Boosting
  * XGBoost
* Hyperparameter tuning
* Cross-validation
* Feature importance analysis
* Building a prediction web app using **Streamlit**
