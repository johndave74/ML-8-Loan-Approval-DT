# ML-8-Loan-Approval-DT
This project aims to build a predictive model that determines whether a customer qualifies for a loan based on their financial and personal attributes.

# Bank Loan Approval Prediction

![image](https://github.com/user-attachments/assets/bbb1c860-4b2d-4465-b46e-97c4cadf73b8)

## Introduction
Banks face significant risks when approving personal loans, as lending to unqualified individuals can result in financial losses. To minimize these risks, financial institutions use data-driven approaches to assess a customer's eligibility. This project aims to build a predictive model that determines whether a customer qualifies for a loan based on their financial and personal attributes.

## Objectives
- Understand the key factors influencing loan approval decisions.
- Perform **Exploratory Data Analysis (EDA)** to uncover patterns in the dataset.
- Train a **Decision Tree Classifier** to predict loan approval.
- Evaluate the model's performance and interpret its results.
- Suggest potential improvements for better accuracy and reliability.

## Dataset
The dataset used is **`bankloan.csv`**, which consists of:
- `Age`: Customer's age.
- `Experience`: Work experience in years.
- `Income`: Annual income in $1000s.
- `ZIP Code`: Customer's ZIP code.
- `Family`: Number of family members.
- `CCAvg`: Average monthly credit card spending.
- `Education`: Education level (1 = Undergraduate, 2 = Graduate, 3 = Advanced/Professional).
- `Mortgage`: Value of house mortgage if any.
- `Securities Account`: Does the customer have a securities account? (1 = Yes, 0 = No)
- `CD Account`: Does the customer have a certificate of deposit account? (1 = Yes, 0 = No)
- `Online`: Does the customer use online banking? (1 = Yes, 0 = No)
- `CreditCard`: Does the customer have a credit card issued by the bank? (1 = Yes, 0 = No)
- `Personal.Loan`: **Target variable** (1 = Loan approved, 0 = Not approved).

## Steps to Run the Project

### 1. Install Dependencies
Ensure Python is installed along with the required libraries.

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### 2. Load the Data
```python
import pandas as pd

# Load dataset
bank_loan = pd.read_csv('bankloan.csv')
bank_loan.head()
```

## Exploratory Data Analysis (EDA)
### Understanding the Data
- The dataset contains **numerical** and **categorical** variables.
- The **target variable** (`Personal.Loan`) indicates whether a customer was approved for a loan.

### Data Distribution
- **Count Plot**: Visualizes the distribution of loan approvals.
```python
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(12,4))
sns.countplot(x='Personal.Loan', data=bank_loan, palette='Set2')
plt.show()
```
- **Correlation Analysis**: Identifies relationships between variables.
```python
import seaborn as sns
import numpy as np

plt.figure(figsize=(10,6))
sns.heatmap(bank_loan.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Feature Correlation Heatmap')
plt.show()
```
- **Key Findings from EDA**:
  - **Income and Credit Card Spending are strong indicators of loan approval.**
  - **Customers with a CD (Certificate of Deposit) account are more likely to be approved.**
  - **Education level plays a significant role in determining loan eligibility.**
  - **People with an online banking account are more likely to get loans.**

## Data Preprocessing
- **Feature Selection**: Excluding unnecessary columns (`ZIP Code`).
- **Encoding**: Categorical features converted into numerical representations.
- **Splitting Data**: 70% for training, 30% for testing.

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

X = bank_loan.drop(columns=['Personal.Loan', 'ZIP Code'])
y = bank_loan['Personal.Loan']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```

## Model Training
### Algorithm Used: Decision Tree Classifier
- A **Decision Tree Classifier** was chosen due to its interpretability and ability to handle both numerical and categorical data.
- The model learns rules based on patterns in the data to predict loan approval.

```python
from sklearn.tree import DecisionTreeClassifier

dtree = DecisionTreeClassifier(min_samples_leaf=3000, random_state=42)
dtree.fit(X_train, y_train)
```

### Model Evaluation
- **Accuracy Score**
```python
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, dtree.predict(X_test))
print("Accuracy:", accuracy)
```
- **Classification Report**
```python
from sklearn.metrics import classification_report
print(classification_report(y_test, dtree.predict(X_test)))
```
- **Confusion Matrix**
```python
from sklearn.metrics import confusion_matrix
import seaborn as sns

cm = confusion_matrix(y_test, dtree.predict(X_test))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
```

## Insights
- **Income and credit card spending are the most influential factors in loan approval.**
- **Customers with a CD account are highly likely to get approved.**
- **Decision Trees provide clear classification rules, making them useful for financial decision-making.**
- **The model provides an accuracy score that helps in understanding its reliability.**

## Future Improvements
- Tune hyperparameters to improve accuracy.
- Test alternative models like **Random Forest** or **Logistic Regression** for comparison.
- Deploy the model as an API or web-based application.

## Repository Structure
```
|-- bankloan.ipynb  # Jupyter Notebook with analysis
|-- bankloan.csv    # Dataset file
|-- README.md       # Project documentation
```

## Contributing
Feel free to open issues or submit pull requests to improve the project.

## License
This project is licensed under the MIT License.
