import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# Load the dataset
df = pd.read_csv(r"C:\Users\Deeya Srivastava\Desktop\aiml dataset.csv")

# Display the first 5 rows
print("First 5 rows of the dataset:")
print(df.head())

# Check for missing values
missing_values = df.isnull().sum()
print("\nMissing values in each column:")
print(missing_values)

# Separate numerical and categorical columns
numeric_columns = df.select_dtypes(include=['number']).columns
categorical_columns = df.select_dtypes(include=['object']).columns

# Fill missing values
df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())
df[categorical_columns] = df[categorical_columns].fillna(df[categorical_columns].mode().iloc[0])

# Display summary statistics
print("\nSummary statistics for numerical columns:")
print(df.describe())

# Display data types
print("\nData Types of Columns:")
print(df.dtypes)

# Churn analysis
churned_customers = df[df['Churn Label'] == 'Churned']
print(f"\nNumber of churned customers: {len(churned_customers)}")
print(f"Churn rate: {len(churned_customers) / len(df) * 100:.2f}%")

# Visualization: Churn Analysis by Contract Type
contract_counts = df.groupby(['Contract', 'Churn Label']).size().unstack()
contract_counts.plot(kind='bar', stacked=True, figsize=(8, 6))
plt.title('Churn Analysis by Contract Type')
plt.xlabel('Contract Type')
plt.ylabel('Count')
plt.legend(title='Churn Label')
plt.show()

# Visualization: Monthly Charges for Churned vs Non-Churned Customers
plt.figure(figsize=(8, 6))
df.boxplot(column='Monthly Charge', by='Churn Label')
plt.title('Monthly Charges for Churned vs Non-Churned Customers')
plt.suptitle('')
plt.xlabel('Churn Label')
plt.ylabel('Monthly Charge')
plt.show()

# Correlation Matrix of Numerical Columns
correlation_matrix = df[numeric_columns].corr()
fig, ax = plt.subplots(figsize=(12, 8))
cax = ax.matshow(correlation_matrix, cmap='coolwarm')
fig.colorbar(cax)
plt.xticks(range(len(numeric_columns)), numeric_columns, rotation=45)
plt.yticks(range(len(numeric_columns)), numeric_columns)
plt.title('Correlation Matrix of Numerical Columns')
plt.show()

# Visualization: Satisfaction Score for Churned vs Non-Churned Customers
plt.figure(figsize=(8, 6))
df.boxplot(column='Satisfaction Score', by='Churn Label')
plt.title('Satisfaction Score for Churned vs Non-Churned Customers')
plt.suptitle('')
plt.xlabel('Churn Label')
plt.ylabel('Satisfaction Score')
plt.show()

# Display value counts for categorical columns
for column in categorical_columns:
    print(f"\nValue counts for {column}:")
    print(df[column].value_counts())

# Add Age Group column
df['Age Group'] = pd.cut(df['Age'], bins=[0, 30, 50, 100], labels=['Under 30', '30-50', 'Above 50'])
print("\nNew Age Group column added:")
print(df[['Age', 'Age Group']].head())

# Encoding categorical variables
label_encoder = LabelEncoder()
df['Gender'] = label_encoder.fit_transform(df['Gender'])
df['Churn Label'] = label_encoder.fit_transform(df['Churn Label'])
df['Contract'] = label_encoder.fit_transform(df['Contract'])
df['Payment Method'] = label_encoder.fit_transform(df['Payment Method'])
df['Offer'] = label_encoder.fit_transform(df['Offer'])

# Model Training
features = ['Age', 'Tenure in Months', 'Monthly Charge', 'Satisfaction Score', 'Contract', 'Payment Method']
target = 'Churn Label'

X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Model Evaluation
y_pred = model.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
plt.matshow(conf_matrix, cmap='Blues', fignum=1)
plt.colorbar()
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
