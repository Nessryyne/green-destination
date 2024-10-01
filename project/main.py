import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression



data = pd.read_csv('greendestination.csv')

x = data[['Age', 'YearsAtCompany', 'MonthlyIncome']]
y = data['Attrition'].apply(lambda x: 1 if x == 'Yes' else 0)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(x_train, y_train)
print('Model Accuracy:', model.score(x_test, y_test))

age = data["Age"]

years_at_company = data["YearsAtCompany"]

MonthlyIncome = data["MonthlyIncome"]

Attrition = data["Attrition"]

print(data.isnull().sum())

attrition_rate = data["Attrition"].value_counts(normalize=True) * 100
print(attrition_rate)

plt.figure(figsize=(10,6))
plt.hist([age[Attrition == 'Yes'], age[Attrition == 'No']], label=['Attrition', 'No Attrition'], bins=10, alpha=0.7)
plt.xlabel('Age')
plt.ylabel('Count')
plt.legend()
plt.title('Attrition by Age')
plt.show()

plt.figure(figsize=(10,6))
plt.hist([years_at_company[Attrition == 'Yes'], years_at_company[Attrition == 'No']], label=['Attrition', 'No Attrition'], bins=10, alpha=0.7)
plt.xlabel('Years at Company')
plt.ylabel('Count')
plt.legend()
plt.title('Attrition by Years at Company')
plt.show()

plt.figure(figsize=(10,6))
plt.hist([MonthlyIncome[Attrition == 'Yes'], MonthlyIncome[Attrition == 'No']], label=['Attrition', 'No Attrition'], bins=10, alpha=0.7)
plt.xlabel('Monthly Income')
plt.ylabel('Count')
plt.legend()
plt.title('Attrition by Monthly Income')
plt.show()

data['Attrition_numeric'] = data['Attrition'].apply(lambda x: 1 if x == Yes else 0)

correlation_matrix = data.corr()
print(correlation_matrix)

plt.figure(figsize=(10,6))
data.boxplot(column="Age", by="Attrition")
plt.title("Age vs. Attrition")
plt.suptitle('')
plt.show()