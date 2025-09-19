import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats
from scipy.stats import ttest_ind
from scipy.stats import f_oneway
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression

df = pd.read_csv("boston_housing.csv")
df

# For the "Median value of owner-occupied homes" provide a boxplot
plt.figure(figsize=(12, 6))
plt.boxplot(df['MEDV'], vert=False)
plt.title("Median value of owner-occupied homes")
plt.show()

# Provide a bar plot for the Charles river variable
charles = df['CHAS'].value_counts()

plt.bar(charles.index, charles.values, color=['purple', 'blue'])
plt.title('Bar Plot for Charles River Variable')
plt.ylabel('Count')
plt.xticks(charles.index, labels=['Not Near River', 'Near River'])

# Provide a boxplot for the MEDV variable vs the AGE variable.
# (Discretize the age variable into three groups of 35 years and younger, between 35 and 70 years and 70 years and older)
bins = [0, 35, 70, float('inf')]
labels = ['35 years and younger', 'Between 35 and 70 years', '70 years and older']
df['AGE_Group'] = pd.cut(df['AGE'], bins=bins, labels=labels, right=False)

plt.figure(figsize=(10, 6))
plt.boxplot([df[df['AGE_Group'] == label]['MEDV'] for label in labels], tick_labels=labels)
plt.title('MEDV vs. AGE Groups')
plt.ylabel('MEDV')
plt.show()

# Provide a scatter plot to show the relationship between Nitric oxide concentrations and the proportion of
# non-retail business acres per town. What can you say about the relationship?
plt.scatter(df['INDUS'], df['NOX'], color='blue', alpha=0.9)
plt.title('NOX vs. INDUS')
plt.xlabel('INDUS')
plt.ylabel('NOX')
plt.show()

# Create a histogram for the pupil to teacher ratio variable
plt.hist(df['PTRATIO'], bins=10, color='blue', edgecolor='black')
plt.title('Pupil-Teacher Ratio')
plt.show()

# Is there a significant difference in median value of houses bounded by the Charles river or not? (T-test for independent samples)
bounded = df[df['CHAS'] == 1]['MEDV']
not_bounded = df[df['CHAS'] == 0]['MEDV']

t_stat, p_value = ttest_ind(bounded, not_bounded)

print('p-value:', p_value)
print('p-value is smaller than alpha value, so there is a significant difference.')

# Is there a difference in Median values of houses (MEDV) for each proportion of owner occupied units built prior to 1940 (AGE)? (ANOVA)
f_stat, p_value = f_oneway(df[df['AGE_Group'] == '35 years and younger']['MEDV'],
                           df[df['AGE_Group'] == 'Between 35 and 70 years']['MEDV'],
                           df[df['AGE_Group'] == '70 years and older']['MEDV'])

print('p-value:', p_value)
print('p-value is smaller than alpha value, so there is a difference.')

# Can we conclude that there is no relationship between Nitric oxide concentrations
# and proportion of non-retail business acres per town? (Pearson Correlation)
correlation_coefficient, p_value = pearsonr(df['NOX'], df['INDUS'])
print('Pearson Correlation Coefficient:', correlation_coefficient)
print('p-value:', p_value)
print('Pearson Correlation Coefficient value (0.763651446920915) indicates a strong relationship. Also, p-value is so small. Therefore, there is a relationship.')

plt.show()

# What is the impact of an additional weighted distance to the five Boston employment centres on the median value of owner occupied homes?
# (Regression analysis)
x = df[['DIS']]  
y = df['MEDV']

model = LinearRegression()
model.fit(x, y)

correlation_coefficient, p_value = pearsonr(df['DIS'], df['MEDV'])

plt.scatter(x, y, color='blue', label='Actual Data')
plt.plot(x, model.predict(x), color='red', linewidth=2, label='Regression Line')
plt.title('Regression Analysis: DIS vs. MEDV')
plt.xlabel('DIS')
plt.ylabel('MEDV')
plt.legend()
plt.show()

print('Pearson Correlation Coefficient:', correlation_coefficient)
print('p-value:', p_value)
print('p-value is smaller than alpha value and Pearson Correlation Coefficient is bigger than 0. Therefore, DIS has a positive impact on MEDV.')

