import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


sns.set(style="whitegrid")


df = pd.read_csv('/content/diamonds.csv')

print(df.head())

print(df.describe())

print(df.isnull().sum())

plt.figure(figsize=(10, 6))
sns.histplot(df['price'], bins=50, kde=True)
plt.title('Distribution of Diamond Prices')
plt.xlabel('Price ($)')
plt.ylabel('Frequency')
plt.show()


plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='carat', y='price', alpha=0.5)
plt.title('Price vs. Carat Weight')
plt.xlabel('Carat Weight')
plt.ylabel('Price ($)')
plt.show()


plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='cut', y='price')
plt.title('Price Distribution by Cut Quality')
plt.xlabel('Cut Quality')
plt.ylabel('Price ($)')
plt.show()


plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='color', y='price')
plt.title('Price Distribution by Color')
plt.xlabel('Color')
plt.ylabel('Price ($)')
plt.show()


plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='clarity', y='price')
plt.title('Price Distribution by Clarity')
plt.xlabel('Clarity')
plt.ylabel('Price ($)')
plt.show()


numeric_df = df.select_dtypes(include=[float, int])

plt.figure(figsize=(12, 8))
correlation_matrix = numeric_df.corr()
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.title('Correlation Matrix of Diamond Features')
plt.show()


plt.savefig('correlation_matrix.png')