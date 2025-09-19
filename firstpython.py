#Print the output
print("New Python file")


import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import pandas as pd

# Load iris dataset into a DataFrame
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

# 1. Boxplot (example: sepal length)
plt.figure(figsize=(6, 4))
sns.boxplot(y=df["sepal length (cm)"])
plt.title("Boxplot of Sepal Length")
plt.ylabel("Sepal Length (cm)")
plt.show()

# 2. Barplot (species count)
plt.figure(figsize=(6, 4))
sns.countplot(x="species", data=df)
plt.title("Barplot of Species Counts")
plt.xlabel("Species")
plt.ylabel("Count")
plt.show()

# 3. Boxplot (Petal length vs species)
plt.figure(figsize=(6, 4))
sns.boxplot(x="species", y="petal length (cm)", data=df)
plt.title("Boxplot of Petal Length by Species")
plt.xlabel("Species")
plt.ylabel("Petal Length (cm)")
plt.show()

# 4. Scatter plot (Sepal length vs Sepal width)
plt.figure(figsize=(6, 4))
sns.scatterplot(x="sepal length (cm)", y="sepal width (cm)", hue="species", data=df)
plt.title("Scatter Plot: Sepal Length vs Sepal Width")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Sepal Width (cm)")
plt.show()

# 5. Histogram (Petal width distribution)
plt.figure(figsize=(6, 4))
sns.histplot(df["petal width (cm)"], bins=15, kde=True)
plt.title("Histogram of Petal Width")
plt.xlabel("Petal Width (cm)")
plt.ylabel("Frequency")
plt.show()
