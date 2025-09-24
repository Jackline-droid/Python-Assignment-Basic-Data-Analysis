# iris_analysis.py
# Assignment: Data loading, analysis, and visualization with pandas & matplotlib

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

try:
    # -------------------------------
    # Task 1: Load and Explore Dataset
    # -------------------------------
    iris = load_iris(as_frame=True)
    df = iris.frame
    df["target"] = iris.target  # add target column

    print("✅ Iris dataset loaded successfully!\n")
    print("First 5 rows of the dataset:")
    print(df.head())

    print("\n🔍 Dataset Info:")
    print(df.info())

    print("\n📊 Missing Values:")
    print(df.isnull().sum())

    # Clean dataset (Iris has no missing values, but we’ll drop any just in case)
    df = df.dropna()

    # -------------------------------
    # Task 2: Basic Data Analysis
    # -------------------------------
    print("\n📈 Basic Statistics:")
    print(df.describe())

    print("\n📌 Average Sepal Length by Species:")
    print(df.groupby("target")["sepal length (cm)"].mean())

    # -------------------------------
    # Task 3: Data Visualizations
    # -------------------------------
    # Line Chart: Petal length sorted
    df_sorted = df.sort_values(by="petal length (cm)")
    plt.figure(figsize=(8, 5))
    plt.plot(df_sorted["petal length (cm)"], marker="o", linestyle="-")
    plt.title("Line Chart: Petal Length Trend")
    plt.xlabel("Sample Index")
    plt.ylabel("Petal Length (cm)")
    plt.grid(True)
    plt.show()

    # Bar Chart: Average petal length per species
    plt.figure(figsize=(8, 5))
    sns.barplot(x="target", y="petal length (cm)", data=df, estimator="mean", palette="muted")
    plt.title("Average Petal Length per Species")
    plt.xlabel("Species (0=setosa, 1=versicolor, 2=virginica)")
    plt.ylabel("Avg Petal Length (cm)")
    plt.show()

    # Histogram: Sepal length distribution
    plt.figure(figsize=(8, 5))
    plt.hist(df["sepal length (cm)"], bins=15, color="skyblue", edgecolor="black")
    plt.title("Histogram: Sepal Length Distribution")
    plt.xlabel("Sepal Length (cm)")
    plt.ylabel("Frequency")
    plt.show()

    # Scatter Plot: Sepal length vs Petal length
    plt.figure(figsize=(8, 5))
    sns.scatterplot(
        x="sepal length (cm)", 
        y="petal length (cm)", 
        hue="target", 
        data=df, 
        palette="deep"
    )
    plt.title("Scatter Plot: Sepal vs Petal Length")
    plt.xlabel("Sepal Length (cm)")
    plt.ylabel("Petal Length (cm)")
    plt.legend(title="Species")
    plt.show()

    # -------------------------------
    # Findings / Observations
    # -------------------------------
    print("\n🔎 Observations:")
    print("- Setosa (target=0) generally has smaller petal lengths and widths.")
    print("- Virginica (target=2) tends to have the largest measurements overall.")
    print("- Versicolor (target=1) falls in between setosa and virginica.")
    print("- Scatter plot shows clear separation between species based on petal length.")

except FileNotFoundError:
    print("❌ Error: Dataset file not found.")
except ImportError as e:
    print(f"❌ Import Error: {e}")
except Exception as e:
    print(f"❌ An unexpected error occurred: {e}")
