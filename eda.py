




import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.preprocessing import LabelEncoder


pd.set_option('display.max_columns', None)
sns.set_style('whitegrid')

os.makedirs('eda_plots', exist_ok=True)




df = pd.read_csv('dataset_phishing.csv')

print("\n========== DATASET OVERVIEW ==========")

print("First 5 rows:")
print(df.head())

print("\nDataset Shape:", df.shape)

print("\nDataset Information:")
df.info()

print("\nStatistical Summary:")
print(df.describe().T)




print("\n========== MISSING VALUES ==========")

print(df.isnull().sum().sum())

print(df.isnull().sum()[df.isnull().sum() > 0])

plt.figure(figsize=(10, 5))

sns.heatmap(
    df.isnull(),
    cbar=False,
    yticklabels=False
)

plt.title("Missing Values Analysis")

plt.tight_layout()
plt.savefig("eda_plots/01_missing_values.png")
plt.show()



print("\n========== DUPLICATE ANALYSIS ==========")

print("Duplicate Rows:", df.duplicated().sum())

print("Duplicate URLs:", df['url'].duplicated().sum())




print("\n========== TARGET DISTRIBUTION ==========")

print(df['status'].value_counts())

print("\nTarget Percentage:")

print(df['status'].value_counts(normalize=True) * 100)

plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x='status',
    order=['legitimate', 'phishing']
)

plt.title("Phishing vs Legitimate Websites")

plt.xlabel("Website Status")
plt.ylabel("Number of Websites")

plt.tight_layout()
plt.savefig("eda_plots/02_target_distribution.png")
plt.show()




print("\n========== FEATURE TYPES ==========")

numerical_features = df.select_dtypes(
    include=np.number
).columns.tolist()

categorical_features = df.select_dtypes(
    include=['object', 'category', 'bool']
).columns.tolist()

print("Numerical Features:", len(numerical_features))
print("Categorical Features:", categorical_features)



print("\n========== FEATURE DISTRIBUTIONS ==========")

selected_features = numerical_features[:12]

df[selected_features].hist(
    figsize=(16, 12),
    bins=30,
    edgecolor='black'
)

plt.suptitle("Numerical Feature Distributions")

plt.tight_layout()
plt.savefig("eda_plots/03_feature_distributions.png")
plt.show()




print("\n========== CORRELATION HEATMAP ==========")

correlation = df[numerical_features].corr()

plt.figure(figsize=(22, 18))

sns.heatmap(
    correlation,
    cmap='coolwarm',
    center=0
)

plt.title("Feature Correlation Heatmap")

plt.tight_layout()
plt.savefig("eda_plots/04_correlation_heatmap.png")
plt.show()




print("\n========== TARGET CORRELATION ==========")

# Encode target for correlation analysis only

encoder = LabelEncoder()

encoded_status = encoder.fit_transform(df['status'])

correlation_data = df[numerical_features].copy()

correlation_data['status_encoded'] = encoded_status

target_corr = correlation_data.corr()['status_encoded'].drop(
    'status_encoded'
)

target_corr = target_corr.reindex(
    target_corr.abs().sort_values(
        ascending=False
    ).index
)

print("\nTop 15 Correlated Features:")

print(target_corr.head(15))

plt.figure(figsize=(12, 8))

sns.barplot(
    x=target_corr.head(15).values,
    y=target_corr.head(15).index
)

plt.title("Top 15 Features Correlated with Target")

plt.xlabel("Correlation")
plt.ylabel("Features")

plt.tight_layout()
plt.savefig("eda_plots/05_target_correlation.png")
plt.show()



print("\n========== OUTLIER DETECTION ==========")

selected_outlier_features = numerical_features[:8]

plt.figure(figsize=(16, 10))

for i, col in enumerate(selected_outlier_features, 1):

    plt.subplot(2, 4, i)

    sns.boxplot(x=df[col])

    plt.title(col)

plt.suptitle("Outlier Detection")

plt.tight_layout()
plt.savefig("eda_plots/06_outlier_detection.png")
plt.show()






        


print("\n========== FEATURE VS TARGET ==========")

selected_features = numerical_features[:6]

plt.figure(figsize=(16, 12))

for i, col in enumerate(selected_features, 1):

    plt.subplot(2, 3, i)

    sns.boxplot(
        data=df,
        x='status',
        y=col
    )

    plt.title(f'{col} vs Status')

plt.tight_layout()
plt.savefig("eda_plots/07_feature_vs_target.png")
plt.show()



print("\n========== CONSTANT FEATURES ==========")

constant_features = [
    col for col in numerical_features
    if df[col].nunique() <= 1
]

print("Constant Features:")
print(constant_features)

print("Total Constant Features:", len(constant_features))



print("\n========== HIGH CORRELATION PAIRS ==========")

feature_corr = df[numerical_features].corr().abs()

upper_triangle = feature_corr.where(
    np.triu(
        np.ones(feature_corr.shape),
        k=1
    ).astype(bool)
)

high_corr = (
    upper_triangle
    .stack()
    .loc[lambda x: x > 0.90]
    .sort_values(ascending=False)
)

print(high_corr)



print("\n========== FINAL EDA SUMMARY ==========")

print("Total Records:", df.shape[0])

print("Total Columns:", df.shape[1])

print("Total Numerical Features:", len(numerical_features))

print("Missing Values:", df.isnull().sum().sum())

print("Duplicate Rows:", df.duplicated().sum())

print("Duplicate URLs:", df['url'].duplicated().sum())

print("Constant Features:", len(constant_features))

print("\nTarget Distribution:")

print(df['status'].value_counts())

print("\nEDA COMPLETED SUCCESSFULLY!")

print("Plots saved in eda_plots folder.")
