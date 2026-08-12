import pandas as pd
import numpy as np

from scipy.stats import skew
from sklearn.preprocessing import StandardScaler


# 1. Load Dataset


df = pd.read_csv("titanic.csv")

print("=" * 50)
print("DATASET INFORMATION")
print("=" * 50)

print("\nShape:", df.shape)

print("\nColumns:")
print(df.columns)

print("\nFirst 5 Rows")
print(df.head())

print("\nData Types")
print(df.dtypes)

print("\nSummary Statistics")
print(df.describe(include="all"))


# 2. Missing Values


print("\nMissing Values")
print(df.isnull().sum())


if "Age" in df.columns:
    df["Age"] = df["Age"].fillna(df["Age"].median())


if "Embarked" in df.columns:
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])


if "Cabin" in df.columns:
    df.drop(columns=["Cabin"], inplace=True)


# 3. Remove Duplicates


duplicates = df.duplicated().sum()

print("\nDuplicate Rows:", duplicates)

df.drop_duplicates(inplace=True)


# 4. Handle Incorrect Data Types


print("\nCorrecting Data Types")

for col in df.columns:
    if df[col].dtype == "object":
        df[col] = df[col].astype(str)


# 5. Remove Irrelevant Features


remove_columns = ["PassengerId", "Name", "Ticket"]

for col in remove_columns:
    if col in df.columns:
        df.drop(columns=[col], inplace=True)

print("\nRemaining Columns")
print(df.columns)


# 5b. Set aside the target column


TARGET_COLUMN = "Survived"
target_series = None

if TARGET_COLUMN in df.columns:
    target_series = df[TARGET_COLUMN].copy()
    df.drop(columns=[TARGET_COLUMN], inplace=True)


# 6. Handle Outliers using IQR


numeric_columns = df.select_dtypes(include=np.number).columns

for col in numeric_columns:

    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)

    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    df[col] = np.where(df[col] < lower, lower, df[col])
    df[col] = np.where(df[col] > upper, upper, df[col])

print("\nOutliers handled using IQR")


# 7. Encode Categorical Variables


categorical = df.select_dtypes(include=["object", "str"]).columns

print("\nCategorical Columns")
print(categorical)

df = pd.get_dummies(df, columns=categorical, drop_first=True)


# 8. Handle Skewness


print("\nChecking Skewness")

numeric_columns = df.select_dtypes(include=np.number).columns

for col in numeric_columns:

    value = skew(df[col])

    print(col, ":", round(value, 2))

    if abs(value) > 1:
        df[col] = np.log1p(df[col])

print("\nSkewness handled.")


# 9. Feature Scaling


scaler = StandardScaler()

numeric_columns = df.select_dtypes(include=np.number).columns

df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

print("\nFeature Scaling Completed")


# 9b. Add the target column back (unscaled)


if target_series is not None:
    df.insert(0, TARGET_COLUMN, target_series.values)


# 10. Final Dataset


print("\nFinal Shape:", df.shape)

print("\nFirst Five Rows")

print(df.head())

df.to_csv("cleaned_titanic.csv", index=False)

print("\nCleaned dataset saved as cleaned_titanic.csv")
