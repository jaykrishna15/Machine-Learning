# 🧠 Feature Engineering — Day 1

## Handling Missing Values 

### 🎯 Introduction

In this first part of the Feature Engineering series, we focus on Handling Missing Values — one of the most important steps in preparing data for machine learning models.
In real-world datasets, missing values are extremely common, and how you handle them directly impacts the accuracy and reliability of your model.

### ❓ Why Do Missing Values Occur?

Missing values appear when certain information is not recorded or available.
For example, in a survey, a participant might skip a question about salary, leading to a missing value.
These gaps can occur due to data entry errors, system failures, or respondents not providing complete information.

### ⚙️ Mechanisms of Missing Data

#### Missing Completely at Random (MCAR)

The missingness is completely random and unrelated to any feature.

Example: accidental data entry or technical errors.

#### Missing at Random (MAR)

The missingness depends only on other observed variables.

Example: men may avoid reporting their income, while women may avoid reporting their age.

#### Missing Not at Random (MNAR)

The missingness depends on unobserved variables or the missing value itself.

Example: employees dissatisfied with their salaries may choose not to disclose them.

#### 🚢 Example Dataset

The Titanic dataset is a popular example for understanding missing values, as it contains a mix of numerical and categorical variables with missing data.
It helps illustrate how different handling techniques affect real-world datasets.

### 🔍 Identifying Missing Values

Before handling missing data, it is essential to identify where the gaps exist.
You can explore the dataset to see which columns contain missing values and how many are missing in each.

### 🗑️ Deleting Missing Values

One approach is to remove rows or columns containing missing values.
However, this method can cause significant data loss and is generally used only when the missing data proportion is small or the dataset is very large.
Columns with too many missing values may also be dropped if they provide little information.

### 🧮 Imputation Techniques

Instead of deleting data, missing values can be replaced using various imputation methods.

Mean Imputation: Replace missing values with the average of that column. Works best for normally distributed numerical data.

Median Imputation: Replace missing values with the median. Best for data with outliers or skewed distributions.

Mode Imputation: Used for categorical data. Replace missing values with the most frequently occurring category.

Random Sample Imputation: Replace missing values with randomly selected existing values from the same column to preserve data variability.

### 🔁 Forward Fill and Backward Fill

These methods are simple yet effective for sequential or time-based data.

Forward Fill (ffill): Fills missing values using the previous available value.

Backward Fill (bfill): Fills missing values using the next available value.

They work well for maintaining continuity in ordered datasets.


| Technique                | Suitable For                   | Description                                |
| ------------------------ | ------------------------------ | ------------------------------------------ |
| Mean Imputation          | Normal distributions           | Replace missing values with mean           |
| Median Imputation        | Skewed or outlier-heavy data   | Replace missing values with median         |
| Mode Imputation          | Categorical data               | Replace missing values with mode           |
| Random Sample Imputation | Numeric variables              | Replace missing values randomly            |
| Forward/Backward Fill    | Sequential or time-series data | Fill using previous or next value          |
| Deletion                 | Large datasets                 | Remove missing data when impact is minimal |


#### 🧠 Key Takeaways

Missing data is categorized into MCAR, MAR, and MNAR.

Always identify the cause of missingness before applying a solution.

Avoid deleting data unless the dataset is large enough to absorb the loss.

Use imputation methods to maintain dataset completeness.

Forward and backward fill are efficient for time-series or ordered data.



# 📅 Day 2 – Outlier Detection, Data Cleaning & Feature Scaling

On Day 2, we extended the concepts from Day 1 (Handling Missing Values) and focused on improving data quality by identifying and treating outliers, duplicates, and applying feature scaling to numerical columns.
These steps are critical in real-world data analysis and machine learning pipelines, as they directly impact model performance and business insights.

### 1️⃣ Outliers – Definition & Importance

Outliers are data points that significantly differ from the majority of observations.
They may occur due to:

Data entry errors

Measurement issues

Rare but valid events

If not handled properly, outliers can:

Skew averages and distributions

Mislead visualizations

Degrade machine learning model accuracy

### 2️⃣ IQR (Interquartile Range) Method
🔹 Definition

The IQR method detects outliers using the spread of the middle 50% of the data.

IQR = Q3 − Q1

Lower Bound = Q1 − 1.5 × IQR

Upper Bound = Q3 + 1.5 × IQR

Values outside this range are treated as outliers.

🔹 When to Use

When data is not normally distributed

When you want a robust, statistics-based method

Commonly used in EDA (Exploratory Data Analysis)

### 3️⃣ Capping (Clipping)
🔹 Definition

Capping limits extreme values by replacing them with predefined minimum and maximum thresholds instead of removing them.

🔹 When to Use

When outliers are valid but extreme

When removing data points could cause data loss

Useful in financial, sales, and pricing data

### 4️⃣ Winsorization
🔹 Definition

Winsorization is a specialized form of capping where extreme values are replaced using percentile limits (e.g., 1st and 99th percentiles).

🔹 When to Use

When extreme values affect mean and variance

When you want to preserve dataset size

Frequently used before statistical modeling

### 5️⃣ Handling Duplicate Records
🔹 Definition

Duplicates occur when the same record appears more than once in the dataset.

🔹 When to Handle Duplicates

When duplicates inflate counts, sums, or averages

When working with transactional or customer-level data

Before building dashboards or ML models

Removing duplicates ensures data accuracy and consistency.

### 6️⃣ Feature Scaling (Numerical Columns)
🔹 Definition

Feature scaling transforms numerical values into a comparable range so that no single feature dominates due to scale differences.

Common scaling approaches include:

Standardization

Normalization

🔹 When to Use

Before applying distance-based algorithms (KNN, K-Means)

When features have different units or magnitudes

Essential for machine learning workflows

## 🎯 Key Learnings from Day 2

Identified and treated outliers using multiple statistical techniques

Understood when to remove vs when to cap extreme values

Cleaned duplicate records for reliable analysis

Applied feature scaling to prepare data for modeling
