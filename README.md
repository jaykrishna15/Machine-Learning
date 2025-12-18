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

# 📅 Day 3: Handling Categorical Data in Machine Learning

### Introduction

In real-world datasets, data is rarely entirely numerical. Many features are categorical, such as gender, city, product type, education level, or customer segment. These variables contain important information but cannot be directly processed by most Machine Learning algorithms.

Since Machine Learning models primarily work with numerical inputs, converting categorical columns into meaningful numerical representations is a critical data preprocessing step.

In this project, the dataset contains both numerical and categorical features. The data is first structured into a DataFrame and then prepared for modeling using the loan_approved dataset. Special attention is given to handling categorical columns so they integrate correctly with numerical features and contribute effectively to model performance.

In Day 3 of this series, the focus is on encoding techniques that transform categorical variables into machine-readable formats while preserving as much information as possible.

### Types of Categorical Data

Before selecting an encoding technique, it is important to understand the type of categorical feature, as the choice of encoding depends on it.

#### 1. Nominal Data

Categories have no inherent order

##### Example:

Color (Red, Blue, Green)

City names

#### 2. Ordinal Data

Categories have a meaningful order

##### Example:

Education level

Ratings (Low, Medium, High)

#### 1. Label Encoding

##### Definition

Label Encoding converts each unique category into a unique integer value.

##### Example:

Male → 0

Female → 1

##### When to Use

When the categorical feature is binary

When using tree-based models:

Decision Trees

Random Forest

XGBoost

##### When the model does not assume numerical distance or magnitude

##### When Not to Use

For nominal features with more than two categories

For linear models, as it may introduce false ordinal relationships

### 2. One-Hot Encoding

##### Definition

One-Hot Encoding creates separate binary columns for each category.
Each row contains a 1 for the present category and 0 for others.

##### When to Use

For nominal categorical variables

When there is no order between categories

Suitable for models such as:

Linear Regression

Logistic Regression

KNN

Neural Networks

Limitations

Increases dimensionality (curse of dimensionality)

Not efficient for features with many unique categories

### 3. Ordinal Encoding

##### Definition

Ordinal Encoding assigns ordered numerical values to categories based on their ranking or importance.

Example:

Low → 1

Medium → 2

High → 3

##### When to Use

When categories have a clear and meaningful order

When the order carries business or logical significance

Suitable for both linear and tree-based models

##### Important Note

The order must be domain-driven, not arbitrary

Incorrect ordering can negatively affect model performance

### 4. Frequency Encoding

#### Definition

Frequency Encoding replaces each category with its frequency or proportion in the dataset.

Example:

City A → 0.35

City B → 0.15

#####  When to Use

When categorical features have high cardinality

To reduce dimensionality compared to one-hot encoding

Commonly used with:

Large datasets

Tree-based models

##### Advantages

Compact representation

Retains information about category importance

Avoids creating many sparse columns

###### Limitations

Categories with the same frequency receive the same value


# 📊 Feature Engineering — Day 3
Column Transformation (Using Multiple Datasets)
### 🎯 Overview

In Day 3 of the Feature Engineering series, I worked on column transformation techniques and applied them to two different datasets to understand how encoding and scaling strategies change based on feature types and dataset requirements.

The focus was on combining encoding + scaling correctly to make datasets fully model-ready.

#### 📁 Datasets Used
#### 1️⃣ Banana Quality Dataset

Goal: Prepare quality-related features for machine learning models.

##### Techniques Applied:

OrdinalEncoder → For ordered categorical features

OneHotEncoder → For nominal categorical features

StandardScaler → For numerical feature scaling

##### 🔹 Why this approach?

Some banana quality features have natural ordering, making Ordinal Encoding appropriate

Other categorical features have no ranking, so One-Hot Encoding avoids false relationships

Numerical columns were scaled using StandardScaler to ensure equal contribution during model training

#### 2️⃣ Loan Approval Dataset

Goal: Prepare applicant and loan attributes for approval prediction.

##### Techniques Applied:

OneHotEncoder → For all categorical features

StandardScaler → For numerical features

#####🔹 Why this approach?

Loan dataset categorical features are purely nominal

No ordinal relationship exists between categories

Scaling numerical columns improves performance for distance-based and gradient-based models

### 🔍 Column Transformation Summary
Dataset	Encoding Used	Scaling
Banana Quality	OrdinalEncoder, OneHotEncoder	StandardScaler
Loan Approval	OneHotEncoder	StandardScaler
### 🚀 Key Learnings

Column transformation depends on feature meaning, not just data type

Mixing encoders within a dataset is often necessary

StandardScaler ensures numerical stability across models

Correct preprocessing significantly improves downstream ML performance

### 📌 Next Steps

Upcoming topics in the series:

Feature Scaling deep dive

Feature selection techniques

Building end-to-end preprocessing pipelines
