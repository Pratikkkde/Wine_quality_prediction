# Wine_quality_prediction

# Project Overview
This project focuses on predicting the quality of wine based on various physicochemical properties involved in its preparation. Using Python and machine learning techniques, this project aims to classify wine samples into quality categories based on a set of measurable attributes.

The dataset includes features like:
Fixed Acidity
Volatile Acidity
Citric Acid
Residual Sugar
Chlorides
Free Sulfur Dioxide
Total Sulfur Dioxide
Density
pH
Sulphates
Alcohol
Quality (target variable)

# Exploratory data analysis (EDA)
Before building the model, an in-depth descriptive analysis and statistical summary were performed to understand the distribution and central tendencies of the variables.

Key steps in EDA:
Statistical Analysis: Mean, median, mode, standard deviation, etc.
Visualizations:
Box Plots: To detect outliers and spread of the data.
Histograms: To visualize the distribution of each feature.
Correlation Heatmap: To identify relationships between variables.

Data Preprocessing:
Handled skewness in right-skewed data through appropriate transformations.
Cleaned the dataset and prepared it for machine learning.

Histogram Insights
Right-skewed Variables:
volatile_acidity, citric_acid, residual_sugar, chlorides, free_sulfur_dioxide.
→ Most values cluster low with a few high outliers.

Normally Distributed:
fixed_acidity, pH, total_sulfur_dioxide.

Uniform Spread:
alcohol shows an even distribution between 8–14%.

Quality Ratings:
Mostly 5–6 (average wines); few at extreme ends.

Boxplot Insights
High Outliers:
residual_sugar, free_sulfur_dioxide, total_sulfur_dioxide, chlorides show multiple high-value outliers.

Moderate Spread:
volatile_acidity, citric_acid, sulphates, alcohol.

Tightly Clustered:
density, pH.

Correlation Heatmap Insights

Strong Positive:
Residual sugar ↔ Density (0.84)
Free SO₂ ↔ Total SO₂ (0.62)
Alcohol ↔ Quality (0.44)

Strong Negative:
Alcohol ↔ Density (-0.78)
pH ↔ Fixed acidity (-0.43)

Weak Correlations:
Most other variables have low correlation with quality, indicating it depends on multiple subtle factors.

# Machine Learning Model
For prediction, the Random Forest Classifier model was employed due to its robustness and efficiency in handling classification problems.
Model Performance:
Accuracy Achieved: 📈 82.4%
The Random Forest model was trained and tested on the cleaned and processed dataset, achieving reliable performance in predicting wine quality categories.

# Tech Stack
Python 🐍
Pandas
NumPy
Matplotlib
Seaborn
Scikit-learn

# Conclusion
Alcohol content was the strongest individual predictor of wine quality.
Most features showed only moderate or weak correlation with wine quality.
Random Forest classifier performed well with minimal tuning.
Future improvements: hyperparameter tuning, cross-validation, feature engineering, testing with other classifiers (like Gradient Boosting, SVM).
