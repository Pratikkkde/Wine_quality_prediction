import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

#importing the dataset
df = pd.read_csv('wine_quality.csv')

# Standardizing the headers
df.columns = [col.strip().lower().replace(' ', '_') for col in df.columns]
print(df.columns)

# Check for null/empty values (No empty values observed)
print(df.isnull().sum())

# Managing the datatypes of  the columns
for col in df.columns:
    if col != 'quality':
        df[col] = df[col].astype(float)
    else:
        df[col] = df[col].astype(int)
print(df.dtypes)



# Obtaining the Descriptive statistics (central tendency, dispersion, and shape)
print(df.describe().round(2)) 

# Set style
sns.set(style="whitegrid")

# Histograms for all numerical variables
df.hist(bins=20, figsize=(18, 12), color='steelblue', edgecolor='black')
plt.suptitle("Histograms of All Variables", fontsize=20)

# Save the plot to file
plt.savefig('wine_quality_histograms.png', dpi=300, bbox_inches='tight')  
plt.show()


# Boxplots for outlier detection
# Set figure size for multiple boxplots
plt.figure(figsize=(16, 12))

for idx, col in enumerate(df.columns):
    if col != 'quality':
        plt.subplot(4, 3, idx + 1)  # Adjust rows, columns as needed
        sns.boxplot(x=df[col], color='tomato')
        plt.title(f'Boxplot of {col}')

plt.tight_layout()
plt.suptitle("Boxplots of All Features", fontsize=20, y=1.02)

plt.savefig('wine_quality_boxplots.png', dpi=300, bbox_inches='tight')
plt.show()

# Correlation Heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap", fontsize=18)

plt.savefig('wine_quality_correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()


# Save the value count barplot
plt.figure(figsize=(8, 5))
sns.countplot(x='quality', data=df, palette='viridis')
plt.title("Distribution of Wine Quality Scores")
plt.xlabel("Quality Rating")
plt.ylabel("Count")

plt.savefig('wine_quality_value_counts.png', dpi=300, bbox_inches='tight')
plt.show()


# Data distribution showed that some of the columns are head skewed
# List of right-skewed features based on your analysis
skewed_features = ['residual_sugar', 'free_sulfur_dioxide',
                   'total_sulfur_dioxide', 'chlorides', 'volatile_acidity']

# Get original skewness values
original_skewness = {col: skew(df[col]) for col in skewed_features}

# Plot and save original skewness as bar plot
plt.figure(figsize=(8, 5))
sns.barplot(x=list(original_skewness.keys()), y=list(original_skewness.values()))
plt.title("Skewness Before Transformation")
plt.ylabel("Skewness Value")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("skewness_before_transformation.png")
plt.show()

# Apply log1p transformation
for col in skewed_features:
    df[col] = np.log1p(df[col])

# Get skewness after transformation
transformed_skewness = {col: skew(df[col]) for col in skewed_features}

# Plot and save transformed skewness as bar plot
plt.figure(figsize=(8, 5))
sns.barplot(x=list(transformed_skewness.keys()), y=list(transformed_skewness.values()), palette="viridis")
plt.title("Skewness After Log1p Transformation")
plt.ylabel("Skewness Value")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("skewness_after_transformation.png")
plt.show()

# Optionally print both skewness values side by side
print("\nSkewness Comparison:")
for col in skewed_features:
    print(f"{col} - Before: {original_skewness[col]:.3f}, After: {transformed_skewness[col]:.3f}")



# Capping any outliers that might impact the results
# Define thresholds for extreme outliers
residual_sugar_threshold = 30
total_sulfur_dioxide_threshold = 300

# Capping extreme outliers
df_capped = df.copy()  # Create a copy of the original dataframe to keep the original intact
df_capped['residual_sugar'] = df_capped['residual_sugar'].apply(lambda x: min(x, residual_sugar_threshold))
df_capped['total_sulfur_dioxide'] = df_capped['total_sulfur_dioxide'].apply(lambda x: min(x, total_sulfur_dioxide_threshold))

# Display the dataframe with capped values
print(df_capped.head())

# Optionally, save the capped dataframe to a new CSV file
df_capped.to_csv('capped_wine_data.csv', index=False)



# ML model training and evaluation
# Load capped dataset
wine_data = pd.read_csv('capped_wine_data.csv')

# Bin 'quality' into categories
def quality_category(q):
    if q <= 5:
        return 'Low'
    elif q <= 7:
        return 'Medium'
    else:
        return 'High'

wine_data['quality_category'] = wine_data['quality'].apply(quality_category)

# Quick check (will print counts in console)
print(wine_data['quality_category'].value_counts())

# Features and target
X = wine_data.drop(['quality', 'quality_category'], axis=1)
y = wine_data['quality_category']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest Model
rf_model = RandomForestClassifier(n_estimators=200, random_state=42)
rf_model.fit(X_train, y_train)

# Predictions
y_pred = rf_model.predict(X_test)

# Results
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nAccuracy Score:", accuracy_score(y_test, y_pred))

# Feature Importance
importances = rf_model.feature_importances_
feature_names = X.columns
feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)

plt.figure(figsize=(10,5))
sns.barplot(x=feat_imp, y=feat_imp.index)
plt.title('Feature Importance')
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.tight_layout()
plt.show()