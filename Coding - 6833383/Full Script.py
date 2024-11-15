# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 15:53:21 2024

@author: sv00633
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score, precision_score, mean_absolute_error, r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier
import seaborn as sns


file_path= r'C:\Users\sv00633\OneDrive - University of Surrey\dissertation\Smart Meters SME.csv'
Raw_DataFrame = pd.read_csv(file_path)
# Drop any unnamed columns
Raw_DataFrame = Raw_DataFrame.loc[:, ~Raw_DataFrame.columns.str.contains('^Unnamed')]

# Rename the columns: Keep 'Id' and 'Industry', rename the rest starting from Q1 to Q42
renamed_columns = ['Id', 'Industry'] + [f"Q{i}" for i in range(1, Raw_DataFrame.shape[1] - 1)]
Raw_DataFrame.columns = renamed_columns

# Convert non-numeric columns to numeric using ASCII sum
def convert_to_numeric(df):
    for column in df.columns:
        if df[column].dtype == 'object':
            df[column] = df[column].apply(lambda x: sum([ord(char) for char in str(x)]) if pd.notna(x) else np.nan)
    return df

Raw_DataFrame = convert_to_numeric(Raw_DataFrame)

# Impute missing values with the mean
imputer = SimpleImputer(strategy='mean')
Raw_DataFrame_imputed = pd.DataFrame(imputer.fit_transform(Raw_DataFrame), columns=Raw_DataFrame.columns)


# Drop the 'Id' column
Raw_DataFrame_imputed = Raw_DataFrame_imputed.drop(columns=['Id'])
print(Raw_DataFrame_imputed.columns)

# Exploratory Data Analysis (EDA)

# 1. Histograms of the first 15 attributes
plt.figure(figsize=(15, 15)) 
for i, column in enumerate(Raw_DataFrame_imputed.columns[2:17], 1):  # Skip 'Id' and 'Industry'
    plt.subplot(4, 4, i)
    Raw_DataFrame_imputed[column].hist()
    plt.title(column)
plt.tight_layout()
plt.show()

# 2. Boxplot to visualize the outliers in numeric variables
plt.figure(figsize=(15, 8))
Raw_DataFrame_imputed.boxplot()
plt.title('Distribution of all Numeric Variables')
plt.xticks(rotation='vertical')
plt.show()

# Remove variables with low standard deviation
std_devs = Raw_DataFrame_imputed.std()
low_std_columns = std_devs[std_devs < 0.1].index
Raw_DataFrame_imputed = Raw_DataFrame_imputed.drop(columns=low_std_columns)

# 3. Scatter plot for bivariate analysis
x_col = 'Q20'  # Target variable
y_col = 'Q12'  # Example feature
plt.figure(figsize=(10, 6))
plt.scatter(Raw_DataFrame_imputed[x_col], Raw_DataFrame_imputed[y_col], alpha=0.7)
plt.xlabel(x_col)
plt.ylabel(y_col)
plt.title(f'Bivariate Scatter Plot - {x_col} vs {y_col}')
plt.grid(True)
plt.show()

# 4. Correlation matrix plot
plt.figure(figsize=(15, 15))
sns.heatmap(Raw_DataFrame_imputed.corr(), annot=True, vmin=-1, vmax=1, cmap='coolwarm')
plt.title('Correlation Matrix for All Features')
plt.show()

# 5. Outlier removal based on IQR
plt.figure(figsize=(15, 8))
sns.boxplot(data=Raw_DataFrame_imputed, palette="Set3")
plt.title('Boxplot Before Outlier Removal (Based on IQR)')
plt.xticks(rotation=90)
plt.show()

# Outlier removal with a less stringent IQR method
Q1 = Raw_DataFrame_imputed.quantile(0.25)
Q3 = Raw_DataFrame_imputed.quantile(0.75)
IQR = Q3 - Q1

# Use a larger multiplier (e.g., 3) to reduce the number of removed rows
Raw_DataFrame_filtered = Raw_DataFrame_imputed[~((Raw_DataFrame_imputed < (Q1 - 5 * IQR)) | (Raw_DataFrame_imputed > (Q3 + 3 * IQR))).any(axis=1)]

# Boxplot after less stringent outlier removal
plt.figure(figsize=(15, 8))
sns.boxplot(data=Raw_DataFrame_filtered, palette="Set3")
plt.title('Boxplot After Outlier Removal (Based on IQR)')
plt.xticks(rotation=90)
plt.show()

# 6. Summary Statistics
print('Descriptive Statistics Summary:')
print(Raw_DataFrame_filtered.iloc[:, :7].describe())
print(Raw_DataFrame_filtered.describe())

# Get the total number of observations and variables
total_observations = Raw_DataFrame_filtered.shape[0]
total_variables = Raw_DataFrame_filtered.shape[1]

# Calculate the number of observations for training (70%) and testing (30%) sets
train_observations = int(0.7 * total_observations)
test_observations = total_observations - train_observations

print(f"Total Observations: {total_observations}")
print(f"Total Variables: {total_variables}")
print(f"Training Observations: {train_observations}")
print(f"Testing Observations: {test_observations}")


# Standardizing the data
scaler = StandardScaler()
features_scaled = scaler.fit_transform(Raw_DataFrame_filtered.drop(columns=['Q20']))

# Example of interpreting standardized data
print("First 5 rows of standardized features:")
print(pd.DataFrame(features_scaled, columns=Raw_DataFrame_filtered.columns[:-1]).head())

# 7. Decision Tree

# Encode categorical labels as integers
print('DECISION TREE')
label_encoder = LabelEncoder()
target = label_encoder.fit_transform(Raw_DataFrame_filtered['Q20'])
print(target)
# If Q20 is already numerical, directly assign it as the target
# target = Raw_DataFrame_filtered['Q20'].values  # No need for LabelEncoder

# Standardize features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(Raw_DataFrame_filtered.drop(columns=['Q20']))

# Feature selection using RFE with RandomForestClassifier
rf = RandomForestClassifier(random_state=42)
selector = RFE(rf, n_features_to_select=10, step=1)
features_selected = selector.fit_transform(features_scaled, target)

# Initialize the Decision Tree classifier
dt = DecisionTreeClassifier(criterion='gini', max_depth=5, min_samples_leaf=5, min_samples_split=10, random_state=42)

# Function to perform cross-validation and display results
def evaluate_model_with_cv_dt(model, X, y, cv_folds=5):
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    accuracy_scores, f1_scores, recall_scores, precision_scores = [], [], [], []
    for train_index, test_index in cv.split(X, y):
        X_train_fold, X_test_fold = X[train_index], X[test_index]
        y_train_fold, y_test_fold = y[train_index], y[test_index]
        model.fit(X_train_fold, y_train_fold)
        y_pred_fold = model.predict(X_test_fold)
        accuracy_scores.append(accuracy_score(y_test_fold, y_pred_fold))
        f1_scores.append(f1_score(y_test_fold, y_pred_fold, average='macro'))
        recall_scores.append(recall_score(y_test_fold, y_pred_fold, average='macro'))
        precision_scores.append(precision_score(y_test_fold, y_pred_fold, average='macro'))

    print("Overall Performance Across Folds with Decision Tree:")
    print(f"Mean Accuracy: {np.mean(accuracy_scores) * 100:.2f}%")
    print(f"Mean F1 Score: {np.mean(f1_scores):.2f}")
    print(f"Mean Recall: {np.mean(recall_scores):.2f}")
    print(f"Mean Precision: {np.mean(precision_scores):.2f}")

# Evaluate the model using 5-fold cross-validation
evaluate_model_with_cv_dt(dt, features_selected, target, cv_folds=5)

# Train-test split (70-30)
X_train, X_test, y_train, y_test = train_test_split(features_selected, target, test_size=0.3, random_state=42, stratify=target)

# Train the Decision Tree model on the training set
dt.fit(X_train, y_train)

# Predict and evaluate the model on the test set
y_pred_dt = dt.predict(X_test)

# Model evaluation metrics
accuracy_dt = accuracy_score(y_test, y_pred_dt)
conf_matrix_dt = confusion_matrix(y_test, y_pred_dt)
f1_dt = f1_score(y_test, y_pred_dt, average='macro')
recall_dt = recall_score(y_test, y_pred_dt, average='macro')
precision_dt = precision_score(y_test, y_pred_dt, average='macro')

print(f"Decision Tree Classifier Accuracy: {accuracy_dt * 100:.3f}%")
print("Confusion Matrix:\n", conf_matrix_dt)
print(f"F1 Score: {f1_dt:.3f}")
print(f"Recall Score: {recall_dt:.3f}")
print(f"Precision Score: {precision_dt:.3f}")

# Plotting the model evaluation metrics as a bar chart
metrics = {
    'accuracy_dt': accuracy_dt,
    'f1_dt': f1_dt,
    'recall_dt': recall_dt,
    'precision_dt': precision_dt
}

metrics_percent = {k: v * 100 for k, v in metrics.items()}

plt.figure(figsize=(10, 6))
plt.bar(metrics_percent.keys(), metrics_percent.values(), color=['skyblue', 'lightgreen', 'lightcoral', 'lightgoldenrodyellow'])
plt.title('Model Evaluation Metrics')
plt.ylabel('Percentage (%)')
plt.ylim(0, 100)
plt.show()

# Visualization of Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_dt, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

# Plotting True Positives and False Positives
tp = conf_matrix_dt[1, 1]  # True Positives
fp = conf_matrix_dt[0, 1]  # False Positives
tn = conf_matrix_dt[0, 0]  # True Negatives
fn = conf_matrix_dt[1, 0]  # False Negatives

# Plotting bar chart for True Positives and False Positives
plt.figure(figsize=(8, 6))
sns.barplot(x=['True Positives', 'False Positives', 'True Negatives', 'False Negatives'], y=[tp, fp, tn, fn])
plt.title('True Positives, False Positives, True Negatives, and False Negatives')
plt.show()


# For multiclass, we'll sum over each class to get a binary-like TP, FP, TN, FN count for simplicity
TP = np.diag(conf_matrix_dt)  # True positives are the diagonal values
FP = np.sum(conf_matrix_dt, axis=0) - TP  # False positives are the sum of columns minus TP
FN = np.sum(conf_matrix_dt, axis=1) - TP  # False negatives are the sum of rows minus TP
TN = np.sum(conf_matrix_dt) - (FP + FN + TP)  # True negatives are the remaining values

# Display the values
print(f"True Positives (TP): {TP}")
print(f"False Positives (FP): {FP}")
print(f"True Negatives (TN): {TN}")
print(f"False Negatives (FN): {FN}")

# Calculate each metric
Accuracy = 100.0 * (TP + TN) / (TP + FP + FN + TN)
TPR = 100.0 * (TP / (TP + FN))  # True Positive Rate (Sensitivity)
FPR = 100.0 * (FP / (FP + TN))  # False Positive Rate
TNR = 100.0 * (TN / (FP + TN))  # True Negative Rate (Specificity)

# Matthews Correlation Coefficient (MCC)
MCC = ((TP * TN) - (FP * FN)) / np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))

# Handle cases where the denominator in MCC is 0 to avoid division by zero
MCC = np.where(np.isnan(MCC), 0, MCC)  # Replace NaNs with 0

# Print the metrics

# Display the values
print(f"True Positives (TP): {TP}")
print(f"False Positives (FP): {FP}")
print(f"True Negatives (TN): {TN}")
print(f"False Negatives (FN): {FN}")
print(f"Accuracy: {Accuracy}")
print(f"True Positive Rate (TPR): {TPR}")
print(f"False Positive Rate (FPR): {FPR}")
print(f"True Negative Rate (TNR): {TNR}")
print(f"Matthews Correlation Coefficient (MCC): {MCC}")


# Mapping encoded classes back to their original labels
class_labels = label_encoder.inverse_transform(np.unique(target))
# class_labels = np.unique(target)

# Print what each class constitutes of
for i, label in enumerate(class_labels):
    print(f"Class {i}: {label}")

# The rest of your code follows


labels = [f'Class {i} ({class_labels[i]})' for i in range(len(TP))]
x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars



# # Plotting True Positives and False Positives
# labels = [f'Class {i}' for i in range(len(TP))]
# x = np.arange(len(labels))  # the label locations
# width = 0.35  # the width of the bars



# Plotting True Positives and False Positives

fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width/2, TP, width, label='True Positives')
rects2 = ax.bar(x + width/2, FP, width, label='False Positives')

# Add some text for labels, title and axes ticks
ax.set_xlabel('Classes')
ax.set_ylabel('Count')
ax.set_title('Decison Tree (True Positives and False Positives by Class)')
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45)
ax.legend()

fig.tight_layout()
plt.show()

# Plotting True Negatives and False Negatives
fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width/2, TN, width, label='True Negatives')
rects2 = ax.bar(x + width/2, FN, width, label='False Negatives')

# Add some text for labels, title and axes ticks
ax.set_xlabel('Classes')
ax.set_ylabel('Count')
ax.set_title('Decison Tree (True Negatives and False Negatives by Class)')
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45)
ax.legend()

fig.tight_layout()
plt.show()

from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

# Assuming target has the original labels encoded, we need to binarize the labels
n_classes = len(np.unique(target))
y_test_binarized = label_binarize(y_test, classes=np.unique(target))
y_score = dt.predict_proba(X_test)

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_binarized.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Plot all ROC curves
plt.figure(figsize=(10, 8))
colors = plt.cm.get_cmap('tab10', n_classes)

for i, color in zip(range(n_classes), colors.colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label=f'ROC curve of class {class_labels[i]} (AUC = {roc_auc[i]:0.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2)  # Diagonal line (random classifier)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Decision Tree (Receiver Operating Characteristic (ROC) Curves - Multiclass)')
plt.legend(loc="lower right")
plt.show()

print('SVM')
# Standardize features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(Raw_DataFrame_filtered.drop(columns=['Q20']))

# Feature selection using RFE with RandomForestClassifier
rf = RandomForestClassifier(random_state=42)
selector = RFE(rf, n_features_to_select=10, step=1)
features_selected = selector.fit_transform(features_scaled, target)


# Initialize the Support Vector Machine (SVM)
svm = SVC(kernel='linear', random_state=42)

# Function to perform cross-validation and display results in a table format
def evaluate_model_with_cv_svm_table(model, X, y, cv_folds=5):
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    results = {
        'Fold': [],
        'Accuracy': [],
        'F1 Score': [],
        'Recall': [],
        'Precision': []
    }
    
    fold_number = 1
    for train_index, test_index in cv.split(X, y):
        X_train_fold, X_test_fold = X[train_index], X[test_index]
        y_train_fold, y_test_fold = y[train_index], y[test_index]
        model.fit(X_train_fold, y_train_fold)
        y_pred_fold = model.predict(X_test_fold)

        # Append results for each fold
        results['Fold'].append(f"Fold {fold_number}")
        results['Accuracy'].append(accuracy_score(y_test_fold, y_pred_fold))
        results['F1 Score'].append(f1_score(y_test_fold, y_pred_fold, average='macro'))
        results['Recall'].append(recall_score(y_test_fold, y_pred_fold, average='macro'))
        results['Precision'].append(precision_score(y_test_fold, y_pred_fold, average='macro'))
        fold_number += 1

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    results_df['Accuracy'] = results_df['Accuracy'] * 100  # Convert to percentage
    results_df['F1 Score'] = results_df['F1 Score'] * 100  # Convert to percentage
    results_df['Recall'] = results_df['Recall'] * 100  # Convert to percentage
    results_df['Precision'] = results_df['Precision'] * 100  # Convert to percentage

    # Print the results as a table
    print("Cross-Validation Results by Fold:")
    print(results_df.to_string(index=False))

# Evaluate the model using 5-fold cross-validation and display results in a table
evaluate_model_with_cv_svm_table(svm, features_selected, target, cv_folds=5)

# Train-test split (70-30)
X_train, X_test, y_train, y_test = train_test_split(features_selected, target, test_size=0.3, random_state=42, stratify=target)

# Train the SVM model on the training set
svm.fit(X_train, y_train)

# Predict and evaluate the model on the test set
y_pred_svm = svm.predict(X_test)

# Model evaluation metrics
accuracy_svm = accuracy_score(y_test, y_pred_svm)
conf_matrix_svm = confusion_matrix(y_test, y_pred_svm)
f1_svm = f1_score(y_test, y_pred_svm, average='macro')
recall_svm = recall_score(y_test, y_pred_svm, average='macro')
precision_svm = precision_score(y_test, y_pred_svm, average='macro')

print(f"Support Vector Machine Accuracy: {accuracy_svm * 100:.3f}%")
print("Confusion Matrix:\n", conf_matrix_svm)
print(f"F1 Score: {f1_svm:.3f}")
print(f"Recall Score: {recall_svm:.3f}")
print(f"Precision Score: {precision_svm:.3f}")

# Plotting the model evaluation metrics as a bar chart
metrics = {
    'accuracy_svm': accuracy_svm,
    'f1_svm': f1_svm,
    'recall_svm': recall_svm,
    'precision_svm': precision_svm
}

metrics_percent = {k: v * 100 for k, v in metrics.items()}

plt.figure(figsize=(10, 6))
plt.bar(metrics_percent.keys(), metrics_percent.values(), color=['skyblue', 'lightgreen', 'lightcoral', 'lightgoldenrodyellow'])
plt.title('Model Evaluation Metrics')
plt.ylabel('Percentage (%)')
plt.ylim(0, 100)
plt.show()

# Visualization of Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_svm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()


# Plotting True Positives and False Positives
# Get the list of unique classes that were present during training
present_classes = np.unique(y_train)

# Get the original labels for the classes that were seen during training
class_labels_present = label_encoder.inverse_transform(present_classes)

# Create a mapping from the class index to the label
class_labels = {i: label for i, label in zip(present_classes, class_labels_present)}

# Ensure the TP, FP, TN, and FN arrays match the number of present classes
TP = np.diag(conf_matrix_svm)  # True positives for present classes
FP = np.sum(conf_matrix_svm, axis=0) - TP  # False positives
FN = np.sum(conf_matrix_svm, axis=1) - TP  # False negatives
TN = np.sum(conf_matrix_svm) - (FP + FN + TP)  # True negatives
 
# Calculate each metric
Accuracy = 100.0 * (TP + TN) / (TP + FP + FN + TN)
TPR = 100.0 * (TP / (TP + FN))  # True Positive Rate (Sensitivity)
FPR = 100.0 * (FP / (FP + TN))  # False Positive Rate
TNR = 100.0 * (TN / (FP + TN))  # True Negative Rate (Specificity)

# Matthews Correlation Coefficient (MCC)
MCC = ((TP * TN) - (FP * FN)) / np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))

# Handle cases where the denominator in MCC is 0 to avoid division by zero
MCC = np.where(np.isnan(MCC), 0, MCC)  # Replace NaNs with 0

# Print the metrics

# Display the values
print(f"True Positives (TP): {TP}")
print(f"False Positives (FP): {FP}")
print(f"True Negatives (TN): {TN}")
print(f"False Negatives (FN): {FN}")
print(f"Accuracy: {Accuracy}")
print(f"True Positive Rate (TPR): {TPR}")
print(f"False Positive Rate (FPR): {FPR}")
print(f"True Negative Rate (TNR): {TNR}")
print(f"Matthews Correlation Coefficient (MCC): {MCC}")


# Plotting bar chart for True Positives and False Positives
labels = [f'Class {i} ({class_labels[i]})' for i in present_classes]  # Only label present classes
x = np.arange(len(present_classes))  # Adjust x locations for present classes
width = 0.35  # Width of the bars

fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width/2, TP, width, label='True Positives')
rects2 = ax.bar(x + width/2, FP, width, label='False Positives')

ax.set_xlabel('Classes')
ax.set_ylabel('Count')
ax.set_title('True Positives and False Positives by Class')
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45)
ax.legend()

fig.tight_layout()
plt.show()

# Plotting True Negatives and False Negatives
fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width/2, TN, width, label='True Negatives')
rects2 = ax.bar(x + width/2, FN, width, label='False Negatives')

ax.set_xlabel('Classes')
ax.set_ylabel('Count')
ax.set_title('True Negatives and False Negatives by Class')
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45)
ax.legend()

fig.tight_layout()
plt.show()



from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc

# Binarize the labels for ROC curve computation
# Ensure the SVM was trained with probability=True
svm = SVC(kernel='linear', probability=True, random_state=42)

# Train the SVM model on the training set
svm.fit(X_train, y_train)

# Predict probabilities on the test set
y_score = svm.predict_proba(X_test)

# Binarize the labels for ROC curve computation
y_test_binarized = label_binarize(y_test, classes=present_classes)

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()

for i, class_idx in enumerate(present_classes):
    fpr[class_idx], tpr[class_idx], _ = roc_curve(y_test_binarized[:, i], y_score[:, i])
    roc_auc[class_idx] = auc(fpr[class_idx], tpr[class_idx])

# Compute micro-average ROC curve and ROC area (if there are multiple classes)
if len(present_classes) > 1:
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test_binarized.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Plot all ROC curves
plt.figure(figsize=(10, 8))
colors = plt.cm.get_cmap('tab10', len(present_classes))

for i, color in zip(present_classes, colors.colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label=f'ROC curve of class {class_labels[i]} (AUC = {roc_auc[i]:0.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2)  # Diagonal line (random classifier)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('SVM (Receiver Operating Characteristic (ROC) Curves - Multiclass)')
plt.legend(loc="lower right")
plt.show()


print('ANN')
# Encode categorical labels as integers
label_encoder = LabelEncoder()
target = label_encoder.fit_transform(Raw_DataFrame_filtered['Q20'])

# Standardize features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(Raw_DataFrame_filtered.drop(columns=['Q20']))

# Feature selection using RFE with RandomForestClassifier
rf = RandomForestClassifier(random_state=42)
selector = RFE(rf, n_features_to_select=10, step=1)
features_selected = selector.fit_transform(features_scaled, target)


# Initialize the Artificial Neural Network (ANN) with reduced complexity
ann = MLPClassifier(hidden_layer_sizes=(10, 10), max_iter=1000, random_state=42)

# Function to perform cross-validation and display results
def evaluate_model_with_cv_ann(model, X, y, cv_folds=5):
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    accuracy_scores, f1_scores, recall_scores, precision_scores = [], [], [], []
    for train_index, test_index in cv.split(X, y):
        X_train_fold, X_test_fold = X[train_index], X[test_index]
        y_train_fold, y_test_fold = y[train_index], y[test_index]
        model.fit(X_train_fold, y_train_fold)
        y_pred_fold = model.predict(X_test_fold)
        accuracy_scores.append(accuracy_score(y_test_fold, y_pred_fold))
        f1_scores.append(f1_score(y_test_fold, y_pred_fold, average='macro'))
        recall_scores.append(recall_score(y_test_fold, y_pred_fold, average='macro'))
        precision_scores.append(precision_score(y_test_fold, y_pred_fold, average='macro'))

    print("Overall Performance Across Folds with ANN:")
    print(f"Mean Accuracy: {np.mean(accuracy_scores) * 100:.2f}%")
    print(f"Mean F1 Score: {np.mean(f1_scores):.2f}")
    print(f"Mean Recall: {np.mean(recall_scores):.2f}")
    print(f"Mean Precision: {np.mean(precision_scores):.2f}")

# Evaluate the model using 5-fold cross-validation
evaluate_model_with_cv_ann(ann, features_selected, target, cv_folds=5)

# Train-test split (70-30)
X_train, X_test, y_train, y_test = train_test_split(features_selected, target, test_size=0.3, random_state=42, stratify=target)

# Train the ANN model on the training set
ann.fit(X_train, y_train)

# Predict and evaluate the model on the test set
y_pred_ann = ann.predict(X_test)

# Get the list of classes in the original label encoding
class_labels = label_encoder.classes_

# Compute the confusion matrix
conf_matrix_ann = confusion_matrix(y_test, y_pred_ann, labels=range(5))

# Ensure it's a 5x5 confusion matrix
if conf_matrix_ann.shape != (5, 5):
    full_conf_matrix = np.zeros((5, 5), dtype=int)
    min_dim = min(conf_matrix_ann.shape[0], conf_matrix_ann.shape[1])
    full_conf_matrix[:min_dim, :min_dim] = conf_matrix_ann[:min_dim, :min_dim]
    conf_matrix_ann = full_conf_matrix

print("Confusion Matrix (5x5):\n", conf_matrix_ann)

# Model evaluation metrics
accuracy_ann = accuracy_score(y_test, y_pred_ann)
f1_ann = f1_score(y_test, y_pred_ann, average='macro')
recall_ann = recall_score(y_test, y_pred_ann, average='macro')
precision_ann = precision_score(y_test, y_pred_ann, average='macro')

print(f"Artificial Neural Network Accuracy: {accuracy_ann * 100:.3f}%")
print(f"F1 Score: {f1_ann:.3f}")
print(f"Recall Score: {recall_ann:.3f}")
print(f"Precision Score: {precision_ann:.3f}")


# Plotting the model evaluation metrics as a bar chart
metrics = {
    'accuracy_ann': accuracy_ann,
    'f1_ann': f1_ann,
    'recall_ann': recall_ann,
    'precision_ann': precision_ann
}

metrics_percent = {k: v * 100 for k, v in metrics.items()}

plt.figure(figsize=(10, 6))
plt.bar(metrics_percent.keys(), metrics_percent.values(), color=['skyblue', 'lightgreen', 'lightcoral', 'lightgoldenrodyellow'])
plt.title('Model Evaluation Metrics')
plt.ylabel('Percentage (%)')
plt.ylim(0, 100)
plt.show()

# Visualization of Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_ann, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()
# Plotting True Positives and False Positives
TP = np.diag(conf_matrix_ann)  # True positives are the diagonal values
FP = np.sum(conf_matrix_ann, axis=0) - TP  # False positives are the sum of columns minus TP
FN = np.sum(conf_matrix_ann, axis=1) - TP  # False negatives are the sum of rows minus TP
TN = np.sum(conf_matrix_ann) - (FP + FN + TP)  # True negatives are the remaining values 
# Calculate each metric
Accuracy = 100.0 * (TP + TN) / (TP + FP + FN + TN)
TPR = 100.0 * (TP / (TP + FN))  # True Positive Rate (Sensitivity)
FPR = 100.0 * (FP / (FP + TN))  # False Positive Rate
TNR = 100.0 * (TN / (FP + TN))  # True Negative Rate (Specificity)

# Matthews Correlation Coefficient (MCC)
MCC = ((TP * TN) - (FP * FN)) / np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))

# Handle cases where the denominator in MCC is 0 to avoid division by zero
MCC = np.where(np.isnan(MCC), 0, MCC)  # Replace NaNs with 0

# Print the metrics

# Display the values
print(f"True Positives (TP): {TP}")
print(f"False Positives (FP): {FP}")
print(f"True Negatives (TN): {TN}")
print(f"False Negatives (FN): {FN}")
print(f"Accuracy: {Accuracy}")
print(f"True Positive Rate (TPR): {TPR}")
print(f"False Positive Rate (FPR): {FPR}")
print(f"True Negative Rate (TNR): {TNR}")
print(f"Matthews Correlation Coefficient (MCC): {MCC}")

# Plotting bar chart for True Positives and False Positives
labels = [f'Class {i}' for i in range(5)]  # Ensure labels cover all expected classes
x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width/2, TP, width, label='True Positives')
rects2 = ax.bar(x + width/2, FP, width, label='False Positives')

ax.set_xlabel('Classes')
ax.set_ylabel('Count')
ax.set_title('True Positives and False Positives by Class')
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45)
ax.legend()

fig.tight_layout()
plt.show()

# Plotting True Negatives and False Negatives
fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width/2, TN, width, label='True Negatives')
rects2 = ax.bar(x + width/2, FN, width, label='False Negatives')

ax.set_xlabel('Classes')
ax.set_ylabel('Count')
ax.set_title('True Negatives and False Negatives by Class')
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45)
ax.legend()

fig.tight_layout()
plt.show()

# ROC Curve for the ANN
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np

# Binarize the labels for ROC curve computation
# We binarize only for the classes that are actually present in the y_test
present_classes = np.unique(y_test)
y_test_binarized = label_binarize(y_test, classes=present_classes)
y_score = ann.predict_proba(X_test)

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()

for i, class_idx in enumerate(present_classes):
    fpr[class_idx], tpr[class_idx], _ = roc_curve(y_test_binarized[:, i], y_score[:, i])
    roc_auc[class_idx] = auc(fpr[class_idx], tpr[class_idx])

# Compute micro-average ROC curve and ROC area (if there are multiple classes)
if len(present_classes) > 1:
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test_binarized.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Plot all ROC curves
plt.figure(figsize=(10, 8))
colors = plt.cm.get_cmap('tab10', len(present_classes))

for i, color in zip(present_classes, colors.colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label=f'ROC curve of class {class_labels[i]} (AUC = {roc_auc[i]:0.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2)  # Diagonal line (random classifier)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ANN (Receiver Operating Characteristic (ROC) Curves - Multiclass)')
plt.legend(loc="lower right")
plt.show()


# Initialize models
dt = DecisionTreeClassifier(criterion='gini', max_depth=5, min_samples_leaf=5, min_samples_split=10, random_state=42)
ann = MLPClassifier(hidden_layer_sizes=(10, 10), max_iter=1000, random_state=42)
svm = SVC(kernel='linear', probability=True, random_state=42)

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.model_selection import StratifiedKFold

# Updated Cross-validation function with plotting
def evaluate_model_with_cv_table(model, X, y, model_name, cv_folds=5):
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    results = {
        'Model': [],
        'Fold': [],
        'Accuracy': [],
        'F1 Score': [],
        'Recall': [],
        'Precision': []
    }

    fold_number = 1
    for train_index, test_index in cv.split(X, y):
        X_train_fold, X_test_fold = X[train_index], X[test_index]
        y_train_fold, y_test_fold = y[train_index], y[test_index]
        model.fit(X_train_fold, y_train_fold)
        y_pred_fold = model.predict(X_test_fold)

        # Append results for each fold
        results['Model'].append(model_name)
        results['Fold'].append(f"Fold {fold_number}")
        results['Accuracy'].append(accuracy_score(y_test_fold, y_pred_fold) * 100)
        results['F1 Score'].append(f1_score(y_test_fold, y_pred_fold, average='macro') * 100)
        results['Recall'].append(recall_score(y_test_fold, y_pred_fold, average='macro') * 100)
        results['Precision'].append(precision_score(y_test_fold, y_pred_fold, average='macro') * 100)
        fold_number += 1

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    print(f"Cross-Validation Results by Fold for {model_name}:")
    print(results_df.to_string(index=False))

    # Plotting cross-validation results
    plt.figure(figsize=(10, 6))
    plt.plot(results_df['Fold'], results_df['Accuracy'], marker='o', label='Accuracy', color='skyblue')
    plt.plot(results_df['Fold'], results_df['F1 Score'], marker='s', label='F1 Score', color='lightgreen')
    plt.plot(results_df['Fold'], results_df['Recall'], marker='^', label='Recall', color='lightcoral')
    plt.plot(results_df['Fold'], results_df['Precision'], marker='d', label='Precision', color='lightgoldenrodyellow')

    plt.title(f'Cross-Validation Results for {model_name}')
    plt.xlabel('Fold')
    plt.ylabel('Score (%)')
    plt.ylim(0, 100)
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()

# Evaluate models
evaluate_model_with_cv_table(dt, features_selected, target, "Decision Tree")
evaluate_model_with_cv_table(ann, features_selected, target, "Artificial Neural Network")
evaluate_model_with_cv_table(svm, features_selected, target, "Support Vector Machine")


#Regression Models
print('LR')
target = Raw_DataFrame_imputed['Q20']
features = Raw_DataFrame_imputed.drop(columns=['Q20','Industry'])

# Standardize features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, test_size=0.3, random_state=42)

# Initialize the Linear Regression model
lr = LinearRegression()

# Fit the model on the training data
lr.fit(X_train, y_train)

# Predict on the training set
y_train_pred_lr = lr.predict(X_train)

# Predict on the testing set
y_test_pred_lr = lr.predict(X_test)

# Evaluate the model on the training set
mae_lr_train = mean_absolute_error(y_train, y_train_pred_lr)
mse_lr_train = mean_squared_error(y_train, y_train_pred_lr)
rmse_lr_train = np.sqrt(mse_lr_train)
r2_lr_train = r2_score(y_train, y_train_pred_lr)

# Evaluate the model on the testing set
mae_lr_test = mean_absolute_error(y_test, y_test_pred_lr)
mse_lr_test = mean_squared_error(y_test, y_test_pred_lr)
rmse_lr_test = np.sqrt(mse_lr_test)
r2_lr_test = r2_score(y_test, y_test_pred_lr)

# Print the evaluation metrics
print("\nLinear Regression Performance on Train Set:")
print(f"MAE: {mae_lr_train:.3f}")
print(f"MSE: {mse_lr_train:.3f}")
print(f"RMSE: {rmse_lr_train:.3f}")
print(f"R-squared: {r2_lr_train:.3f}")

print("\nLinear Regression Performance on Test Set:")
print(f"MAE: {mae_lr_test:.3f}")
print(f"MSE: {mse_lr_test:.3f}")
print(f"RMSE: {rmse_lr_test:.3f}")
print(f"R-squared: {r2_lr_test:.3f}")

# Plot Actual vs Predicted for the Test Set
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_test_pred_lr, alpha=0.7)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')  # Identity line
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Values - Linear Regression')
plt.grid(True)
plt.show()

print('RFR')
target = Raw_DataFrame_imputed['Q20']
features = Raw_DataFrame_imputed.drop(columns=['Q20','Industry'])

# Standardize features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Feature selection using RFE with RandomForestRegressor
rf_regressor = RandomForestRegressor(random_state=42)
selector = RFE(rf_regressor, n_features_to_select=10, step=1)
features_selected = selector.fit_transform(features_scaled, target)

# Hyperparameter tuning with GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2']
}

grid_search_rf = GridSearchCV(estimator=RandomForestRegressor(random_state=42), 
                              param_grid=param_grid, 
                              cv=5, 
                              scoring='neg_mean_absolute_error', 
                              n_jobs=-1, 
                              verbose=2)
grid_search_rf.fit(features_selected, target)

# Best parameters found by GridSearchCV
best_params_rf = grid_search_rf.best_params_
print(f"\nBest parameters found by GridSearchCV: {best_params_rf}")

# Refit the RandomForestRegressor with the best parameters
best_rf = grid_search_rf.best_estimator_

# Evaluate the best model on the training and test sets
X_train, X_test, y_train, y_test = train_test_split(features_selected, target, test_size=0.3, random_state=42)

# Train the best model on the training set
best_rf.fit(X_train, y_train)

# Predict and evaluate the model on the train and test sets
y_train_pred_rf = best_rf.predict(X_train)
y_test_pred_rf = best_rf.predict(X_test)

# Model evaluation metrics for train set
mae_rf_train = mean_absolute_error(y_train, y_train_pred_rf)
mse_rf_train = mean_squared_error(y_train, y_train_pred_rf)
rmse_rf_train = np.sqrt(mse_rf_train)
r2_rf_train = r2_score(y_train, y_train_pred_rf)

# Model evaluation metrics for test set
mae_rf_test = mean_absolute_error(y_test, y_test_pred_rf)
mse_rf_test = mean_squared_error(y_test, y_test_pred_rf)
rmse_rf_test = np.sqrt(mse_rf_test)
r2_rf_test = r2_score(y_test, y_test_pred_rf)

print("\nRandom Forest Regressor Performance on Train Set (Best Model):")
print(f"MAE: {mae_rf_train:.3f}")
print(f"MSE: {mse_rf_train:.3f}")
print(f"RMSE: {rmse_rf_train:.3f}")
print(f"R-squared: {r2_rf_train:.3f}")

print("\nRandom Forest Regressor Performance on Test Set (Best Model):")
print(f"MAE: {mae_rf_test:.3f}")
print(f"MSE: {mse_rf_test:.3f}")
print(f"RMSE: {rmse_rf_test:.3f}")
print(f"R-squared: {r2_rf_test:.3f}")

# Plot Actual vs Predicted for the Test Set
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_test_pred_rf, alpha=0.7)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')  # Identity line
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Values - Random Forest Regressor (Best Model)')
plt.grid(True)
plt.show()
 

print('GBR')

# Set Q20 as the target variable
target = Raw_DataFrame_imputed['Q20']
features = Raw_DataFrame_imputed.drop(columns=['Q20', 'Industry'])

# Standardize features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Hyperparameter tuning with GridSearchCV for Gradient Boosting Regressor
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 4, 5],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'subsample': [0.8, 0.9, 1.0]
}

grid_search_gbr = GridSearchCV(estimator=GradientBoostingRegressor(random_state=42), 
                               param_grid=param_grid, 
                               cv=5, 
                               scoring='neg_mean_absolute_error', 
                               n_jobs=-1, 
                               verbose=2)
grid_search_gbr.fit(features_scaled, target)

# Best parameters found by GridSearchCV
best_params_gbr = grid_search_gbr.best_params_
print(f"\nBest parameters found by GridSearchCV: {best_params_gbr}")

# Refit the Gradient Boosting Regressor with the best parameters
best_gbr = grid_search_gbr.best_estimator_

# Evaluate the best model on the training and test sets
X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, test_size=0.3, random_state=42)

# Train the best model on the training set
best_gbr.fit(X_train, y_train)

# Predict and evaluate the model on the train and test sets
y_train_pred_gbr = best_gbr.predict(X_train)
y_test_pred_gbr = best_gbr.predict(X_test)

# Model evaluation metrics for train set
mae_gbr_train = mean_absolute_error(y_train, y_train_pred_gbr)
mse_gbr_train = mean_squared_error(y_train, y_train_pred_gbr)
rmse_gbr_train = np.sqrt(mse_gbr_train)
r2_gbr_train = r2_score(y_train, y_train_pred_gbr)

# Model evaluation metrics for test set
mae_gbr_test = mean_absolute_error(y_test, y_test_pred_gbr)
mse_gbr_test = mean_squared_error(y_test, y_test_pred_gbr)
rmse_gbr_test = np.sqrt(mse_gbr_test)
r2_gbr_test = r2_score(y_test, y_test_pred_gbr)

print("\nGradient Boosting Regressor Performance on Train Set (Best Model):")
print(f"MAE: {mae_gbr_train:.3f}")
print(f"MSE: {mse_gbr_train:.3f}")
print(f"RMSE: {rmse_gbr_train:.3f}")
print(f"R-squared: {r2_gbr_train:.3f}")

print("\nGradient Boosting Regressor Performance on Test Set (Best Model):")
print(f"MAE: {mae_gbr_test:.3f}")
print(f"MSE: {mse_gbr_test:.3f}")
print(f"RMSE: {rmse_gbr_test:.3f}")
print(f"R-squared: {r2_gbr_test:.3f}")

# Plot Actual vs Predicted for the Test Set
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_test_pred_gbr, alpha=0.7)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')  # Identity line
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Values - Gradient Boosting Regressor (Best Model)')
plt.grid(True)
plt.show()


# Assuming X_train, X_test, y_train, y_test have been defined earlier

# Hyperparameter tuning using GridSearchCV for LR, RFR, and GBR

# 1. Linear Regression (No hyperparameters to tune, but included for completeness)
lr = LinearRegression()
lr.fit(X_train, y_train)
y_train_pred_lr = lr.predict(X_train)
y_test_pred_lr = lr.predict(X_test)
best_score_lr = r2_score(y_train, y_train_pred_lr)
best_params_lr = 'N/A'

# 2. Random Forest Regressor
rf_params = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2']
}
rf = GridSearchCV(RandomForestRegressor(random_state=42), rf_params, cv=5)
rf.fit(X_train, y_train)
best_params_rf = rf.best_params_
best_score_rf = rf.best_score_

# 3. Gradient Boosting Regressor
gbr_params = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 4, 5],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'subsample': [0.8, 0.9, 1.0]
}
gbr = GridSearchCV(GradientBoostingRegressor(random_state=42), gbr_params, cv=5)
gbr.fit(X_train, y_train)
best_params_gbr = gbr.best_params_
best_score_gbr = gbr.best_score_

# Create Hyperparameter Table for Best Parameters (Table 3)
hyperparameters_data = {
    'Model': ['Linear Regression', 'RandomForestRegressor', 'GradientBoostingRegressor'],
    'Best Parameters': [best_params_lr, best_params_rf, best_params_gbr],
    'Best Score': [best_score_lr, best_score_rf, best_score_gbr]
}
hyperparameters_df = pd.DataFrame(hyperparameters_data)
print("TABLE 3: Hyperparameter Table for Best Parameters:")
print(hyperparameters_df)

