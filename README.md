5. import pandas as pd
import numpy as np
df = pd.read_csv('Healthcare-Diabetes.csv')
df.head()
print(df.isnull().sum())
df.info()
df = pd.read_csv('Healthcare-Diabetes.csv')
zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[zero_cols] = df[zero_cols].replace(0, np.nan)
print("Step 1  Replaced 0s with NaN in:", zero_cols)
for col in zero_cols:
    median_val = df[col].median()
    df[col] = df[col].fillna(median_val)
print(" Imputed missing values with median.")
print(df[zero_cols].isnull().sum())
df['AgeGroup'] = pd.cut(df['Age'], 
                        bins=[20, 30, 40, 50, 60, 70, 80], 
                        labels=['21-30', '31-40', '41-50', '51-60', '61-70', '71-80'])
print("\nStep 2  Created AgeGroup buckets:")
print(df['AgeGroup'].value_counts())
df = pd.get_dummies(df, columns=['AgeGroup'], drop_first=True)
print("\n One-hot encoded AgeGroup:")
print([col for col in df.columns if 'AgeGroup' in col])
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler, LabelEncoder
df['BMI_Glucose'] = df['BMI'] * df['Glucose']
print("\nCreated interaction feature BMI_Glucose.")
num_cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
            'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'BMI_Glucose']

scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])
print("\nStep 5: Normalized numerical features.")
print(df[num_cols].describe().T)
original_bmi = df['BMI'] * scaler.scale_[num_cols.index('BMI')] + scaler.mean_[num_cols.index('BMI')]
df['Obese'] = (original_bmi > 30).astype(int)
print("\nStep 6  Created 'Obese' binary feature (BMI > 30):")
print(df['Obese'].value_counts())
X = df.drop(columns='Outcome')
y = df['Outcome']
X = pd.get_dummies(X, drop_first=True)
X = X.dropna()
y = y.loc[X.index]
X = X.astype(np.float64)
assert X.dtypes.nunique() == 1 and X.dtypes.unique()[0] == np.float64, "Non-numeric types remain!"
pip install xgboost lightgbm scikit-learn
6.import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, stratify=y_temp, random_state=42)
X_train_np = X_train.values
X_val_np = X_val.values
X_test_np = X_test.values
models = {
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
    "LightGBM": LGBMClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42)}
for name, model in models.items():
    model.fit(X_train_np, y_train)
    val_preds = model.predict(X_val_np)
    test_preds = model.predict(X_test_np)
    val_acc = accuracy_score(y_val, val_preds)
    test_acc = accuracy_score(y_test, test_preds)
    print(f"\n {name}")
    print(f"Validation Accuracy: {val_acc:.4f}")
    print(f"Test Accuracy      : {test_acc:.4f}")
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
for name, model in models.items():
    test_preds = model.predict(X_test_np)
    cm = confusion_matrix(y_test, test_preds)
    print(f"\n Confusion Matrix for {name}:\n{cm}")
    report = classification_report(y_test, test_preds)
    print(f"\n Classification Report for {name}:\n{report}")
    plt.figure(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Predicted: No', 'Predicted: Yes'],
                yticklabels=['Actual: No', 'Actual: Yes'])
    plt.title(f'{name} - Confusion Matrix')
    plt.xlabel('Prediction')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.show()






