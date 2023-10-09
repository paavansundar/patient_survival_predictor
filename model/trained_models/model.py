import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier



"""### Load the dataset"""

# Load dataset
df = pd.read_csv('./model/datasets/heart_failure_clinical_records_dataset.csv')
df.head()

# Shape of dataset
df.shape

"""### Check missing values"""

# Check for missing values
df.isna().sum()

"""### Handle Outliers"""

# Checking for outliers
df.boxplot()
plt.xticks(rotation=90)
plt.show()

# Handing outliers
outlier_colms = ['creatinine_phosphokinase', 'ejection_fraction', 'platelets', 'serum_creatinine', 'serum_sodium']
df1 = df.copy()

def handle_outliers(df, colm):
    '''Change the values of outlier to upper and lower whisker values '''
    q1 = df.describe()[colm].loc["25%"]
    q3 = df.describe()[colm].loc["75%"]
    iqr = q3 - q1
    lower_bound = q1 - (1.5 * iqr)
    upper_bound = q3 + (1.5 * iqr)
    for i in range(len(df)):
        if df.loc[i,colm] > upper_bound:
            df.loc[i,colm]= upper_bound
        if df.loc[i,colm] < lower_bound:
            df.loc[i,colm]= lower_bound
    return df

for colm in outlier_colms:
    df1 = handle_outliers(df1, colm)

# Recheck for outliers
df1.boxplot()
plt.xticks(rotation=90)
plt.show()

"""### Split into training and testing set"""

# Split dataset into training and testing set, considering all features for prediction

X = df1.iloc[:, :-1].values
y = df1['DEATH_EVENT'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, stratify = y, random_state= 123)

X_train[1]

"""### Model Training"""

xgb_clf = XGBClassifier(n_estimators=200, max_depth=4, max_leaves=5, random_state=42)
xgb_clf.fit(X_train, y_train)

"""### Model Performance"""

# Accuracy

train_acc = accuracy_score(y_train, xgb_clf.predict(X_train))
test_acc = accuracy_score(y_test, xgb_clf.predict(X_test))
print("Training accuracy: ", train_acc)
print("Testing accuracy: ", test_acc)

# F1-score

train_f1 = f1_score(y_train, xgb_clf.predict(X_train))
test_f1 = f1_score(y_test, xgb_clf.predict(X_test))
print("Training F1 score: ", train_f1)
print("Testing F1 score: ", test_f1)

"""### Save the trained model"""

# Prepare versioned save file name
save_file_name = "xgboost-model.pkl"

joblib.dump(xgb_clf, save_file_name)