
from clean import clean_data
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix, f1_score



df = pd.read_csv('train_data.csv')
test = pd.read_csv('test_data.csv')

# clean dataset
clean = clean_data(df)
df_no_impact = clean[clean.Impact==0]
df_impact = clean[clean.Impact==1]
df_no_impact = df_no_impact.sample(frac=0.2, replace=False, random_state=1)
df_cleaned = pd.concat([df_impact, df_no_impact])
X = df_cleaned.drop(columns=['Impact'])
y = df_cleaned['Impact']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0, stratify=y)
X_train.head()


# logistic regression
model_lr = LogisticRegression(
    random_state=0,
    penalty = 'l2',
    solver = 'lbfgs',
    C = 1,
    class_weight = 'balanced',
    fit_intercept = False,
    max_iter = 1000,
)

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
scaler = StandardScaler()
# linear regression
pl_lr = make_pipeline(imputer, model_lr)
pl_lr.fit(X_train, y_train)
y_pred_lr = pl_lr.predict(X_test)
y_prob_lr = pl_lr.predict_proba(X_test)
# tn, fp, fn, tp = confusion_matrix(y_test, y_pred_lr).ravel()
score_lr = f1_score(y_test, y_pred_lr)

# random forest
model_rf = RandomForestClassifier(
    n_estimators=10,
    max_depth=10,
)
pl_rf = make_pipeline(imputer, model_rf)
pl_rf.fit(X_train, y_train)
y_pred_rf = pl_rf.predict(X_test)
y_prob_rf = pl_rf.predict_proba(X_test)
score_rf = f1_score(y_test, y_pred_rf)



# bagging
model_bg = BaggingClassifier(DecisionTreeClassifier(), max_samples=0.5, max_features=1.0, n_estimators=20)
pl_bg = make_pipeline(imputer, model_bg)
pl_bg.fit(X_train, y_train)
y_pred_bg = pl_bg.predict(X_test)
score_bg = f1_score(y_test, y_pred_bg)

print("lr: %f , rf: %f, bg: %f" % (score_lr, score_rf, score_bg))

# Voting Classification
ecl = VotingClassifier(estimators=[('lr', model_lr), ('rf', model_rf), ('bg', model_bg)], voting='hard')
pl_ecl = make_pipeline(imputer, ecl)
pl_ecl.fit(X_train, y_train)
y_pred_ecl = pl_ecl.predict(X_test)
score_ecl = f1_score(y_test, y_pred_ecl)
cfs_ecl = confusion_matrix(y_test, y_pred_ecl)
print(cfs_ecl)
print(score_ecl)
