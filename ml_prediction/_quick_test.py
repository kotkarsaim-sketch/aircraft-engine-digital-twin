import sys, time, joblib
sys.path.insert(0, r"C:\pylibs")
sys.path.insert(0, r"C:\twin\ml_prediction\src")
from data.dataset import C172PDataset, load_config
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np

cfg = load_config("config.yaml")
ds = C172PDataset(cfg)
raw = ds.generate_raw_data()
raw = ds.compute_rul(raw)
raw = ds.engineer_features(raw)
train, val, test = ds.split_by_seed(raw)
ds.fit_scalers(train)

# Classifier
X_train_clf, y_train_clf = ds.prepare_classifier_data(train)
X_train_clf_s = ds.scaler.transform(X_train_clf)
y_train_clf_enc = ds.label_encoder.transform(y_train_clf)

print("Training classifier...")
clf = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
clf.fit(X_train_clf_s, y_train_clf_enc)
print("Classifier done")

# RUL
window = cfg["features"]["default_rul_window"]
X_train_rul, y_train_rul = ds.create_rolling_features(train, window)
X_train_rul_s = ds.rul_scaler.transform(X_train_rul)

print("Training RUL...")
rul = GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, max_depth=4, random_state=42)
rul.fit(X_train_rul_s, y_train_rul)
print("RUL done")

# Test eval
X_test_clf, y_test_clf = ds.prepare_classifier_data(test)
X_test_clf_s = ds.scaler.transform(X_test_clf)
y_test_clf_enc = ds.label_encoder.transform(y_test_clf)
X_test_rul, y_test_rul = ds.create_rolling_features(test, window)
X_test_rul_s = ds.rul_scaler.transform(X_test_rul)

from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error
y_pred_clf = clf.predict(X_test_clf_s)
y_pred_rul = rul.predict(X_test_rul_s)
print("Accuracy: {:.4f}".format(accuracy_score(y_test_clf_enc, y_pred_clf)))
print("F1 macro: {:.4f}".format(f1_score(y_test_clf_enc, y_pred_clf, average="macro")))
print("MAE: {:.4f}h".format(mean_absolute_error(y_test_rul, y_pred_rul)))
print("RMSE: {:.4f}h".format(np.sqrt(np.mean((y_test_rul - y_pred_rul)**2))))