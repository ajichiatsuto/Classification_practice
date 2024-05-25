from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import numpy as np

cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, stratify=cancer.target, random_state=42)

# 訓練データの特徴量ごとに最小値を計算
min_on_training = X_train.min(axis=0)
# 訓練データの特徴量ごとにレンジ（最大値-最小値）を計算
range_on_training = (X_train - min_on_training).max(axis=0)

# 訓練データから最小値を引き、レンジで割ることで、データのスケールを0から1の間に変換
X_train_scaled = (X_train - min_on_training) / range_on_training

svc = SVC()
svc.fit(X_train_scaled, y_train)

print("Accuracy on training set: {:.3f}".format(svc.score(X_train_scaled, y_train)))

# テストデータも変換
X_test_scaled = (X_test - min_on_training) / range_on_training
print("Accuracy on test set: {:.3f}".format(svc.score(X_test_scaled, y_test)))