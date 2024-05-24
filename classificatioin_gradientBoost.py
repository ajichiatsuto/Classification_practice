from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, stratify=cancer.target, random_state=42)

# 学習率0.01、深さ1の勾配ブースティングモデルを構築
gbrt = GradientBoostingClassifier(random_state=0, learning_rate=0.01, max_depth=1)
gbrt.fit(X_train, y_train)

print("Training set score: {:.3f}".format(gbrt.score(X_train, y_train)))
print("Test set score: {:.3f}".format(gbrt.score(X_test, y_test)))