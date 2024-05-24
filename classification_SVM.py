from sklearn.svm import SVC
import mglearn as mg
import matplotlib.pyplot as plt

X, y = mg.tools.make_handcrafted_dataset()
# SVMモデルをカーネルトリックを用いて構築
svm = SVC(kernel='rbf', C=10, gamma=0.1).fit(X, y)
mg.plots.plot_2d_separator(svm, X, eps=.5)
mg.discrete_scatter(X[:, 0], X[:, 1], y)
# サポートベクタをプロット
sv = svm.support_vectors_
# サポートベクタのクラスラベルはサポートベクタマーカーの色で示す
sv_labels = svm.dual_coef_.ravel() > 0
mg.discrete_scatter(sv[:, 0], sv[:, 1], sv_labels, s=15, markeredgewidth=3)
plt.show()