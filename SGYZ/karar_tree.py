from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.tree import export_text
import matplotlib.pyplot as plt
from sklearn.tree import export_text


iris = load_iris()

X, y = iris.data, iris.target

print(X)
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, y)
print(clf.predict([[2., 0.1, 2., 2.]]))

plt.figure(figsize=(12,12))
tree.plot_tree(clf, fontsize=6)
plt.savefig('tree_high_dpi', dpi=600)


r = export_text(clf, feature_names=iris['feature_names'])
print(r)
