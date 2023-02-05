from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Iris veri setini yükleyin
iris_dataset = load_iris()
#Yapay zeka aşağıdaki kodu verdi buna göre de 0,1,2 sütunları ve Accuracy:0,97 verdi.

print(iris_dataset)


X = iris_dataset.data
y = iris_dataset.target

# Veri setini eğitim ve test setlerine ayırın
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)



# Tüm 3 özellik kombinasyonlarını deneyin
combinations = [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]
max_accuracy = 0
best_combination = []
for combination in combinations:
    # KNN sınıflandırıcısını kullanarak eğitin
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(X_train[:, combination], y_train)

    # Test setini kullanarak accuracy değerini hesaplayın
    accuracy = knn.score(X_test[:, combination], y_test)
    if accuracy > max_accuracy:
        max_accuracy = accuracy
        best_combination = combination

# En yüksek accuracy değerini ve en iyi özellik kombinasyonunu yazdırın
print(f"En yüksek accuracy değeri: {max_accuracy:.2f}")
print(f"En iyi özellik kombinasyonu: {best_combination}")

