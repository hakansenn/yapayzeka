import pandas as pd # pandas veri manipülasyon modülü

from sklearn.tree import DecisionTreeClassifier # Karar Ağacı Sınıflandırıcı

from sklearn.model_selection import train_test_split # Eğitim ve Test için verileri ayrıştıran fonksiyon

from sklearn import metrics # Sınıflandırmanın ne kadar başarılı olduğunu gösteren ölçütler

from sklearn.metrics import classification_report, confusion_matrix


veriseti= pd.read_csv("spambase.data")
veriseti.columns=["Column"+str(i) for i in range(1, 59)]
print(veriseti.head())
# X icerisinde öznitelikler
# y içerisinde ssınıflar var
X = veriseti.drop(columns=['Column58'], axis=1) 
y = veriseti['Column58']

print("Siniflardaki eleman sayilari: ",veriseti['Column58'].value_counts())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1) # 80% training and 20% test


# Create Decision Tree classifer object
clf = DecisionTreeClassifier()
#clf = DecisionTreeClassifier(class_weight={'g': 0.6, 'b': 0.4})

# Eğitim setini ver. Fonksiyonu uydursun.
# Fonksiyon bir ağaç
clf = clf.fit(X_train,y_train)

# Test yapıyoruz. Test verisini ver.
# Tahmin ettiği sınıfları döndürsün.
# y_pred = tahmin edilen sınıflar
y_pred = clf.predict(X_test)


# Model Accuracy, how often is the classifier correct?
print(confusion_matrix(y_test, y_pred,labels=[1,0]))

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Balanced Accuracy:",metrics.balanced_accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

