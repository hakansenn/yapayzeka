import pandas as pd
from sklearn.tree import DecisionTreeClassifier # Karar Ağacı Sınıflandırıcı
from sklearn.model_selection import train_test_split # Eğitim ve Test için verileri ayrıştıran fonksiyon
from sklearn import metrics # Sınıflandırmanın ne kadar başarılı olduğunu gösteren ölçütler
from sklearn.metrics import classification_report, confusion_matrix


# load dataset
ionosphere= pd.read_csv("ionosphere_data.csv")


# 
X = ionosphere.drop(columns='column_ai', axis=1) 
y = ionosphere['column_ai']

#print("Siniflardaki eleman sayilari: ",ionosphere['column_ai'].value_counts())
# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1) # 80% training and 20% test
# random_state= 1 her seferinde aynı veriyi kullansın.


# Create Decision Tree classifer object
clf = DecisionTreeClassifier()
#clf = DecisionTreeClassifier(class_weight={'g': 0.6, 'b': 0.4})

# Train Decision Tree Classifer
# Eğitim setini veri fonksiyonu oluştursun.
# fonksiyon bir ağaç
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
# test yapıyoruz test verisini ver tahmin ettiği sınıfları döndürsün
y_pred = clf.predict(X_test)

print("-------------------------------------------------")
# Model Accuracy, how often is the classifier correct?

#confusion matris hata matrisi yada karışım matrisi
print(confusion_matrix(y_test, y_pred))

print("-------------------------------------------------")

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

print("------------------------------------------------------")
print("Balanced Accuracy:",metrics.balanced_accuracy_score(y_test, y_pred))

print("------------------------------------------------------")

print(classification_report(y_test, y_pred))

