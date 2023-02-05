import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import metrics
from sklearn.feature_selection import VarianceThreshold
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold

dataset = "spambase.data"
sinif = "Column58"


def normalize():
    for kolon in df.columns:
        if kolon == sinif: # sınıfı atla
            continue
        df[kolon] = df[kolon] / df[kolon].abs().max() # en buyuk elemana bolerek normalize ediyoruz
    print(df.head(50))

def varyansaGoreEle():
    selector = VarianceThreshold(threshold=0.30)    #0.0105
    selector.fit_transform(X)
    #concol icerisinde eşik şartını sağlamayan öznitelikler var
    concol = [column for column in X.columns if column not in X.columns[selector.get_support()]]
    for features in concol:
        print(features)
    X.drop(concol,axis=1,inplace=True)
    print("Varyans elemesi sonrası öznitelik sayısı:",len(X.columns))

def cross_validation():
    #kf = KFold(n_splits=5)
    kf = StratifiedKFold(n_splits=5)
    #Support vektor icin yuzeyi nasil bolecegin 
    clf = SVC(kernel='linear',)
    i=1
    #for train_index, test_index in kf.split(X):
    for train_index, test_index in kf.split(X,y):
        #print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        clf = clf.fit(X_train,y_train)
        
        # Test yapıyoruz. Test verisini ver.
        # Tahmin ettiği sınıfları döndürsün.
        # y_pred = tahmin edilen sınıflar
        y_pred = clf.predict(X_test)

        # Hata matrisini oluştur. 
        print(confusion_matrix(y_test, y_pred))

        print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
        print("Balanced Accuracy:",metrics.balanced_accuracy_score(y_test, y_pred))
        print(classification_report(y_test, y_pred))
        i = i + 1



df= pd.read_csv(dataset)
df.columns=["Column"+str(i) for i in range(1, 59)]


# Bos olan degere sahip olan satirlari at.
print("Satir sayisi: ", len(df.index))
#df.dropna(axis=0,inplace=True)
print("Satir sayisi: ", len(df.index))

#Normalize et
#normalize()

X = df.drop(sinif, axis=1)
y = df[sinif]

# Varyansa göre bilgi vermeyen öznitelikleri ele.
#varyansaGoreEle()


cross_validation()


#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)
# Create Decision Tree classifer object
#clf = svclassifier = SVC(kernel='rbf',gamma=10, C = 100)

# Eğitim setini ver. Fonksiyonu uydursun.
# Fonksiyon bir ağaç
#clf = clf.fit(X_train,y_train)

# Test yapıyoruz. Test verisini ver.
# Tahmin ettiği sınıfları döndürsün.
# y_pred = tahmin edilen sınıflar
#y_pred = clf.predict(X_test)


# Model Accuracy, how often is the classifier correct?
# print(confusion_matrix(y_test, y_pred))
# print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
# print("Balanced Accuracy:",metrics.balanced_accuracy_score(y_test, y_pred))
# print(classification_report(y_test, y_pred))

