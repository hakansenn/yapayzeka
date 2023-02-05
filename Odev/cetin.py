import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics 
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import StratifiedKFold


dataset = "australian.dat"
sinif = "A15"
surekli = ["A2","A3","A7","A10","A13","A14"]


def normalize():
    for kolon in df_surekli.columns:
        if kolon == "A15":
            continue
        df_surekli[kolon] = df_surekli[kolon] / df_surekli[kolon].abs().max() # en buyuk elemana bolerek normalize ediyoruz
    print(df_surekli.head())

def varyansaGoreEle():
    selector = VarianceThreshold(threshold=0.0105)    #0.0105
    selector.fit_transform(df_surekli)
    #concol icerisinde eşik şartını sağlamayan öznitelikler var
    concol = [column for column in df_surekli.columns if column not in df_surekli.columns[selector.get_support()]]
    for features in concol:
        print(features)
    df_surekli.drop(concol,axis=1,inplace=True)
    print("Varyans elemesi sonrası öznitelik sayısı:",len(df_surekli.columns))


def kikareyeGoreEle():
    selector = SelectKBest(chi2,k=4)
    selector.fit_transform(df_kategorik,y)
    concol = [column for column in df_kategorik.columns if column not in df_kategorik.columns[selector.get_support()]]
    for features in concol:
        print(features)
    df_kategorik.drop(concol,axis=1,inplace=True)
    print("Kikare elemesi sonrası öznitelik sayısı:",len(df_kategorik.columns))

def cross_validation():
    #kf = KFold(n_splits=5)
    kf = StratifiedKFold(n_splits=5)
    #Support vektor icin yuzeyi nasil bolecegin 
    clf = tree.DecisionTreeClassifier()
    i=1
    toplam_accuracy=0
    toplam_balanced_accuracy=0
    ort_accuracy=0
    ort_balanced_accuracy=0
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
        
        toplam_accuracy+=metrics.accuracy_score(y_test, y_pred)
        toplam_balanced_accuracy+=metrics.balanced_accuracy_score(y_test, y_pred)
        i = i + 1
    ort_accuracy=toplam_accuracy/5
    ort_balanced_accuracy=toplam_balanced_accuracy/5
    print("Ortalama Accuracy:",ort_accuracy)
    print("Ortalama Balanced Accuracy:",ort_balanced_accuracy)

df = pd.read_csv(dataset, header=None)
df.columns=["A"+str(i) for i in range(1, 16)]

# Bos olan degere sahip olan satirlari at.
print(len(df.index))
df.dropna(axis=0,inplace=True) #inplace true deyince yaptığını üzerinde değiştir, false yaparsak yeni bir değere eşitlemek gerekirdi.
print(len(df.index))


y= df[sinif]
df= df.drop(sinif,axis=1)

df_kategorik = df.drop(surekli,axis=1)
df_surekli = df.drop(df_kategorik,axis=1)


normalize()
varyansaGoreEle()
kikareyeGoreEle()


X = pd.concat([df_surekli,df_kategorik],axis=1)

cross_validation()
