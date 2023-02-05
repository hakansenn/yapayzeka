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


dataset = "adult2.csv"
sinif = "class"
surekli = ["age","fnlwgt","educationNum","capital-gain","caiptal-lost","hoursperwee"]
split_value = 5


def normalize():
    for kolon in df_surekli.columns:
        df_surekli[kolon] = df_surekli[kolon] / df_surekli[kolon].abs().max() # en buyuk elemana bolerek normalize ediyoruz


def varyansaGoreEle():
    selector = VarianceThreshold(threshold=0.01)    #0.0105
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
    total_accuracy = total_balanced_accuracy = average_accuracy = average_balanced_accuracy = 0
    #kf = KFold(n_splits=5)
    kf = StratifiedKFold(n_splits=split_value)
    #Support vektor icin yuzeyi nasil bolecegin 
    #clf = SVC(kernel='linear')
    #Karar Ağacına göre işlem yapacağız
    clf = tree.DecisionTreeClassifier()

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

        total_accuracy += metrics.accuracy_score(y_test, y_pred)
        total_balanced_accuracy += metrics.balanced_accuracy_score(y_test, y_pred)
        i = i + 1
    
    average_accuracy = total_accuracy/split_value
    average_balanced_accuracy = total_balanced_accuracy/split_value
    print()
    print("Average Accuracy: ",average_accuracy)
    print("Average Balanced Accuracy: ",average_balanced_accuracy)
    

df = pd.read_csv(dataset)


# Bos olan degere sahip olan satirlari at.
print(len(df.index))
df.dropna(axis=0,inplace=True) 
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
