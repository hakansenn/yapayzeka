import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import metrics
from sklearn.feature_selection import VarianceThreshold, chi2, SelectKBest #bu varyansı kullanabilmek için indirmemiz gereken bi r kütüphanedir.
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold

def normalize():#değerleri düzenleyerek belli bir varyans değeri altınta kalabilecek olan özniteliklerin atılması engellernir.
    for kolon in df.columns:
        if kolon == "s1":
            continue
        df[kolon] = df[kolon] / df[kolon].abs().max() # en buyuk elemana bolerek normalize ediyoruz
    print(df.head(50))

def varyansaGoreEle():
    selector = VarianceThreshold(threshold=0.1)    #0.0105 normalde tyreshold değer yazmazsan 0 olarak kabul edielir. Ama burada değer verilebilir.
    selector.fit_transform(X) #verileri verdik
    concol = [column for column in X.columns if column not in X.columns[selector.get_support()]]#selkecter.get support aslında hangi nitelik sağlıyor hangisi sağlamıyorsa onu döndürüyor.dkodun amacı ise elenenler bir yerde concolda topluyor
    for features in concol:#istenilen varyansı sağlamayan öznitelikleri ele alır ve bunları yazdırır.
        print(features)
    X.drop(concol,axis=1,inplace=True)#topladığımız elenenleri atııyor. inplace true ise bunu x üzerinde yap demektir.
    print("Varyans elemesi sonrası öznitelik sayısı:",len(X.columns))#eleme sonrası kalan öznitelik sayısını vermekte.

def cross_validation():
    kf = KFold(n_splits=5)
    #kf = StratifiedKFold(n_splits=5)
    clf = tree.DecisionTreeClassifier()
    i=1
    for train_index, test_index in kf.split(X):
    #for train_index, test_index in kf.split(X,y):
        #print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        clf = clf.fit(X_train,y_train)
        
        # Test yapıyoruz. Test verisini ver.
        # Tahmin ettiği sınıfları döndürsün.
        # y_pred = tahmin edilen sınıflar
        y_pred = clf.predict(X_test)

    # Model Accuracy, how often is the classifier correct?
        print(confusion_matrix(y_test, y_pred))

        print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
        print("Balanced Accuracy:",metrics.balanced_accuracy_score(y_test, y_pred))
        print(classification_report(y_test, y_pred))
        #agaciKaydet(clf,i)
        i = i + 1


def agaciKaydet(siniflandirici,agac_numarasi):
    plt.figure(figsize=(12,12))
    tree.plot_tree(siniflandirici, fontsize=6)
    plt.savefig('Agac'+str(agac_numarasi), dpi=600)




df= pd.read_csv("wine.data")
df.columns=["s"+str(i) for i in range(1, 15)]


# Yeni bir oznitelik ekliyoruz.
#df["Column59"] = df["Column55"]/df["Column56"]
#print(df.head(50))

# Bos olan degere sahip olan satirlari at.
print(len(df.index))
df.dropna(axis=0,inplace=True)#inplace true o satırı atar. inplace false ise atılan tablonun atılmış halinin kopyasını oluturu bu bizi uğraştırır. esas 0 değerlerini atan isen dropna fonksiyonudur.
print(len(df.index))

#Normalize et
# normalize()

X = df.drop("s1", axis=1)
y = df["s1"]

# Varyansa göre bilgi vermeyen öznitelikleri ele.
varyansaGoreEle()


#cross_validation()


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.40, random_state=1)
# Create Decision Tree classifer object
clf = tree.DecisionTreeClassifier()
#clf = DecisionTreeClassifier(class_weight={'g': 0.6, 'b': 0.4})

# Eğitim setini ver. Fonksiyonu uydursun.
# Fonksiyon bir ağaç
clf = clf.fit(X_train,y_train)

# Test yapıyoruz. Test verisini ver.
# Tahmin ettiği sınıfları döndürsün.
# y_pred = tahmin edilen sınıflar
y_pred = clf.predict(X_test)


# Model Accuracy, how often is the classifier correct?
print(confusion_matrix(y_test, y_pred))

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Balanced Accuracy:",metrics.balanced_accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))



test = SelectKBest(score_func=chi2, k=4)
fit = test.fit(X, y)

X_best= test.fit_transform(X,y)

scores=fit.scores_
print(scores)


print('Original number of features:', X.shape)
print('Reduced number of features:', X_best.shape)