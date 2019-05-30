import numpy as np
from sklearn import preprocessing, neighbors
from sklearn.model_selection import train_test_split
import pandas as pd

accuracies = []
for i in range(25):
    df = pd.read_csv('breast-cancer-wisconsin.txt')
    df.replace('?', -99999, inplace=True) # ?에 -99999로 대체
    df.drop(['id'],1,inplace=True) # 제일 앞부분 id부분은 필요 없으므로 drop시킴

    X = np.array(df.drop(['class'],1))
    y = np.array(df['class'])

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

    clf = neighbors.KNeighborsClassifier(n_jobs=1)
    clf.fit(X_train, y_train)

    accuaracy = clf.score(X_test, y_test)
    # print(accuaracy)
    #
    # example_measures = np.array([[4,2,1,1,1,2,3,2,1],[4,2,1,2,2,2,3,2,1]]) 앞으로 알고 싶은 데이터 입력
    # example_measures = example_measures.reshape(len(example_measures),-1)
    #
    # prediction = clf.predict(example_measures)
    # print(prediction)
    accuracies.append(accuaracy)
print(sum(accuracies)/len(accuracies))