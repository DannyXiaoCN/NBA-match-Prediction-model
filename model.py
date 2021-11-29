import re
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def data_preprocess(path):
    train_data = pd.read_csv(path)
    x_columns = ['Attractive','HomeGrind','VisitorGrind','RecentGame','BBBGame','MultiVisitorGame']
    X = train_data[x_columns]
    y = train_data['y_true']
    return X, y

def train(X, y):
    model = RandomForestClassifier(random_state=725)
    model.fit(X,y)
    score_pre = cross_val_score(model, X, y, cv=10).mean()
    print(score_pre)

def optimize_n_estimators_step1(X, y):
    score_lt = []
    for i in range(0, 200, 10):
        model_x = RandomForestClassifier(n_estimators=i+1, random_state=725)
        score = cross_val_score(model_x, X, y, cv=10).mean()
        score_lt.append(score)
    score_max = max(score_lt)
    print("Best AUC score: {}".format(score_max),
          "Number of child tree: {}".format(score_lt.index(score_max)*10+1))
    
    x_axis = np.arange(1, 201, 10)
    plt.subplot(111)
    plt.plot(x_axis, score_lt, 'r-')
    plt.show()
    
    
def optimize_n_estimators_step2(X, y):
    score_lt = []
    for i in range(30, 50):
        model_x = RandomForestClassifier(n_estimators=i, random_state=725)
        score = cross_val_score(model_x, X, y, cv=10).mean()
        score_lt.append(score)
    score_max = max(score_lt)
    print("Best AUC score: {}".format(score_max),
          "Number of child tree: {}".format(score_lt.index(score_max)+30))
    
    x_axis = np.arange(30, 50)
    plt.subplot(111)
    plt.plot(x_axis, score_lt, 'o-')
    plt.show()
    
def optimize_max_depth(X, y):
    
    model_g = RandomForestClassifier(n_estimators=40, random_state=725)
    
    param_grid = {'max_depth':np.arange(1,20)}
    GS = GridSearchCV(model_g, param_grid, cv=10)
    GS.fit(X, y)
    print(GS.best_params_, GS.best_score_)
    
def optimize_max_feature(X, y):
    
    model_g = RandomForestClassifier(n_estimators=40, random_state=725, max_depth=4)
    
    param_grid = {'max_features':np.arange(2, 7)}
    GS = GridSearchCV(model_g, param_grid, cv=10)
    GS.fit(X, y)
    print(GS.best_params_, GS.best_score_)
    
def accuracy_test(X, y, X_test, y_test):
    model = RandomForestClassifier(random_state=725, n_estimators=55,
                                   max_depth=4, max_features=2)
    model.fit(X,y)
    predict = model.predict(X_test)
    result = pd.DataFrame(predict, columns=["Predict"])
    result['Real World'] = 0
    i = 0
    for index, row in result.iterrows():
        result.loc[index, 'Real World'] = y_test[i]
        i += 1
    print(result)
    score = model.score(X_test, y_test)
    print(score)
    
def feature_estimate(X, y):
    model = RandomForestClassifier(random_state=725)
    model.fit(X,y)
    fpr, tpr, thresholds = roc_curve(y, model.predict_proba(X)[:,1])
    plt.plot(fpr, tpr, linewidth=2, label='ROC')
    plt.xlabel('false positive rate')    
    plt.ylabel('true positive rate')
    plt.show()

if __name__ == "__main__":
    
    X, y = data_preprocess("training_data.csv")
    X_test, y_test = data_preprocess("test_data.csv")
    # train(X, y)
    # optimize_n_estimators_step1(X, y)
    # optimize_n_estimators_step2(X, y)
    # optimize_max_depth(X, y)
    # optimize_max_feature(X, y)
    accuracy_test(X, y, X_test, y_test)
    # feature_estimate(X, y)
    