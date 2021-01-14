# -*-coding:utf-8-*-

import pandas as pd
import numpy as np
import logging
# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import ExtraTreesClassifier
from mlxtend.classifier import StackingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score
from sklearn.metrics import mean_absolute_error, accuracy_score

logging.basicConfig(level=logging.DEBUG,
                    filename='logging4.log',
                    filemode='a',
                    format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s')


def logistic_regression(X, y):
    print("\n------------logistic_regression------------")
    logging.info("\n------------logistic_regression------------")
    # remember effective alpha scores are 0<alpha<infinity
    C_vals = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    # Choose a cross validation strategy.
    cv = StratifiedShuffleSplit(n_splits=10, test_size=.25)
    # setting param for param_grid in GridSearchCV.
    param = {'C': C_vals}
    LogisticRegression()
    # Calling on GridSearchCV object.
    grid = GridSearchCV(
        estimator=LogisticRegression(penalty = 'l2',
                      solver = 'liblinear',
                      class_weight = None,
                      C = 1,
                      random_state = None,
                      n_jobs = 1),
        param_grid=param,
        scoring='accuracy',
        n_jobs=-1,
        cv=cv
    )
    # Fitting the model
    grid.fit(X, y)
    # Getting the best of everything.
    print(grid.best_score_)
    print(grid.best_params_)
    print(grid.best_estimator_)
    # Using the best parameters from the grid-search.
    logreg_grid = grid.best_estimator_
    print("logreg_grid.score: {}".format(logreg_grid.score(X, y)))
    logging.info("logreg_grid.score: {}".format(logreg_grid.score(X, y)))
    return logreg_grid


# Random forest
def random_forest(X, y):
    print("\n------------random_forest------------")
    logging.info("\n------------random_forest------------")
    n_estimators = [140, 145, 150, 155, 160]
    max_depth = range(2, 16)
    criterions = ['gini', 'entropy']
    cv = StratifiedShuffleSplit(n_splits=10, test_size=.30, random_state=15)
    parameters = {'n_estimators': n_estimators,
                  'max_depth': max_depth,
                  'criterion': criterions}
    grid = GridSearchCV(estimator=RandomForestClassifier(max_features='auto'),
                        param_grid=parameters,
                        cv=cv,
                        n_jobs=-1)
    grid.fit(X, y)
    print(grid.best_score_)
    print(grid.best_params_)
    print(grid.best_estimator_)
    logging.info(grid.best_score_)
    logging.info(grid.best_params_)
    logging.info(grid.best_estimator_)
    rf_grid = grid.best_estimator_
    rf_grid.score(X, y)
    print("rf_grid.score: {}".format(rf_grid.score(X, y)))
    logging.info("rf_grid.score: {}".format(rf_grid.score(X, y)))
    return rf_grid


# Decision Tree
def decision_tree(X, y):
    print("\n------------decision_tree------------")
    logging.info("\n------------decision_tree------------")
    max_depth = range(1, 31)
    max_feature = [2, 3, 4, 5, 6, 7, 8, 9, 10, 'auto']
    criterion = ["entropy", "gini"]
    param = {'max_depth': max_depth,
             'max_features': max_feature,
             'criterion': criterion}
    grid = GridSearchCV(DecisionTreeClassifier(),
                        param_grid=param,
                        verbose=False,
                        cv=StratifiedShuffleSplit(n_splits=20, random_state=15),
                        n_jobs=-1)
    grid.fit(X, y)
    print(grid.best_params_)
    print(grid.best_score_)
    print(grid.best_estimator_)
    logging.info(grid.best_score_)
    logging.info(grid.best_params_)
    logging.info(grid.best_estimator_)
    dectree_grid = grid.best_estimator_
    # using the best found hyper paremeters to get the score.
    print("dectree_grid.score: {}".format(dectree_grid.score(X, y)))
    logging.info("dectree_grid.score: {}".format(dectree_grid.score(X, y)))
    return dectree_grid


# Bagging Classifier
def bagging(X, y):
    print("\n------------bagging------------")
    logging.info("\n------------bagging------------")
    n_estimators = [50, 70, 80, 150, 160, 170, 175, 180, 185];
    cv = StratifiedShuffleSplit(n_splits=10, test_size=.30, random_state=15)
    parameters = {'n_estimators': n_estimators}
    grid = GridSearchCV(BaggingClassifier(base_estimator=None,  # If None, then the base estimator is a decision tree.
                                          bootstrap_features=False),
                        param_grid=parameters,
                        cv=cv,
                        n_jobs=-1)
    grid.fit(X, y)
    print(grid.best_score_)
    print(grid.best_params_)
    print(grid.best_estimator_)
    logging.info(grid.best_score_)
    logging.info(grid.best_params_)
    logging.info(grid.best_estimator_)
    bagging_grid = grid.best_estimator_
    # using the best found hyper paremeters to get the score.
    print("bagging_grid.score: {}".format(bagging_grid.score(X, y)))
    logging.info("bagging_grid.score: {}".format(bagging_grid.score(X, y)))
    return bagging_grid


# AdaBoost
def AdaBoost(X, y):
    print("\n------------AdaBoost------------")
    logging.info("\n------------AdaBoost------------")
    n_estimators = [100, 140, 145, 150, 160, 170, 175, 180, 185];
    cv = StratifiedShuffleSplit(n_splits=10, test_size=.30, random_state=15)
    learning_r = [0.1, 1, 0.01, 0.5]
    parameters = {'n_estimators': n_estimators,
                  'learning_rate': learning_r}
    grid = GridSearchCV(AdaBoostClassifier(base_estimator=None),  # If None, then the base estimator is a decision tree.
                        param_grid=parameters,
                        cv=cv,
                        n_jobs=-1)
    grid.fit(X, y)
    print(grid.best_score_)
    print(grid.best_params_)
    print(grid.best_estimator_)
    logging.info(grid.best_score_)
    logging.info(grid.best_params_)
    logging.info(grid.best_estimator_)
    adaBoost_grid = grid.best_estimator_
    # using the best found hyper paremeters to get the score.
    print("adaBoost_grid.score: {}".format(adaBoost_grid.score(X, y)))
    logging.info("adaBoost_grid.score: {}".format(adaBoost_grid.score(X, y)))
    return adaBoost_grid


# Gradient Boosting
def gradient_boosting(X, y, X_val, y_val):
    print("\n------------gradient_boosting------------")
    logging.info("\n------------gradient_boosting------------")
    gradient_boost = GradientBoostingClassifier()
    gradient_boost.fit(X, y)
    y_pred = gradient_boost.predict(X_val)
    gradient_accy = round(accuracy_score(y_pred, y_val), 3)
    print("gradient_accy: {}".format(gradient_accy))
    logging.info("gradient_accy: {}".format(gradient_accy))
    return gradient_boost


# XGB
def XGB(X_train, y_train, X_val, y_val):
    print("\n------------XGB------------")
    logging.info("\n------------XGB------------")
    from xgboost import XGBClassifier
    XGBClassifier = XGBClassifier(n_estimators=2000,
                                  max_depth=4,
                                  min_child_weight=2,
                                  # gamma=1,
                                  gamma=0.9,
                                  subsample=0.8,
                                  colsample_bytree=0.8,
                                  objective='binary:logistic',
                                  nthread=-1,
                                  scale_pos_weight=1).fit(X_train, y_train)
    y_pred = XGBClassifier.predict(X_val)
    XGBClassifier_accy = round(accuracy_score(y_pred, y_val), 3)
    print("XGBClassifier_accy: {}".format(XGBClassifier_accy))
    logging.info("XGBClassifier_accy: {}".format(XGBClassifier_accy))
    return XGBClassifier


# Extra Trees
def extra_trees(X, y, X_val, y_val):
    from sklearn.ensemble import ExtraTreesClassifier
    print("\n------------extra_trees------------")
    logging.info("\n------------extra_trees------------")
    ExtraTreesClassifier = ExtraTreesClassifier()
    ExtraTreesClassifier.fit(X, y)
    y_pred = ExtraTreesClassifier.predict(X_val)
    extraTree_accy = round(accuracy_score(y_pred, y_val), 3)
    print("extraTree_accy: {}".format(extraTree_accy))
    logging.info("extraTree_accy: {}".format(extraTree_accy))
    return ExtraTreesClassifier


# Voting Classifier
def voting(X, y, X_val, y_val):
    print("\n------------voting------------")
    logging.info("\n------------voting------------")
    voting_classifier = VotingClassifier(estimators=[
        # ('lr_grid', logreg_grid),
        ('random_forest', rf_grid),
        ('decision_tree_grid', dectree_grid),
        ('XGB_Classifier', XGBClassifier),
        ('bagging_classifier', bagging_grid),
        ('adaBoost_classifier', adaBoost_grid),
        ('ExtraTrees_Classifier', ExtraTreesClassifier)], voting='soft')

    # voting_classifier = voting_classifier.fit(train_x,train_y)
    voting_classifier = voting_classifier.fit(X, y)
    y_pred = voting_classifier.predict(X_val)
    voting_accy = round(accuracy_score(y_pred, y_val), 3)
    print("voting_accy: {}".format(voting_accy))
    logging.info("voting_accy: {}".format(voting_accy))
    return voting_classifier


if __name__ == '__main__':
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")
    y = train['label']
    X = train.drop(['label'], axis=1)

    X_d1, X_d2, y_d1, y_d2 = train_test_split(X, y, test_size=0.5, random_state=2020)
    dataset_d1 = np.zeros((X_d2.shape[0], 7))
    dataset_d2 = np.zeros((test.shape[0], 7))

    np.set_printoptions(suppress=True, precision=8)

    # logreg_grid = logistic_regression(X, y)
    # logreg_predict = logreg_grid.predict_proba(test)
    # np.savetxt("logreg_submission.txt", logreg_predict[:, [1]], fmt='%.06f')
    k = 0
    rf_grid = random_forest(X_d1, y_d1)
    dataset_d1[:, k] = rf_grid.predict_proba(X_d2)[:, -1]
    dataset_d2[:, k] = rf_grid.predict_proba(test)[:, -1]
    k += 1

    dectree_grid = decision_tree(X_d1, y_d1)
    dataset_d1[:, k] = dectree_grid.predict_proba(X_d2)[:, -1]
    dataset_d2[:, k] = dectree_grid.predict_proba(test)[:, -1]
    k+=1

    XGBClassifier = XGB(X_d1, y_d1, X_d2, y_d2)
    dataset_d1[:, k] = XGBClassifier.predict_proba(X_d2)[:, -1]
    dataset_d2[:, k] = XGBClassifier.predict_proba(test)[:, -1]
    k += 1

    bagging_grid = bagging(X_d1, y_d1)
    dataset_d1[:, k] = bagging_grid.predict_proba(X_d2)[:, -1]
    dataset_d2[:, k] = bagging_grid.predict_proba(test)[:, -1]
    k += 1

    adaBoost_grid = AdaBoost(X_d1, y_d1)
    dataset_d1[:, k] = adaBoost_grid.predict_proba(X_d2)[:, -1]
    dataset_d2[:, k] = adaBoost_grid.predict_proba(test)[:, -1]
    k += 1

    ExtraTreesClassifier = extra_trees(X_d1, y_d1, X_d2, y_d2)
    dataset_d1[:, k] = ExtraTreesClassifier.predict_proba(X_d2)[:, -1]
    dataset_d2[:, k] = ExtraTreesClassifier.predict_proba(test)[:, -1]
    k += 1

    voting_classifier = voting(X_d1, y_d1, X_d2, y_d2)
    dataset_d1[:, k] = voting_classifier.predict_proba(X_d2)[:, -1]
    dataset_d2[:, k] = voting_classifier.predict_proba(test)[:, -1]
    k += 1

    clf = GradientBoostingClassifier()
    clf.fit(dataset_d1, y_d2)
    y_submission = clf.predict_proba(dataset_d2)[:, -1]
    np.savetxt("y_submission.txt", y_submission, fmt='%.06f')

