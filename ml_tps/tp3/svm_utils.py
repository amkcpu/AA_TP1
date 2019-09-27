import pandas as pd
from sklearn import svm
from ml_tps.utils.evaluation_utils import getConfusionMatrix, computeAccuracy


def test_svm_configurations(kernels: list, c_values: list,
                            X_train: pd.DataFrame, y_train: pd.Series,
                            X_cv_set: pd.DataFrame, y_cv_set: pd.Series):
    svm_values = pd.DataFrame(columns=["Kernel", "C value", "Training set accuracy", "CV set accuracy"])

    i = 0
    for kernel in kernels:
        for c_value in c_values:
            clf = svm.SVC(kernel=kernel, C=c_value, gamma="scale", cache_size=500)
            clf.fit(X_train, y_train)
            accuracy_train = get_svm_accuracy(fitted_classifier=clf, X=X_train, y=y_train)
            accuracy_cv = get_svm_accuracy(fitted_classifier=clf, X=X_cv_set, y=y_cv_set)

            svm_values.loc[i] = [kernel, c_value, accuracy_train, accuracy_cv]
            i += 1

    best_svm_values = svm_values.sort_values(by="CV set accuracy", ascending=False).head(1)
    best_svm = svm.SVC(kernel=best_svm_values.iat[0, 0], C=best_svm_values.iat[0, 1], gamma="scale", cache_size=500)

    return svm_values, best_svm


def get_svm_accuracy(fitted_classifier, X: pd.DataFrame, y: pd.Series):
    predictions = pd.Series(fitted_classifier.predict(X).T)
    return computeAccuracy(predictions, y)