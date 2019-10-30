import pandas as pd
from sklearn import svm
from ml_tps.utils.evaluation import getConfusionMatrix, computeAccuracy


def test_svm_configurations(kernels: list, c_values: list,
                            X_train: pd.DataFrame, y_train: pd.Series,
                            X_cv_set: pd.DataFrame, y_cv_set: pd.Series,
                            printConfusionMatrices: bool = False):
    svm_values = pd.DataFrame(columns=["Kernel", "C value", "Training set accuracy", "CV set accuracy"])

    i = 0
    for kernel in kernels:
        for c_value in c_values:
            clf = svm.SVC(kernel=kernel, C=c_value, gamma="scale", cache_size=1000)
            clf.fit(X_train, y_train)
            predictions_train = pd.Series(clf.predict(X_train))
            predictions_cv = pd.Series(clf.predict(X_cv_set))
            accuracy_train = computeAccuracy(predictions_train, y_train)
            accuracy_cv = computeAccuracy(predictions_cv, y_cv_set)

            configuration_data = [kernel, c_value, accuracy_train, accuracy_cv]
            svm_values.loc[i] = configuration_data
            i += 1

            if printConfusionMatrices:
                print("\n", configuration_data[:2])
                print(getConfusionMatrix(predictions_cv, y_cv_set))

    best_svm_values = svm_values.sort_values(by="CV set accuracy", ascending=False).head(1)
    best_svm = svm.SVC(kernel=best_svm_values.iat[0, 0], C=best_svm_values.iat[0, 1], gamma="scale", cache_size=500)

    return svm_values, best_svm
