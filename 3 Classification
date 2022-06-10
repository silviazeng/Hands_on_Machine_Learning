import os
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "classification"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

def plot_digit(data):
    image = data.reshape(28, 28)
    plt.imshow(image, cmap = mpl.cm.binary,
               interpolation="nearest")
    plt.axis("off")

#-----Download Data-----
from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784', version=1)
type(mnist)
mnist.keys()
    #Datasets loaded by Scikit-Learn generally have a similar dictionary structure including:
    # • A DESCR key describing the dataset
    # • A data key containing an array with one row per instance and one column per feature
    # • A target key containing an array with the labels


#-----Preview Data-----
X, y = mnist["data"], mnist["target"]
X.shape #70,000 images, each has 784 features.(28×28 pixels)
y.shape
type(X)

import matplotlib as mpl
import matplotlib.pyplot as plt
some_digit = X.iloc[0]
type(some_digit.values)
some_digit_image = some_digit.values.reshape(28, 28)
plt.imshow(some_digit_image, cmap = mpl.cm.binary, interpolation="nearest") #image show
plt.axis("off")
plt.show()

import numpy as np
print(y[0])
y = y.astype(np.uint8)

#-----Create a Test Set-----
X_train, X_test, y_train, y_test=X[:60000],X[60000:],y[:60000],y[60000:]
type(X_train)
type(y_train)


#-----Train a Binary Classifier to identify "5"-----
y_train_5 = (y_train==5) #True for all 5s, False for all other digits.
y_test_5 = (y_test==5)

 #Model 1: SGDClassifier
from sklearn.linear_model import SGDClassifier
sgd_clf = SGDClassifier(random_state=42)
    #SGDClassifier relies on randomness during training (hence “stochastic”). If you want reproducible results, you should set random_state parameter.
sgd_clf.fit(X_train, y_train_5)
sgd_clf.predict([some_digit])

  #cross validation score
  #Note: The cross_validate() function differs from cross_val_score() in two ways:
    # -It allows specifying multiple metrics for evaluation.
    # -It returns a dict containing fit-times, score-times (and optionally training scores as well as fitted estimators) in addition to the test score.
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)
    #T he function cross_val_predict has a similar interface to cross_val_score, but returns, for each element in the input, the prediction that was obtained for that element when it was in the test set. Only cross-validation strategies that assign all elements to a test set exactly once can be used (otherwise, an exception is raised).
    # The function cross_val_score takes an average over cross-validation folds, whereas cross_val_predict simply returns the labels (or probabilities) from several distinct models undistinguished. Thus, cross_val_predict is not an appropriate measure of generalisation error.
    # cross_val_predict() is appropriate for: Visualization of predictions obtained from different models;Model blending: When predictions of one supervised estimator are used to train another estimator in ensemble methods.

confusion_matrix(y_train_5, y_train_pred) #attention to the sequence!
    #true neg, false positive
    #false neg, true positive
precision_score(y_train_5, y_train_pred)
recall_score(y_train_5, y_train_pred)
f1_score(y_train_5, y_train_pred)
cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring='f1')

  # The following code does roughly the same thing as Scikit-Learn’s cross_val_score() function, and prints the same result
  # When the cv argument is an integer, cross_val_score() uses the KFold or StratifiedKFold strategies by default, the latter being used if the estimator derives from ClassifierMixin
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
skfolds = StratifiedKFold(n_splits=3, random_state=42, shuffle=True)
for train_index, test_index in skfolds.split(X_train, y_train_5):
    # split generate indices to split data into training and test set (for 3 times)
    clone_clf = clone(sgd_clf)# At each iteration,creates a clone of the classifier, trains that clone on the training folds, and makes predictions on the test fold.
    X_train_folds = X_train.loc[train_index]
    y_train_folds = y_train_5[train_index]
    X_test_fold = X_train.loc[test_index]
    y_test_fold = y_train_5[test_index]
    clone_clf.fit(X_train_folds, y_train_folds)
    y_pred = clone_clf.predict(X_test_fold)
    n_correct = sum(y_pred == y_test_fold)
    print(n_correct / len(y_pred))

  #Adjust the threshold (default 0) for Prediction using Decision functions
y_scores =cross_val_predict(sgd_clf, X_train, y_train_5, cv=3, method="decision_function")
y_scores.shape
print(y_scores)
threshold = 0
y_pred = (y_scores > threshold)

from sklearn.metrics import precision_recall_curve
precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)
def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1],'b--',label='Precision')
    plt.plot(thresholds, recalls[:-1],'g-',label='Recall')
    plt.legend(loc="center right", fontsize=16)
    plt.xlabel("Threshold", fontsize=16)
    plt.grid(True)
    plt.axis([-50000, 50000, 0, 1])

recall_90_precision = recalls[np.argmax(precisions >= 0.90)]
threshold_90_precision = thresholds[np.argmax(precisions >= 0.90)]


plt.figure(figsize=(8,4))
plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
plt.plot([threshold_90_precision, threshold_90_precision], [0., 0.9], "r:")
plt.plot([-50000, threshold_90_precision], [0.9, 0.9], "r:")
plt.plot([-50000, threshold_90_precision], [recall_90_precision, recall_90_precision], "r:")
plt.plot([threshold_90_precision], [0.9], "ro")
plt.plot([threshold_90_precision], [recall_90_precision], "ro")
save_fig("precision_recall_vs_threshold_plot")
plt.show()

    # suppose you decide to aim for 90% precision, you can search for the lowest threshold that gives you at least 90% precision (np.argmax() will give us the first index of the maximum value, which in this case means the first True value).
    # To make predictions (on the training set for now), instead of calling the classifier’s predict() method, you can just run this code:
y_train_pred_90 = (y_scores >= threshold_90_precision)

#-----The ROC Curve (sensitivity/recall v.s. (1-specificity) )-----
 #very similar to the precision/recall curve, but instead of plotting precision versus recall, the ROC curve plots the true positive rate (aka recall) against the false positive rate (ratio of neg instances that are classified as pos)
from sklearn.metrics import roc_curve

fpr, tpr, threshold = roc_curve(y_train_5, y_scores)

def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')
      # dashed diagonal, represents the ROC curve of a purely random classifier; a good classifier stays as far away from that line as possible (toward the top-left corner)
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate (Fall-Out)', fontsize=16)
    plt.ylabel('True Positive Rate (Recall)', fontsize=16)
    plt.grid(True)

plt.figure(figsize=(8, 6))
plot_roc_curve(fpr, tpr)
fpr_90 = fpr[np.argmax(tpr >= recall_90_precision)]
plt.plot([fpr_90, fpr_90], [0., recall_90_precision], "r:")
plt.plot([0.0, fpr_90], [recall_90_precision, recall_90_precision], "r:")
plt.plot([fpr_90], [recall_90_precision], "ro")
save_fig("roc_curve_plot")
plt.show()

# One way to compare classifiers is to measure the area under the curve (AUC)
from sklearn.metrics import roc_auc_score
roc_auc_score(y_train_5, y_scores)

#Since the ROC curve is so similar to the precision/recall (or PR) curve, you may wonder how to decide which one to use. As a rule of thumb, you should prefer the PR curve whenever the positive class is rare or when you care more about the false positives than the false negatives, and the ROC curve otherwise.

 #Model 2: RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
forest_clf = RandomForestClassifier(random_state=42)
y_probas_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3, method="predict_proba")

y_scores_forest = y_probas_forest[:, 1] # score = proba of positive class
fpr_forest, tpr_forest, thresholds_forest = roc_curve(y_train_5,y_scores_forest)

plt.plot(fpr, tpr, "b:", label="SGD")
plot_roc_curve(fpr_forest, tpr_forest, "Random Forest")
plt.legend(loc="lower right")
plt.show()

roc_auc_score(y_train_5, y_scores_forest)


#-----Multiclass Classification-----
sgd_clf.fit(X_train, y_train) # y_train, not y_train_5
    #Under the hood,ittrained 10 binary classifiers, got their decision scores for the image, and selected the class with the highest score.
sgd_clf.predict([some_digit])

  #If you want to force ScikitLearn to use one-versus-one or one-versus-all, you can use the OneVsOneClassifier or OneVsRestClassifier classes.:
from sklearn.multiclass import OneVsOneClassifier
ovo_clf = OneVsOneClassifier(SGDClassifier(random_state=42))
ovo_clf.fit(X_train, y_train)
ovo_clf.predict([some_digit])
len(ovo_clf.estimators_)

forest_clf.fit(X_train, y_train)
forest_clf.predict([some_digit])
forest_clf.predict_proba([some_digit])

cross_val_score(sgd_clf, X_train, y_train, cv=3, scoring="accuracy")

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))
cross_val_score(sgd_clf, X_train_scaled, y_train, cv=3, scoring="accuracy")


#------------------------------------------------------------
#-----Error Analysis-----
y_train_pred = cross_val_predict(sgd_clf, X_train_scaled, y_train, cv=3)
conf_mx = confusion_matrix(y_train, y_train_pred)
plt.matshow(conf_mx, cmap=plt.cm.gray)
plt.show()

row_sums = conf_mx.sum(axis=1, keepdims=True)
norm_conf_mx = conf_mx / row_sums
# Now let’s fill the diagonal with zeros to keep only the errors, and let’s plot the result:
np.fill_diagonal(norm_conf_mx, 0)
plt.matshow(norm_conf_mx, cmap=plt.cm.gray)
plt.show()


#-----------------------------------------------------------------
#------Multilabel Classification-----
from sklearn.neighbors import KNeighborsClassifier
y_train_large = (y_train >= 7)
y_train_odd = (y_train % 2 == 1)
y_multilabel = np.c_[y_train_large, y_train_odd]
knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train, y_multilabel)

knn_clf.predict([some_digit])

#-----Multioutput Classification-----
noise = np.random.randint(0, 100, (len(X_train), 784))
X_train_mod = X_train + noise
noise = np.random.randint(0, 100, (len(X_test), 784))
X_test_mod = X_test + noise
y_train_mod = X_train
y_test_mod = X_test

some_index=0
knn_clf.fit(X_train_mod, y_train_mod)
clean_digit = knn_clf.predict([X_test_mod.iloc[some_index]])
plot_digit(clean_digit)
