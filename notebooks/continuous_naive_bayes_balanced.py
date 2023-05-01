import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate, cross_val_predict
from sklearn.pipeline import Pipeline

from notebooks.selection import get_balanced_dataset


xtrain, xtest, ytrain, ytest = get_balanced_dataset('dataset/training.csv')

"""
subset of continuous features gives slightly higher accuracy and AUC
Mean accuracy for 5-fold CV: 0.626 -- much worse than unbalanced which is expected
Mean AUC for 5-fold CV: 0.68 -- slightly better than with unbalanced dataset
"""
continuous_feats = ['VehicleAge', 'VehOdo', 'MMRAcquisitionAuctionAveragePrice']

cont_transformer = Pipeline([
    ("scaler", StandardScaler())
])

preprocessor = ColumnTransformer([
    ("cont", cont_transformer, continuous_feats),
])

clf = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", GaussianNB())
])

results = cross_validate(clf, xtrain[continuous_feats], ytrain['IsBadBuy'], scoring=['accuracy', 'balanced_accuracy', 'roc_auc'])
mean_accuracy = np.array(results['test_accuracy']).mean()
mean_balanced_accuracy = np.array(results['test_balanced_accuracy']).mean()
mean_auc = np.array(results['test_roc_auc']).mean()
print("Mean accuracy ", mean_accuracy)
print("Mean balanced accuracy ", mean_balanced_accuracy)
print("Mean AUC ", mean_auc)
#%%
from sklearn.metrics import RocCurveDisplay
import matplotlib.pyplot as plt

clf.fit(xtrain[continuous_feats], ytrain['IsBadBuy'])
plot = RocCurveDisplay.from_estimator(clf, xtrain[continuous_feats], ytrain['IsBadBuy'])
plt.show()
#%%
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

y_pred = cross_val_predict(clf, X=xtrain, y=ytrain['IsBadBuy'])
mat = confusion_matrix(y_pred, ytrain['IsBadBuy'])
cm_display = ConfusionMatrixDisplay(mat).plot(cmap='binary')


plt.show()