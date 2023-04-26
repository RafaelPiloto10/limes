import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


training = pd.read_csv("../dataset/training.csv")
y_train = training[["IsBadBuy"]]
training.drop(columns="IsBadBuy", inplace=True)

continuous_feats = ['VehicleAge', 'VehOdo', 'MMRAcquisitionAuctionAveragePrice']
# features = ['VehicleAge', 'VehOdo',
#        'MMRAcquisitionAuctionAveragePrice', 'MMRAcquisitionAuctionCleanPrice',
#        'MMRAcquisitionRetailAveragePrice', 'MMRAcquisitonRetailCleanPrice',
#        'MMRCurrentAuctionAveragePrice', 'MMRCurrentAuctionCleanPrice',
#        'MMRCurrentRetailAveragePrice', 'MMRCurrentRetailCleanPrice', 'VehBCost', 'WarrantyCost']

cont_transformer = Pipeline([
    ("imputer", SimpleImputer()),
    ("scaler", StandardScaler())
])

preprocessor = ColumnTransformer([
    ("cont", cont_transformer, continuous_feats),
])

clf = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", GaussianNB())
])


results = cross_validate(clf, training[continuous_feats], y_train['IsBadBuy'], scoring=['accuracy', 'balanced_accuracy', 'roc_auc'])
mean_accuracy = np.array(results['test_accuracy']).mean()
mean_balanced_accuracy = np.array(results['test_balanced_accuracy']).mean()
mean_auc = np.array(results['test_roc_auc']).mean()
print("Mean accuracy ", mean_accuracy)
print("Mean balanced accuracy ", mean_balanced_accuracy)
print("Mean AUC ", mean_auc)
#%%
from sklearn.metrics import RocCurveDisplay
import matplotlib.pyplot as plt

clf.fit(training[continuous_feats], y_train['IsBadBuy'])
plot = RocCurveDisplay.from_estimator(clf, training[continuous_feats], y_train['IsBadBuy'])
plt.show()
#%%
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

y_pred = cross_val_predict(clf, X=training, y=y_train['IsBadBuy'])
mat = confusion_matrix(y_pred, y_train['IsBadBuy'])
cm_display = ConfusionMatrixDisplay(mat).plot(cmap='binary')


plt.show()