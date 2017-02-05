import numpy as np
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error

data = np.genfromtxt('data/train.csv', delimiter=",", dtype=None, names=True)

labels = []
inputs = []
for e in data:
    labels.append(e["SalePrice"])
    inputs.append([e["GrLivArea"], e["TotalBsmtSF"], e["LowQualFinSF"], e["BsmtUnfSF"], e["RoofStyle"]=="Gable"])

reg = Lasso()
reg.fit(inputs, labels)

pred = reg.predict(inputs)

print "Score: ", mean_squared_error(labels, pred)
