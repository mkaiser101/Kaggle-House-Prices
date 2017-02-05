import csv
import numpy as np
from pprint import pprint
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error

#with open('data/train.csv', 'rb') as fh:
#    header = fh.readline().strip().split(",")
#    header = [h for h in header if "SF" in h]

#pprint(header)

data = np.genfromtxt('data/train.csv', delimiter=",", dtype=None, names=True)

labels = []
inputs = []
for e in data:
    labels.append(e["SalePrice"])
    inputs.append([e["GrLivArea"], e["TotalBsmtSF"], e["LowQualFinSF"], e["BsmtUnfSF"]])

labels = np.array(labels, dtype=np.float)
inputs = np.array(inputs, dtype=np.float)

reg = Lasso()
reg.fit(inputs, labels)

pred = reg.predict(inputs)

print "Score: ", mean_squared_error(labels, pred)
