import numpy as np
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score

TRAIN_FILE="data/train.csv"
TEST_FILE="data/test.csv"

def labels():
    """ Get the labels from the training set """
    data = np.genfromtxt('data/train.csv', delimiter=",", dtype=None, names=True)
    return [d["SalePrice"] for d in data]


def get_features(data):
    """ Extract/transform features """
    def filter(v):
        if v == "NA":
            return 0

        return v

    features = [data["GrLivArea"],
                data["TotalBsmtSF"],
                data["LowQualFinSF"],
                data["BsmtUnfSF"],
                float(data["RoofStyle"]=="Gable")]

    return list(map(filter, features))


def get_train_features():
    data = np.genfromtxt(TRAIN_FILE, delimiter=",", dtype=None, names=True)
    labels = []
    features = []
    for d in data:
        labels.append(d["SalePrice"])
        features.append(get_features(d))

    return np.array(features, dtype=np.float), labels

def get_test_features():
    data = np.genfromtxt(TEST_FILE, delimiter=",", dtype=None, names=True)
    ids = []
    features = []
    for d in data:
        ids.append(d["Id"])
        features.append(get_features(d))

    return np.array(features, dtype=np.float), ids


train_features, train_labels = get_train_features()
test_features, test_ids = get_test_features()

reg = Lasso()
reg.fit(train_features, train_labels)
pred = reg.predict(test_features)

print "R2 score on training set", r2_score(train_labels, reg.predict(train_features))

sub = [",".join(["Id", "SalePrice"])]
for i, p in enumerate(pred):
    sub.append(",".join([str(test_ids[i]), str(p)]))

with open("sub.csv", "w") as fh:
    fh.write("\n".join(sub) + "\n")
