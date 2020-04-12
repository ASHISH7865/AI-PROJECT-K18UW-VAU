import pandas as pd
from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import Normalizer

names = ["duration", "protocol_type", "service", "flag", "src_bytes",
         "dst_bytes", "land", "wrong_fragment", "urgent", "hot", "num_failed_logins",
         "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root",
         "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
         "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
         "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
         "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
         "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
         "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
         "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label"]
data = pd.read_csv("kddcup.data_10_percent.csv", names=names)

pd.set_option('display.width', 2000)
pd.set_option('display.max_column', 50)
pd.set_option('precision', 3)
data1 = data
data1 = data1.drop('protocol_type', axis=1)
data1 = data1.drop('service', axis=1)
data1 = data1.drop('flag', axis=1)
data1 = data1.drop('label', axis=1)
import matplotlib.pyplot as plt

cor = data1.corr()
fig = plt.figure()
subFig = fig.add_subplot(111)

cax = subFig.matshow(cor, vmin=-1, vmax=1)
fig.colorbar(cax)
import numpy as np


ticks = np.arange(0, len(data1.columns))
subFig.set_xticks(ticks)
subFig.set_yticks(ticks)
subFig.set_xticklabels(data1.columns, rotation='vertical', size=7)
subFig.set_yticklabels(data1.columns, size=7)

plt.show()


x = list(cor.columns)
g = []

for i in x:
    for j in x[:x.index(i)]:
        if abs(cor[i][j]) >= 0.7:
            g.append((i, j))
o, c = [], 0
for i in g:
    for v in o:
        if i[0] in [x for x in v] or i[1] in [x for x in v]:
            c += 1
    if c == 0:
        s = {i[0], i[1]}
        for j in g[g.index(i) + 1:]:
            if i[0] == j[0] or i[0] == j[1] or i[1] == j[0] or i[1] == j[1]:
                s.add(j[0])
                s.add(j[1])

        o.append(s)
    c = 0
b = []
listd = set(b)
for i in o:
    for j in list(i)[:-1]:
        listd.add(j)

for i in listd:
    data = data.drop(i, axis=1)




dataX = data.values[:, :data.shape[1] - 1]
dataY = data.values[:, data.shape[1] - 1]
dataY1 = dataY
dataX[:, 1] = LabelEncoder().fit_transform(dataX[:, 1])
dataX[:, 2] = LabelEncoder().fit_transform(dataX[:, 2])
dataX[:, 3] = LabelEncoder().fit_transform(dataX[:, 3])
dataY = LabelEncoder().fit_transform(dataY)
label = {}


for i in range(len(dataY1)):
    label[dataY[i]] = dataY1[i]


def acuu(x, y):
    l = []
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.33, random_state=7)
    fit = Normalizer().fit(x_train)
    x_train = fit.fit_transform(x_train)


    from sklearn.neighbors import KNeighborsClassifier

    knn = KNeighborsClassifier()
    knn.fit(x_train, y_train)
    y_pred = knn.predict(x_val)
    acc_knn = round(accuracy_score(y_pred, y_val) * 100, 2)
    print("MODEL-8: Accuracy of k-Nearest Neighbors : ", acc_knn)
    l.append(acc_knn)


from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE

model = LogisticRegression()
rfe = RFE(model, 17)
fit = rfe.fit(dataX, dataY)
train2 = fit.transform(dataX)
print("RFE")
acuu(train2, dataY)

