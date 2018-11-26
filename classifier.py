import librosa
import sys
import pandas as pd
from scipy import stats
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score, accuracy_score
from sklearn.neighbors import KNeighborsClassifier

# read in training data
data_headers = ["tempo", "beat_frames_std", "duration", "stftD_std", "stftD_mean", "freq_std", "freq_mean", "cqt_std", "cqt_mean", "iirt_std", "iirt_mean", "H_std", "H_mean", "P_mean", "P_std", "rms_std", "rms_mean", "onset_env_std", "onset_env_mean", "zcr_std", "zcr_mean", "cent_std", "cent_mean", "contrast_std", "contrast_mean", "spec_bw_std", "spec_bw_mean", "rolloff_std", "rolloff_mean", "mfcc_std", "mfcc_mean", "genre"]

csvIn = pd.read_csv("allTrainCSV.csv", header=None, names=data_headers)
train_data = pd.DataFrame(csvIn, columns = data_headers)
train_X = train_data.loc[:, "tempo":"mfcc_std"]
X_headers = list(train_X)
train_X = train_X.as_matrix()
train_y = train_data["genre"].as_matrix()

# read in test data
testIn = pd.read_csv("testCSV.csv", header=None, names=data_headers)
test_data = pd.DataFrame(testIn, columns = data_headers)
test_X = test_data.loc[:, "tempo":"mfcc_std"]
test_X = test_X.as_matrix()
test_y = test_data["genre"].as_matrix()

#create decision tree
dt = DecisionTreeClassifier(criterion="entropy",min_impurity_decrease=.01)
dt.fit(train_X,train_y)
pred_y = dt.predict(test_X)
print(pred_y)

knn = KNeighborsClassifier(n_neighbors = 10)
knn.fit(train_X,train_y)
pred_y = knn.predict(test_X)
print(pred_y)

mlp = MLPClassifier(solver="sgd",max_iter=1000,hidden_layer_sizes=1000)
mlp.fit(train_X,train_y)
pred_y = mlp.predict(test_X)
print(pred_y)
