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
import numpy as np
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDClassifier
from sklearn import tree

def find_dist(point, centroid):
	running_sum = 0.0
	for i in range(len(point)):
		running_sum += float(pow((point[i] - centroid[i]),2))
	return pow(running_sum,0.5)

# read in training data
#data_headers = ["tempo", "beat_frames_std", "duration", "stftD_std", "stftD_mean", "freq_std", "freq_mean", "cqt_std", "cqt_mean", "iirt_std", "iirt_mean", "H_std", "H_mean", "P_mean", "P_std", "rms_std", "rms_mean", "onset_env_std", "onset_env_mean", "zcr_std", "zcr_mean", "cent_std", "cent_mean", "contrast_std", "contrast_mean", "spec_bw_std", "spec_bw_mean", "rolloff_std", "rolloff_mean", "mfcc_std", "mfcc_mean", "genre"]
data_headers = ["tempo", "beat_frames_std", "duration", "stftD_std", "stftD_mean", "freq_std", "freq_mean", "cqt_std", "cqt_mean", "iirt_std", "iirt_mean", "H_std", "H_mean", "P_mean", "P_std", "genre"]

csvIn = pd.read_csv("CutallTrainCSV.csv", header=None, names=data_headers)
train_data = pd.DataFrame(csvIn, columns = data_headers)
#train_X = train_data.loc[:, "tempo":"mfcc_std"]
train_X = train_data.loc[:, "tempo":"P_std"]
X_headers = list(train_X)
train_X = train_X.as_matrix()
train_y = train_data["genre"].as_matrix()

# read in test data
testIn = pd.read_csv("CuttestingCSV.csv", header=None, names=data_headers)
test_data = pd.DataFrame(testIn, columns = data_headers)
#test_X = test_data.loc[:, "tempo":"mfcc_std"]
test_X = test_data.loc[:, "tempo":"P_std"]
test_X = test_X.as_matrix()
test_y = test_data["genre"].as_matrix()

#create decision tree
dt = DecisionTreeClassifier(criterion="gini",min_impurity_decrease=.01)
dt.fit(train_X,train_y)
pred_y = dt.predict(test_X)
print("Decision tree: " + str(pred_y))

# kNN
knn = KNeighborsClassifier(n_neighbors = 5, weights="distance")
knn.fit(train_X,train_y)
pred_y = knn.predict(test_X)
print("kNN: " + str(pred_y))

#MLP
mlp = MLPClassifier(solver="adam",max_iter=1000,hidden_layer_sizes=1000)
mlp.fit(train_X,train_y)
pred_y = mlp.predict(test_X)
print("mlp: " + str(pred_y))

# kmeans clustering
km = KMeans(n_clusters = 6, init='k-means++', max_iter=1000, algorithm="elkan")
km.fit(train_X,train_y)
ans = []
for i in range(len(test_X)):
	point = test_X[i]
	d1 = find_dist(point,km.cluster_centers_[0])
	d2 = find_dist(point,km.cluster_centers_[1])
	d3 = find_dist(point,km.cluster_centers_[2])
	d4 = find_dist(point,km.cluster_centers_[3])
	d5 = find_dist(point,km.cluster_centers_[4])
	d6 = find_dist(point,km.cluster_centers_[5])
	distances = [d1, d2, d3, d4, d5, d6]
	ans.append(distances.index(min(distances)) + 1)
print("kmeans: " + str(ans))

#SGD
clf = SGDClassifier(loss="log", penalty="l1", max_iter=1000)
clf.fit(train_X, train_y)
pred_y = clf.predict(test_X)
print("sgd: " + str(pred_y))
