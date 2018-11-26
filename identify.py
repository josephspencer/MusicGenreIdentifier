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
import os
# command line: python identify.py /path/to/testfile.mp3

def loadData(path):
	y, sr = librosa.core.load(path)
	# get features
	tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
	duration = librosa.get_duration(y=y, sr=sr)
	stftD = np.abs(librosa.stft(y))
	frequencies, d = librosa.ifgram(y, sr=sr)
	cqt = np.abs(librosa.cqt(y, sr=sr))
	iirt = np.abs(librosa.iirt(y))
	tuning = librosa.estimate_tuning(y=y, sr=sr)
	H, P = librosa.decompose.hpss(stftD)
	S, phase = librosa.magphase(librosa.stft(y))
	rms = librosa.feature.rmse(S=S)
	onset_env = librosa.onset.onset_strength(y=y, sr=sr)
	zcr = librosa.feature.zero_crossing_rate(y)
	cent = librosa.feature.spectral_centroid(y=y, sr=sr)
	contrast = librosa.feature.spectral_contrast(S=stftD, sr=sr)
	spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
	rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
	mfcc = librosa.feature.mfcc(y=y, sr=sr)
	# extract further data from features
	beat_frames_std = np.std(beat_frames)
	stftD_std = np.std(stftD)
	stftD_mean = np.mean(stftD)
	freq_std = np.std(frequencies)
	freq_mean =  np.mean(frequencies)
	cqt_std = np.std(cqt)
	cqt_mean = np.mean(cqt)
	iirt_std = np.std(iirt)
	iirt_mean = np.mean(iirt)
	H_std = np.std(H)
	H_mean = np.mean(H)
	P_std = np.std(P)
	P_mean = np.mean(P)
	rms_std = np.std(rms)
	rms_mean = np.mean(rms)
	onset_env_std = np.std(onset_env)
	onset_env_mean = np.mean(onset_env)
	zcr_std = np.std(zcr)
	zcr_mean = np.mean(zcr)
	cent_std = np.std(cent)
	cent_mean = np.mean(cent)
	contrast_std = np.std(contrast)
	contrast_mean = np.mean(contrast)
	spec_bw_std = np.std(spec_bw)
	spec_bw_mean = np.mean(spec_bw)
	rolloff_std = np.std(rolloff)
	rolloff_mean = np.mean(rolloff)
	mfcc_std = np.std(mfcc)
	mfcc_mean = np.mean(mfcc)	
	csv = str(tempo) + "," + str(beat_frames_std) + "," + str(duration) + "," + str(stftD_std) + "," + str(stftD_mean) + "," + str(freq_std) + "," + str(freq_mean) + "," + str(cqt_std) + "," + str(cqt_mean) + "," + str(iirt_std) + "," + str(iirt_mean) + "," + str(H_std) + "," + str(H_mean) + "," + str(P_mean) + "," + str(P_std) + "," + str(rms_std) + "," + str(rms_mean) + "," + str(onset_env_std) + "," + str(onset_env_mean) + "," + str(zcr_std) + "," + str(zcr_mean) + "," + str(cent_std) + "," + str(cent_mean) + "," + str(contrast_std) + "," + str(contrast_mean) + "," + str(spec_bw_std) + "," + str(spec_bw_mean) + "," + str(rolloff_std) + "," + str(rolloff_mean) + "," + str(mfcc_std) + "," + str(mfcc_mean) + ",0"
	return csv

# read in test file, get metrics
testFile = sys.argv[1]
outputCSV = "testFile.csv"
with open(outputCSV, "a+") as fCSV:
	print("began " + str(testFile))
	csv = loadData(testFile)
	fCSV.write(csv)
	fCSV.write('\n')
	print("completed " + str(testFile))
	print('\n')
	fCSV.close()

# read in training data
data_headers = ["tempo", "beat_frames_std", "duration", "stftD_std", "stftD_mean", "freq_std", "freq_mean", "cqt_std", "cqt_mean", "iirt_std", "iirt_mean", "H_std", "H_mean", "P_mean", "P_std", "rms_std", "rms_mean", "onset_env_std", "onset_env_mean", "zcr_std", "zcr_mean", "cent_std", "cent_mean", "contrast_std", "contrast_mean", "spec_bw_std", "spec_bw_mean", "rolloff_std", "rolloff_mean", "mfcc_std", "mfcc_mean", "genre"]

csvIn = pd.read_csv("allTrainCSV.csv", header=None, names=data_headers)
train_data = pd.DataFrame(csvIn, columns = data_headers)
train_X = train_data.loc[:, "tempo":"mfcc_std"]
X_headers = list(train_X)
train_X = train_X.as_matrix()
train_y = train_data["genre"].as_matrix()

# read in test data
testIn = pd.read_csv(outputCSV, header=None, names=data_headers)
test_data = pd.DataFrame(testIn, columns = data_headers)
test_X = test_data.loc[:, "tempo":"mfcc_std"]
test_X = test_X.as_matrix()
test_y = test_data["genre"].as_matrix()

#create decision tree
dt = DecisionTreeClassifier(criterion="entropy",min_impurity_decrease=.01)
dt.fit(train_X,train_y)
pred_y = dt.predict(test_X)
print("Decision Tree: " + str(pred_y))

knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(train_X,train_y)
pred_y = knn.predict(test_X)
print("KNN: " + str(pred_y))

mlp = MLPClassifier(solver="sgd",max_iter=1000,hidden_layer_sizes=1000)
mlp.fit(train_X,train_y)
pred_y = mlp.predict(test_X)
print("MLP: " + str(pred_y))

os.system("rm testFile.csv")
