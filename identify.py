import librosa
import sys
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import os
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDClassifier
from sklearn import tree
from collections import Counter

# command line: python identify.py /path/to/testfile.mp3

def genre(num):
	if num == 1:
		return "classical"
	elif num == 2:
		return "country"
	elif num == 3:
		return "electronic"
	elif num == 4:
		return "hip hop"
	elif num == 5:
		return "jazz"
	elif num == 6:
		return "rock"
	return None

def find_dist(point, centroid):
	running_sum = 0.0
	for i in range(len(point)):
		running_sum += float(pow((point[i] - centroid[i]),2))
	return pow(running_sum,0.5)

def loadData(path):
	print("Examining input file for audio data...\n")
	y, sr = librosa.core.load(path)
	# get features
	tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
	duration = librosa.get_duration(y=y, sr=sr)
	stftD = np.abs(librosa.stft(y))
	iirt = np.abs(librosa.iirt(y))
	H, P = librosa.decompose.hpss(stftD)
	S, phase = librosa.magphase(librosa.stft(y))
	rms = librosa.feature.rmse(S=S)
	cent = librosa.feature.spectral_centroid(y=y, sr=sr)
	contrast = librosa.feature.spectral_contrast(S=stftD, sr=sr)
	spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
	rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
	mfcc = librosa.feature.mfcc(y=y, sr=sr)
	# extract further data from features
	beat_frames_std = np.std(beat_frames)
	stftD_std = np.std(stftD)
	iirt_std = np.std(iirt)
	H_std = np.std(H)
	rms_mean = np.mean(rms)
	cent_std = np.std(cent)
	cent_mean = np.mean(cent)
	spec_bw_std = np.std(spec_bw)
	spec_bw_mean = np.mean(spec_bw)
	rolloff_std = np.std(rolloff)
	rolloff_mean = np.mean(rolloff)
	mfcc_std = np.std(mfcc)
	mfcc_mean = np.mean(mfcc)	
	data_list = str(tempo) + "," + str(beat_frames_std) + "," + str(duration) + "," + str(stftD_std) + "," + str(iirt_std) + "," + str(H_std) + "," + str(rms_mean) + "," + str(cent_std) + "," + str(cent_mean) + "," + str(spec_bw_std) + "," + str(spec_bw_mean) + "," + str(rolloff_std) + "," + str(rolloff_mean) + "," + str(mfcc_std) + "," + str(mfcc_mean) + ",0"
	return data_list

# read in test file, get metrics
testFile = sys.argv[1]
song_data = loadData(testFile)

# read in training data
data_headers = ["tempo", "beat_frames_std", "duration", "stftD_std", "iirt_std", "H_std", "rms_mean", "cent_std", "cent_mean", "spec_bw_std", "spec_bw_mean", "rolloff_std", "rolloff_mean", "mfcc_std", "mfcc_mean", "genre"]

csvIn = pd.read_csv("CutallTrainCSV.csv", header=None, names=data_headers)
train_data = pd.DataFrame(csvIn, columns = data_headers)
train_X = train_data.loc[:, "tempo":"mfcc_std"]
X_headers = list(train_X)
train_X = train_X.as_matrix()
train_y = train_data["genre"].as_matrix()

# convert song data to pandas data frame for Machine Learning tool
testIn = song_data.split(",")
#testIn = pd.read_csv("testFile.csv", header=None, names=data_headers)
test_data = pd.DataFrame([testIn], columns=data_headers)
test_X = test_data.loc[:, "tempo":"mfcc_std"]
test_X = test_X.as_matrix()
test_y = test_data["genre"].as_matrix()

# run Random Forest
forest = RandomForestClassifier(n_estimators = 100)
forest.fit(train_X, train_y)
prediction = genre(forest.predict(test_X))
prediction_prob = str(100*max(forest.predict_proba(test_X)[0])) + "%"

print ("We are " + prediction_prob + " sure that this song is a " + prediction + " song.")
print forest.predict_proba(test_X)
