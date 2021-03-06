import librosa
import sys
import pandas as pd
import numpy
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import os
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDClassifier
from sklearn import tree
from collections import Counter

import warnings
warnings.filterwarnings("ignore")

# command line: python identify.py /path/to/testfile.mp3

def genre(num):
	if num == 1:
		return "Classical"
	elif num == 2:
		return "Country"
	elif num == 3:
		return "Electronic"
	elif num == 4:
		return "Hip Hop"
	elif num == 5:
		return "Jazz"
	elif num == 6:
		return "Rock"
	return None

def perc_format(perc):
	return str(100*perc) + "%"

def get_top_three(probs):
	probs_sorted = sorted(probs, reverse=True)
	genres_sorted = numpy.argsort(probs)[::-1]
	g1 = {"prob": probs_sorted[0], "genre": genre(genres_sorted[0]+1)}
	g2 = {"prob": probs_sorted[1], "genre": genre(genres_sorted[1]+1)}
	g3 = {"prob": probs_sorted[2], "genre": genre(genres_sorted[2]+1)}
	return g1, g2, g3

def loadData(path):
	print ("Examining input file for audio data...\n")
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
test_data = pd.DataFrame([testIn], columns=data_headers)
test_X = test_data.loc[:, "tempo":"mfcc_std"]
test_X = test_X.as_matrix()
test_y = test_data["genre"].as_matrix()

# run Random Forest
forest = RandomForestClassifier(n_estimators = 100)
forest.fit(train_X, train_y)
prediction = genre(forest.predict(test_X))
#g1, g2, g3 = get_top_three(forest.predict_proba(test_X))
pred_probs = forest.predict_proba(test_X)[0]
#pred_probs_sorted = sorted(orest.predict_proba(test_X)[0], reverse=True)
g1,g2,g3 = get_top_three(pred_probs)


# output results
print ("We are " + perc_format(g1["prob"]) + " sure that this song is a " + g1["genre"] + " song.")
print ("Top three possible genres:")
print (g1["genre"] + ": " + perc_format(g1["prob"]))
print (g2["genre"] + ": " + perc_format(g2["prob"]))
print (g3["genre"] + ": " + perc_format(g3["prob"]))

