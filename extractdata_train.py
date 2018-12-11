import librosa
import numpy as np
import glob
import sys
import warnings
warnings.filterwarnings("ignore")

# This file is used to extract the data for the training set. This program is run by passing in the name of the directory that contains all of the mp3 files that will be used as the training set, as well as the name of the desired outputted csv file and the genre indicator.

def loadData(path, passedInGenre):
	y, sr = librosa.core.load(path)
	# get features
	tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
	duration = librosa.get_duration(y=y, sr=sr)
	stftD = np.abs(librosa.stft(y))
	frequencies, d = librosa.ifgram(y, sr=sr)
	iirt = np.abs(librosa.iirt(y))
	H, P = librosa.decompose.hpss(stftD)
	S, phase = librosa.magphase(librosa.stft(y))
	rms = librosa.feature.rmse(S=S)
	cent = librosa.feature.spectral_centroid(y=y, sr=sr)
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
	csv = str(tempo) + "," + str(beat_frames_std) + "," + str(duration) + "," + str(stftD_std) + "," + str(iirt_std) + "," + str(H_std) + "," + str(rms_mean) + "," + str(cent_std) + "," + str(cent_mean) + "," + str(spec_bw_std) + "," + str(spec_bw_mean) + "," + str(rolloff_std) + "," + str(rolloff_mean) + "," + str(mfcc_std) + "," + str(mfcc_mean) + "," + str(passedInGenre)
	return csv

# handle command line arguements
directory = sys.argv[1]
directory += "/*" # add /* in order to iterate through each mp3 file in directory
outputCSV = sys.argv[2]
passInGenre = sys.argv[3]
print(directory)
with open(sys.argv[2], "a+") as fCSV: # create and append to CSV file
	for path in glob.iglob(directory): # iterate through directory and write data to file
		print("began " + str(path))
		csv = loadData(path, passInGenre)
		fCSV.write(csv)
		fCSV.write('\n')
		print("completed " + str(path))
		print('\n')
