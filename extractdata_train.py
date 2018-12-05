import librosa
import numpy as np
import glob
import sys


def loadData(path, passedInGenre):
	y, sr = librosa.core.load(path)
	# get features
	tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
	duration = librosa.get_duration(y=y, sr=sr)
	stftD = np.abs(librosa.stft(y))
	frequencies, d = librosa.ifgram(y, sr=sr)
	#cqt = np.abs(librosa.cqt(y, sr=sr))
	iirt = np.abs(librosa.iirt(y))
	#tuning = librosa.estimate_tuning(y=y, sr=sr)
	H, P = librosa.decompose.hpss(stftD)
	S, phase = librosa.magphase(librosa.stft(y))
	rms = librosa.feature.rmse(S=S)
	#onset_env = librosa.onset.onset_strength(y=y, sr=sr)
	#zcr = librosa.feature.zero_crossing_rate(y)
	cent = librosa.feature.spectral_centroid(y=y, sr=sr)
	#contrast = librosa.feature.spectral_contrast(S=stftD, sr=sr)
	spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
	rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
	mfcc = librosa.feature.mfcc(y=y, sr=sr)
	# extract further data from features
	beat_frames_std = np.std(beat_frames)
	stftD_std = np.std(stftD)
	#stftD_mean = np.mean(stftD)
	#freq_std = np.std(frequencies)
	#freq_mean =  np.mean(frequencies)
	#cqt_std = np.std(cqt)
	#cqt_mean = np.mean(cqt)
	iirt_std = np.std(iirt)
	#iirt_mean = np.mean(iirt)
	H_std = np.std(H)
	#H_mean = np.mean(H)
	#P_std = np.std(P)
	#P_mean = np.mean(P)
	#rms_std = np.std(rms)
	rms_mean = np.mean(rms)
	#onset_env_std = np.std(onset_env)
	#onset_env_mean = np.mean(onset_env)
	#zcr_std = np.std(zcr)
	#zcr_mean = np.mean(zcr)
	cent_std = np.std(cent)
	cent_mean = np.mean(cent)
	#contrast_std = np.std(contrast)
	#contrast_mean = np.mean(contrast)
	spec_bw_std = np.std(spec_bw)
	spec_bw_mean = np.mean(spec_bw)
	rolloff_std = np.std(rolloff)
	rolloff_mean = np.mean(rolloff)
	mfcc_std = np.std(mfcc)
	mfcc_mean = np.mean(mfcc)	
	#csv = str(tempo) + "," + str(beat_frames_std) + "," + str(duration) + "," + str(stftD_std) + "," + str(stftD_mean) + "," + str(freq_std) + "," + str(freq_mean) + "," + str(cqt_std) + "," + str(cqt_mean) + "," + str(iirt_std) + "," + str(iirt_mean) + "," + str(H_std) + "," + str(H_mean) + "," + str(P_mean) + "," + str(P_std) + "," + str(rms_std) + "," + str(rms_mean) + "," + str(onset_env_std) + "," + str(onset_env_mean) + "," + str(zcr_std) + "," + str(zcr_mean) + "," + str(cent_std) + "," + str(cent_mean) + "," + str(contrast_std) + "," + str(contrast_mean) + "," + str(spec_bw_std) + "," + str(spec_bw_mean) + "," + str(rolloff_std) + "," + str(rolloff_mean) + "," + str(mfcc_std) + "," + str(mfcc_mean)
	#print(csv)
#	extractedArrays = [[beat_frames], [stftD], [frequencies], [cqt], [iirt], [H], [P], [rms], [onset_env], [zcr], [cent], [contrast], [spec_bw], [rolloff], [mfcc]]
	csv = str(tempo) + "," + str(beat_frames_std) + "," + str(duration) + "," + str(stftD_std) + "," + str(iirt_std) + "," + str(H_std) + "," + str(rms_mean) + "," + str(cent_std) + "," + str(cent_mean) + "," + str(spec_bw_std) + "," + str(spec_bw_mean) + "," + str(rolloff_std) + "," + str(rolloff_mean) + "," + str(mfcc_std) + "," + str(mfcc_mean) + "," + str(passedInGenre)
	return csv

#directory = "/Users/joeyspencer/AI/AI_final_project/songfiles/classical/*"
directory = sys.argv[1]
directory += "/*"
outputCSV = sys.argv[2]
passInGenre = sys.argv[3]
print(directory)
with open(sys.argv[2], "a+") as fCSV:
	for path in glob.iglob(directory):
		print("began " + str(path))
		csv = loadData(path, passInGenre)
		fCSV.write(csv)
		fCSV.write('\n')
		print("completed " + str(path))
		print('\n')
