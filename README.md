# MusicGenreIdentifier
Artificial Intelligence Final Project
Joseph Spencer and Ryan Loizzo

# Installation Instructions
In order to run the programs contained with in this project, the following python packages are needed:
- librosa
- numpy
- glob
- sys
- pandas
- scipy
- sklearn
- collections

# Python Scripts
- extractdata_train.py was the python script we used in order to extract data from the directories containing the song files that we wanted to use and outputted the data of all the song files into a specified CSV file. The command line argument for using this file was follows: python extractdata_train.py /path/to/song/directory outputtedCSVname.csv #

- classifier.py was the python script that we used to gather data on our testing set in comparison to our training set. Within this script, we openned the training set csv as well as the csv file for the desired testing set and ran the classifiers over each song within the CSV file. This script then outputted the genre classification from each song, and we calculated the accuracy based on these outputted genres

- identify.py is the main python script that a user would use for our project. This script will take in an mp3 file and parse it for the desired data. After extracting the desired data, the script will run the classifiers on the mp3 file data and compare it to the training set data in order to output a final genre classification for the specific song. To run this file, the following command line format is used: python identify.py /path/to/song/file.mp3

# Genre Labels:
1 - classical
2 - country
3 - electronic
4 - hiphop
5 - jazz
6 - rock
