import librosa
import sounddevice as sd
import os
from configuration import get_config
import numpy as np
import tensorflow as tf
from utils_deployment import *
import sys

config = get_config()

def main():
	print("Who are you?")
	name = input()
	print("Hi %s, Are you ready? (Press enter to continue. Records immediately)" %name)
	ready = input()
	duration = config.duration
	print("Recording . . . (Plays immediately after recording)")
	recording = sd.rec(int(duration * config.sr), samplerate=config.sr, channels=1)
	sd.wait()
	recording = np.reshape(recording,duration*config.sr)
	print("Done Recording.")

	print("Playing recording. . .")
	sd.play(recording,config.sr)
	sd.wait()
	print("Done.")

	# enroll_path = os.path.join("enroll_voice",name)
	# create_folder(enroll_path)
	print("shape of recording array: ",recording.shape)
	utterances_spec = preprocess(recording)
	verify(utterances_spec,name)

	
if __name__ == '__main__':
	main()
