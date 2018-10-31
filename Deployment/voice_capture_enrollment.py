import librosa
import sounddevice as sd
import os
from configuration import get_config
import numpy as np
import tensorflow as tf
import sys
from utils_deployment import *
import tables

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
	
	
	write_wav(name,recording,config.sr)
	write_npy(name,recording)
	
	print("Playing recording. . .")
	sd.play(recording,config.sr)
	sd.wait()
	print("Done playing.")


	print("shape of recording array: ",recording.shape)
	utterances_spec = preprocess(recording)
	enrolled = enroll(utterances_spec,name)

## Create Tables

import os.path

enrolled_voice_path = 'enrolled_voice_models.hdf5'

if not os.path.isfile(enrolled_voice_path):
	create_dataset(enrolled_voice_path)

	
else:
	hdf5_file = tables.open_file(enrolled_voice_path,mode='a')


	
	
	
	
def create_dataset(enrolled_voice_path)
	hdf5_file = tables.open_file(enrolled_voice_path,mode='w')
	filters = tables.Filters(complevel=5, complib='blosc')
	
	name_enrollment_storage = hdf5_file.create_dataset()
	utterance_enrollment_storage = hdf5_file.create_earray(hdf5_file.root, 'utterance_enrollment'
														  tables.Float32Atom(shape=(), dflt=0.0),
														  shape=enrolled.shape, filters=filters)
	
if __name__ == '__main__':
	main()
