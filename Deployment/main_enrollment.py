from utils_deployment import *
from configuration import get_config
import librosa
import numpy as np
import os

config = get_config()   # get arguments from parser

def main():
	audio_path = 'test_voice'
	
	# enroll_folder = 'enrolled_voices'

	utter_min_len = (config.tisv_frame * config.hop + config.window) * config.sr  # lower bound of utterance length

	for speaker in os.listdir(audio_path):
		speaker_path = os.path.join(audio_path,speaker)
		utterances_spec = []
		print(speaker_path)
		if speaker == '.DS_Store':
			continue
		print('+'*50)
		print('To enroll',speaker,'.')
		for utter_name in os.listdir(speaker_path):
			utter_path = os.path.join(speaker_path, utter_name)         # path of each utterance
			# print(utter_path)
			utter, sr = librosa.core.load(utter_path, config.sr)        # load utterance audio
			intervals = librosa.effects.split(utter, top_db=20)         # voice activity detection
			for interval in intervals:
				if (interval[1]-interval[0]) > utter_min_len:           # If partial utterance is sufficient long,
					utter_part = utter[interval[0]:interval[1]] # save first and last 180 frames of spectrogram.
					S = librosa.core.stft(y=utter_part, n_fft=config.nfft,
						win_length=int(config.window * sr), hop_length=int(config.hop * sr))
					S = np.abs(S) ** 2
					mel_basis = librosa.filters.mel(sr=config.sr, n_fft=config.nfft, n_mels=40)
					S = np.log10(np.dot(mel_basis, S) + 1e-6)           # log mel spectrogram of utterances

					utterances_spec.append(S[:, :config.tisv_frame])    # first 180 frames of partial utterance
					utterances_spec.append(S[:, -config.tisv_frame:])   # last 180 frames of partial utterance

		utterances_spec = np.array(utterances_spec)
		print(utterances_spec.shape)
		enroll(utterances_spec,speaker)
		

		# np.save(os.path.join(enroll_folder, "speaker_%s.npy"%(speaker)), utterances_spec)


def enroll(utters,speaker,path=config.model_path):
	tf.reset_default_graph()

	# draw graph	
	enroll = tf.placeholder(shape=[None, config.N*config.M, 40], dtype=tf.float32) # enrollment batch (time x batch x n_mel)
	batch = enroll
	# embedding lstm (3-layer default)
	with tf.variable_scope("lstm"):
		lstm_cells = [tf.contrib.rnn.LSTMCell(num_units=config.hidden, num_proj=config.proj) for i in range(config.num_layer)]
		lstm = tf.contrib.rnn.MultiRNNCell(lstm_cells)    # make lstm op and variables
		outputs, _ = tf.nn.dynamic_rnn(cell=lstm, inputs=batch, dtype=tf.float32, time_major=True)   # for TI-VS must use dynamic rnn
		embedded = outputs[-1]                            # the last ouput is the embedded d-vector
		embedded = normalize(embedded)

	# enrollment embedded vectors (speaker model)
	enroll_embed = normalize(tf.reduce_mean(tf.reshape(embedded[:config.N*config.M, :], shape= [config.N, config.M, -1]), axis=1))	
	saver = tf.train.Saver(var_list=tf.global_variables())

	with tf.Session() as sess:
		tf.global_variables_initializer().run()

		# load model
		print("model path :", path)
		ckpt = tf.train.get_checkpoint_state(checkpoint_dir=os.path.join(path, "Check_Point"))
		ckpt_list = ckpt.all_model_checkpoint_paths
		loaded = 0
		for model in ckpt_list:
			if config.model_num == int(model[-1]):    # find ckpt file which matches configuration model number
				print("ckpt file is loaded !", model)
				loaded = 1
				saver.restore(sess, model)  # restore variables from selected ckpt file
				break

		if loaded == 0:
			raise AssertionError("ckpt file does not exist! Check config.model_num or config.model_path.")

		try:
			utter_index = np.random.randint(0, utters.shape[0], config.M)   # select M utterances per speaker
			utter_batch = utters[utter_index]     # each speakers utterance [M, n_mels, frames] is appended
			utter_batch = np.transpose(utter_batch, axes=(2,0,1))     # transpose [frames, batch, n_mels]
			print(utter_batch.shape)
			enroll = sess.run(enroll_embed, feed_dict={enroll:utter_batch})
			print('enroll shape: ',enroll.shape)
			print('Enrolled: ',speaker)

			speaker_model_path = 'MODEL' ## put in config?
			create_folder(speaker_model_path)
			np.save(os.path.join(speaker_model_path,'%s.npy'%speaker),enroll)
		except Exception as e:
			print(e)

if __name__ == '__main__':
	main()