from seg_audio import SegAudioPrep, SegAudioModel, SegAudioTrain
# init a segment audio training instance
sat = SegAudioTrain()
# sat.mir1k_sr = sat.data.mir1k_sr = sat.data.args.dataset_sr = pars_arguments.dataset_sr = 16000
assert sat.dropout_rate == 0.95

sat.data.load_train_wavs()
assert sat.data.train_wavs_mono is not None
assert sat.data.train_wavs_music is not None
assert sat.data.train_wavs_voice is not None

sat.data.update_train_batch_wavs2specs(sat.n_fft, sat.hop_length)
assert sat.data.train_cbatch_specs_mono is not None
assert sat.data.train_cbatch_specs_music is not None
assert sat.data.train_cbatch_specs_voice is not None

sat.data.load_val_wavs()
assert sat.data.val_wavs_mono is not None
assert sat.data.val_wavs_music is not None
assert sat.data.val_wavs_voice is not None

sat.data.update_val_batch_wavs2specs(sat.n_fft, sat.hop_length)
assert sat.data.val_cbatch_specs_mono is not None
assert sat.data.val_cbatch_specs_music is not None
assert sat.data.val_cbatch_specs_voice is not None

sat.start_mdl()




sat.start_train()






