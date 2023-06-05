from seg_audio import *
import soundfile as sf

sap = SegAudioPrep()
args = parse_arguments()
def get_2channelswavtommv():
    train_file_list = sap.train_file_list

    assert len(train_file_list) == 200

    test_file_path=train_file_list[0]
    wav_mono, wav_music, wav_voice = chans2wavintommv(test_file_path)
    save_file_path = f"{args.test_temp_dir}/"
    sf.write(f"{save_file_path}/test_2channelswavtommv_mono.wav", wav_mono, samplerate=args.dataset_sr)
    sf.write(f"{save_file_path}/test_2channelswavtommv_music.wav", wav_music, samplerate=args.dataset_sr)    
    sf.write(f"{save_file_path}/test_2channelswavtommv_voice.wav", wav_voice, samplerate=args.dataset_sr)

    return wav_mono, wav_music, wav_voice

def get_wav1to3thenspectrum():
    _wav_mono, _wav_music, _wav_voice = get_2channelswavtommv()
    wavs_mono = [_wav_mono]
    wavs_music = [_wav_music]
    wavs_voice = [_wav_voice]
    hop_length = int(args.n_fft*args.hop_length_rate)
    specs_mono, specs_music, specs_voice = wavs2specs(wavs_mono, wavs_music, 
                                                          wavs_voice, n_fft=args.n_fft, hop_length=hop_length)  
    
    return specs_mono, specs_music, specs_voice


# test case to assert the length of self.train_cbatch_specs_mono is the batch size (=64)
def test_batch_size():

    pass

# test case to assert the length of self.train_mbatch_specs_sframes_mono is the sampling frames size 
def test_min_batch():
    pass

def test_spectrumtoxy():
    specs_mono, specs_music, specs_voice = get_wav1to3thenspectrum()

