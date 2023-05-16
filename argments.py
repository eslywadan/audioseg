class ArgsParse():

  def __init__(self, **kwargs):
    self.dataset_train_dir = kwargs["dataset_train_dir"]
    self.dataset_validate_dir = kwargs["dataset_validate_dir"]


args = ArgsParse(dataset_train_dir='/content/drive/Shareddrives/LanguageModel/RNN/AI源码解读：卷积神经网络（rnn)深度学习案例（python版）工程文件）/項目5 人聲音樂分離/dataset/MIR-1K/Wavfile',
                 dataset_validate_dir='/content/drive/Shareddrives/LanguageModel/RNN/AI源码解读：卷积神经网络（rnn)深度学习案例（python版）工程文件）/項目5 人聲音樂分離/dataset/MIR-1K/UndividedWavfile'
                 model_dir='model',
                model_filename='svmrnn.ckpt',
                dataset_sr=16000,
                learning_rate=0.0001,
                batch_size=64,
                sample_frames=10,
                iterations=30000,
                dropout_rate=0.95,
                n_fft=1024,
                hop_length_rate=0.25,
                test_temp_dir='./tests/temp')