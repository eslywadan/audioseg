import os
import librosa
import numpy as np
import random
import argparse
import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

def parse_arguments(**kwargs):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_train_dir", type=str, 
                        help="資料集訓練資料路徑", default='./dataset/MIR-1K/Wavfile')
    parser.add_argument("--dataset_validate_dir", type=str, 
                        help="資料集驗證資料路徑", default='./dataset/MIR-1K/UndividedWavfile')
    parser.add_argument("--model_dir", type=str, help="模型儲存路徑", default='model')
    parser.add_argument("--model_filename", type=str, help= "模型儲存檔名",default='svmrnn.ckpt')
    parser.add_argument("--dataset_sr", type=int, help="音訊檔案的取樣速率", default=16000)
    parser.add_argument("--learning_rate", type=float, help="學習率", default=0.0001)
    parser.add_argument("--batch_size", type=int, help="小量訓練資料的長度", default=64)
    parser.add_argument("--sample_frames", type=int, help="每次訓練獲取多少幀資料", default=10)
    parser.add_argument("--iterations", type=int, help="訓練迭代次數", default=30000)
    parser.add_argument("--dropout_rate", type=float, help="dropout rate", default=0.95)
    parser.add_argument("--n_fft", type=int, help="短時傅立葉變換的 fft 點數，預設為 windows 長度", default=1024)
    parser.add_argument("--hop_length_rate", type=int, help="取樣的重疊比例",default=0.25)
    parser.add_argument("--test_temp_dir", type=str, help="測試過程暫存資料路徑", default='./tests/temp')
    args = parser.parse_args()
    return args


args = parse_arguments()

class SegAudioPrep():

    def __init__(self):
        ## config - parameters
        self.args = parse_arguments()
        self.dataset_train_dir = self.args.dataset_train_dir
        self.dataset_validate_dir = self.args.dataset_validate_dir
        self.partial = 5
        self.mir1k_sr = self.args.dataset_sr
        self.batchsize = self.args.batch_size
        self.train_file_list = load_file(self.dataset_train_dir, self.partial)
        self.val_file_list = load_file(self.dataset_validate_dir, self.partial)

    def load_train_wavs(self):
        wavs_mono, wavs_music, wavs_voice = load_waves(self.train_file_list, 
                                                       self.mir1k_sr, _mono=False)
        self.train_wavs_mono  = wavs_mono
        self.train_wavs_music = wavs_music
        self.train_wavs_voice = wavs_voice

    def load_val_wavs(self):
        wavs_mono, wavs_music, wavs_voice = load_waves(self.val_file_list, 
                                                       self.mir1k_sr, _mono=False)
        self.val_wavs_mono  = wavs_mono
        self.val_wavs_music = wavs_music
        self.val_wavs_voice = wavs_voice

    def update_train_batch_wavs2specs(self, n_fft, hop_length):
        index = self.get_batch_index(attr='train')
        _wavs_mono = []
        _wavs_music = []
        _wavs_voice = []
        for i in index:
            _wavs_mono.append(self.train_wavs_mono[i])
            _wavs_music.append(self.train_wavs_music[i])
            _wavs_voice.append(self.train_wavs_voice[i])

        specs_mono, specs_music, specs_voice = wavs2specs(_wavs_mono, _wavs_music, 
                                                          _wavs_voice, n_fft=n_fft, hop_length=hop_length)  
        self.train_cbatch_specs_mono = specs_mono
        self.train_cbatch_specs_music = specs_music
        self.train_cbatch_specs_voice = specs_voice

    def update_val_batch_wavs2specs(self, n_fft, hop_length):
        index = self.get_batch_index(attr='val')
        _wavs_mono = []
        _wavs_music = []
        _wavs_voice = []
        for i in index:
            _wavs_mono.append(self.val_wavs_mono[i])
            _wavs_music.append(self.val_wavs_music[i])
            _wavs_voice.append(self.val_wavs_voice[i])

        specs_mono, specs_music, specs_voice = wavs2specs(_wavs_mono, _wavs_music, 
                                                          _wavs_voice, n_fft=n_fft, hop_length=hop_length)  
        self.val_cbatch_specs_mono = specs_mono
        self.val_cbatch_specs_music = specs_music
        self.val_cbatch_specs_voice = specs_voice

    def get_batch_index(self, attr):
        if attr == 'train':
            r = range(0, len(self.train_wavs_mono)-1)
        if attr == 'val':
            r = range(0, len(self.val_wavs_mono)-1)
        rs = random_sample(self.batchsize, r)
        assert len(rs) <= self.batchsize
        return rs    
    
    def acq_batch_min_batch_freq(self, sample_frames, n_fft, hop_length):
         """
         (1) on self.update_train_batch_wavs2specs() is executed, 
         self.train_cbatch_specs_mono
         self.train_cbatch_specs_music
         self.train_cbatch_specs_voice are updated
         (2) sampling mini batch from the spectrum batch
         self.train_data_mono_batch
         self.train_data_music_batch
         self.train_data_voice_batch are reset and appended the new sampling frames
        (3) generate the x & y 
         """
         ## (1) update a new batch 
         self.update_train_batch_wavs2specs(n_fft, hop_length)

         train_mbatch_specs_sframes_mono = []
         train_mbatch_specs_sframes_music = []
         train_mbatch_specs_sframes_voice  = []
         for inx, spec in enumerate(self.train_cbatch_specs_mono):
            num_frames = spec.shape[1]
            assert num_frames >= sample_frames
            ## (2-1) sampling from frames
            start = np.random.randint(num_frames - sample_frames + 1)
            end = start+sample_frames
            train_mbatch_specs_sframes_mono.append(self.train_cbatch_specs_mono[inx][:,start:end])
            train_mbatch_specs_sframes_music.append(self.train_cbatch_specs_music[inx][:,start:end])
            train_mbatch_specs_sframes_voice.append(self.train_cbatch_specs_voice[inx][:,start:end])

         train_mbatch_specs_sframes_mono = np.array(train_mbatch_specs_sframes_mono)
         train_mbatch_specs_sframes_music  = np.array(train_mbatch_specs_sframes_music)
         train_mbatch_specs_sframes_voice = np.array(train_mbatch_specs_sframes_voice)
         self.train_data_mono_batch = train_mbatch_specs_sframes_mono.transpose((0, 2, 1))
         self.train_data_music_batch = train_mbatch_specs_sframes_music.transpose((0, 2, 1))
         self.train_data_voice_batch = train_mbatch_specs_sframes_voice.transpose((0, 2, 1))
         # (3) gen the x & y 
         self.x_mixed_src, _ = separate_magnitude_phase(self.train_data_mono_batch)
         self.y_music_src, _ = separate_magnitude_phase(self.train_data_music_batch)
         self.y_voice_src, _ = separate_magnitude_phase(self.train_data_voice_batch)

def load_file(dir,partial):
    file_list = []
    files = os.listdir(dir)
    i = 0
    for filename in os.listdir(dir):
        i += 1
        if i % partial == 0:
            file_list.append(os.path.join(dir, filename))

    return file_list

def load_waves(_file_list, _sr, _mono=False):
    _wavs_mono = []
    _wavs_music = []
    _wavs_voice = []
    for filename in _file_list:
        wav_mono, wav_music, wav_voice = chans2wavintommv(filename, _sr, _mono)
        _wavs_mono.append(wav_mono)
        _wavs_music.append(wav_music)
        _wavs_voice.append(wav_voice)
    
    return _wavs_mono, _wavs_music, _wavs_voice

# Given wav file in 2 channels trans into mixed(mono) and music and voice 
def chans2wavintommv(filename, _sr=args.dataset_sr, _mono=False):
    wav, _ = librosa.load(filename, sr= _sr, mono = _mono)
    assert(wav.ndim==2 and wav.shape[0]==2)
    wav_mono= librosa.to_mono(wav)*2
    wav_music = wav[0, :]
    wav_voice = wav[1, :]
    return wav_mono,wav_music,wav_voice

def wavs2inputxy():
    pass


def wavs2specs(_wavs_mono, _wavs_music, _wavs_voice, n_fft=1024, hop_length=None):
    stfts_mono = []
    stfts_music = []
    stfts_voice = []
    for wav_mono, wav_music, wav_voice in zip(_wavs_mono, _wavs_music, _wavs_voice):
        stft_mono = librosa.stft((np.asfortranarray(wav_mono)), n_fft = n_fft, hop_length = hop_length)
        stft_music = librosa.stft((np.asfortranarray(wav_music)), n_fft = n_fft, hop_length = hop_length)
        stft_voice = librosa.stft((np.asfortranarray(wav_voice)), n_fft = n_fft, hop_length = hop_length)
        stfts_mono.append(stft_mono)
        stfts_music.append(stft_music)
        stfts_voice.append(stft_voice)

    return stfts_mono, stfts_music, stfts_voice

def random_sample(n:int, r:range, seed=None):
    r_number = int((r.stop - r.start)/r.step)
    if n >  r_number : n = r_number
    return random.sample(r, n)

def separate_magnitude_phase(data):
    return np.abs(data), np.angle(data)


#    根据振幅和相位，得到复数，
#    信号s(t)乘上e^(j*phases)表示信号s(t)移动相位phases
def combine_magnitude_phase(magnitudes, phases):
    return magnitudes * np.exp(1.j * phases)

#    stfts_mono：单声道stft频域数据
#    stfts_music：纯伴奏stft频域数据
#    stfts_music：纯人声stft频域数据
#    batch_size：batch的大小
#    sample_frames：获取多少帧数据
def get_next_batch(stfts_mono, stfts_music, stfts_voice, batch_size = 64, sample_frames = 8):
    stft_mono_batch = list()
    stft_music_batch = list()
    stft_voice_batch = list()
    #    随机选择batch_size个数据
    collection_size = len(stfts_mono) 
    collection_idx = np.random.choice(collection_size, batch_size, replace = True)
    for idx in collection_idx:
        stft_mono = stfts_mono[idx]
        stft_music = stfts_music[idx]
        stft_voice = stfts_voice[idx]
        #    统计有多少帧
        num_frames = stft_mono.shape[1]
        assert  num_frames >= sample_frames
        #    随机获取sample_frames帧数据
        start = np.random.randint(num_frames - sample_frames + 1)
        end = start + sample_frames
        stft_mono_batch.append(stft_mono[:,start:end])
        stft_music_batch.append(stft_music[:,start:end])
        stft_voice_batch.append(stft_voice[:,start:end])
    #    将数据转成np.array，再对形状做一些变换
    # Shape: [batch_size, n_frequencies, n_frames]
    stft_mono_batch = np.array(stft_mono_batch)
    stft_music_batch = np.array(stft_music_batch)
    stft_voice_batch = np.array(stft_voice_batch)
    #    送入RNN的形状要求: [batch_size, n_frames, n_frequencies]
    data_mono_batch = stft_mono_batch.transpose((0, 2, 1))
    data_music_batch = stft_music_batch.transpose((0, 2, 1))
    data_voice_batch = stft_voice_batch.transpose((0, 2, 1))
    return data_mono_batch, data_music_batch, data_voice_batch

class SegAudioModel():
    model_dir = ""
    model_filemame = ""
    dataset_sr = None
    learning_rate = None
    sample_frames = None
    iterations = None
    dropout_rate = None

    def __init__(self, n_fft, num_hidden_units=[256, 256, 256] ):
        # 儲存傳入的參數
        self.n_fft = n_fft
        self.num_features = self.n_fft//2+1
        self.num_hidden_units = num_hidden_units
        self.num_rnn_layers = len(num_hidden_units)
        # 訓練步數
        self.g_step = tf.Variable(0, dtype=tf.int32, name='g_step')
        ## book's code use tf 1.* must be revised to run at tf 2.*
        self.learning_rate = tf.placeholder(tf.float32, shape=[], name='learning_rate')
        self.x_mixed_src = tf.placeholder(tf.float32, shape=[None, None, self.num_features], name= 'x_mixed_src')
        self.y_music_src = tf.placeholder(tf.float32, shape=[None, None, self.num_features], name='y_music_src')
        self.y_voice_src = tf.placeholder(tf.float32, shape=[None, None, self.num_features], name='y_voice_src')
        self.dropout_rate = tf.placeholder(tf.float32)
        # 初始化神經網路
        self.y_pred_music_src, self.y_pred_voice_src = self.network_init()
        self.loss = self.loss_init()
        self.optimizer = self.optimizer_init()
        self.sess = tf.Session()
        self.saver = tf.train.Saver(max_to_keep=1)

    # 損失函數
    def loss_init(self):
        ## https://www.tensorflow.org/api_docs/python/tf/compat/v1/variable_scope 
        ## 
        with tf.variable_scope('loss') as scope:
            # 求方差 (reduce_mean 方法)
            loss = tf.reduce_mean(
                tf.square(self.y_music_src - self.y_pred_music_src)
                + tf.square(self.y_voice_src - self.y_pred_voice_src), name= 'loss'
            )
        return loss

    def optimizer_init(self):
        self.optimizer = tf.train.AdamOptimizer(learning_rate= self.learning_rate).minimize(self.loss)
        
    def network_init(self):
        rnn_layer = []
        # the number of layers is according to the length of num_hidden_units
        for size in self.num_hidden_units:
            # use LSTM to ensure the model accuracy under the big data
            # aviod over-fitting by drop-out
            ##  layer_cell = tf.nn.rnn_cell.LSTMCell(size)
            ## https://github.com/tensorflow/community/blob/master/rfcs/20180920-unify-rnn-interface.md tf1.x -> tf2.x conversion
            layer_cell = tf.nn.rnn_cell.LSTMCell(size)
            layer_cell = tf.nn.rnn_cell.DropoutWrapper(layer_cell, input_keep_prob=self.dropout_rate)
            rnn_layer.append(layer_cell)

        # Create multi-layers && bi-directional rnn
        multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(rnn_layer)
        outputs, state = tf.nn.bidirectional_dynamic_rnn(cell_fw = multi_rnn_cell, cell_bw = multi_rnn_cell, 
                                                         inputs = self.x_mixed_src, dtype= tf.float32)

        out = tf.concat(outputs, 2)
        # full connted nn && use the relu activation function
        y_dense_music_src = tf.layers.dense(
        #y_dense_music_src = tf.keras.layers.dense(
            inputs = out, 
            units = self.num_features, 
            activation = tf.nn.relu,
            name = 'y_dense_music_src'
        )
        y_dense_voice_src = tf.layers.dense(
        #y_dense_voice_src = tf.keras.layers.dense(
            inputs = out,
            units = self.num_features,
            activation = tf.nn.relu,
            name = 'y_dense_voice_src'
        )
        y_music_src = y_dense_music_src / ( y_dense_music_src + y_dense_voice_src + np.finfo(float).eps) * self.x_mixed_src
        y_voice_src = y_dense_voice_src / ( y_dense_music_src + y_dense_voice_src + np.finfo(float).eps) * self.x_mixed_src

        return y_music_src, y_voice_src

    def save(self, directory, filename, global_step):
        if not os.path.exists(directory):
            os.makedirs(directory)
        self.saver.save(self.sess, os.path.join(directory, filename), global_step=global_step)
        return os.path.join(directory, filename)
    
    def load(self, file_dir):
        self.sess.run(tf.global_variables_initializer())
        kpt = tf.train.latest_checkpoint(file_dir)
        print("kpt", kpt)
        startepo = 0
        if kpt != None:
            self.saver.restore(self.sess, kpt)
            ind = kpt.find("-")
            startepo = int(kpt[ind + 1:])
        return startepo
    
    def train(self, x_mixed_src, y_music_src, y_voice_src, learning_rate, dropout_rate):
        # 已訓練步數：
        # step = self.sess.run(self.g_step)
        _, train_loss = self.sess.run([self.optimizer, self.loss],
                        feed_dict = {self.x_mixed_src: x_mixed_src,
                                     self.y_music_src: y_music_src,
                                     self.y_voice_src: y_voice_src,
                                     self.learning_rate: learning_rate,
                                     self.dropout_rate:dropout_rate})
        return train_loss
    
    def validate(self, x_mixed_src, y_music_src, y_voice_src, dropout_rate):
        y_music_src_pred, y_voice_src_pred, validate_loss = self.sess.run([self.y_pred_music_src, 
                        self.y_pred_voice_src, self.loss],
                        feed_dict = {self.x_mixed_src: x_mixed_src,
                                     self.y_music_src: y_music_src,
                                     self.y_voice_src: y_voice_src,
                                     self.dropout_rate:dropout_rate})
        return y_music_src_pred, y_voice_src_pred, validate_loss
    
    def test(self, x_mixed_src, dropout_rate):
        y_music_src_pred, y_voice_src_pred = self.sess.run(
            [self.y_pred_music_src,self.y_pred_voice_src],
            feed_dict = {self.x_mixed_src: x_mixed_src,
                                     self.dropout_rate:dropout_rate})
        return y_music_src_pred, y_voice_src_pred




class SegAudioTrain():

    def __init__(self):
        self.args = parse_arguments()

        
        if not os.path.exists(self.args.dataset_train_dir) or not os.path.exists(self.args.dataset_validate_dir):
            raise NameError('資料集路径"./dataset/MIR-1K/Wavfile"或"./dataset/MIR-1K/UndividedWavfile"不存在!')

 
        self.n_fft = self.args.n_fft
        # 冗餘度
        self.hop_length = self.n_fft // 4

        # Model parameters
        # Learning rate
        self.learning_rate = self.args.learning_rate
        # 用於創建 rnn 結點數
        self.num_hidden_units = [1024, 1024, 1024, 1024, 1024]
        # batch size
        self.batch_size = self.args.batch_size
        self.sample_frames = self.args.sample_frames
        self.iterations = self.args.iterations
        self.dropout_rate = self.args.dropout_rate
        self.model_dir = self.args.model_dir
        self.model_filename = self.args.model_filename
        self.data = SegAudioPrep()
        self.mdl = SegAudioModel(self.args.n_fft, num_hidden_units = self.num_hidden_units)
        self.mdl.optimizer_init()
        self.mdl.loss_init()
        # self.mdl.network_init()

    def main_de(self):
        #1. 导入需要训练的数据集文件路径，存到列表中即可
        self.train_file_list = self.data.train_file_list
        self.valid_file_list = self.data.val_file_list
        self.mir1k_sr = self.data.mir1k_sr

        # train_wavs_mono, train_wavs_music, train_wavs_voice are init on load_train_wavs() is executed 
        self.data.load_train_wavs()
        # train_wavs_mono is accessed by self.data.train_wavs_mono
        # train_wavs_music is accessed by self.data.train_wavs_music
        # train_wavs_voice is accessed by self.data.train_wavs_voice
        
        # train_cbatch_specs_mono, train_cbatch_specs_music, train_cbatch_specs_voice are transformed to frequence domain
        self.data.update_train_batch_wavs2specs(self.n_fft, self.hop_length)
        # train_cbatch_specs_mono is accessed by self.data.train_cbatch_specs_mono 
        # train_cbatch_specs_music is accessed by self.data.train_cbatch_specs_music
        # train_cbatch_specs_voice is accessed by self.data.train_cbatch_specs_voice


        self.data.load_val_wavs()
        # val_wavs_mono is accessed by self.data.val_wavs_mono
        # val_wavs_music is accessed by self.data.val_wavs_music
        # val_wavs_voice is accessed by self.data.val_wavs_voice

        self.data.update_val_batch_wavs2specs(self.n_fft, self.hop_length)
        # self.data.val_cbatch_specs_mono 
        # self.data.val_cbatch_specs_music
        # self.data.val_cbatch_specs_voice

    def start_mdl(self):
        # load model or init parameters
        self.startepo = self.mdl.load(file_dir=self.model_dir)
        print('startepo:' + str(self.startepo))

    def start_train(self):
        index = 0

        print(f"training iteration: {self.iterations}")
        for i in (range(self.iterations)):
            # start from the lattest broken point
            if i < self.startepo:
                continue
            #wavs_mono_train_cut = list()
            #wavs_music_train_cut = list()
            #wavs_voice_train_cut = list()

            #self.gen_xy_train_data(i, wavs_mono_train_cut, wavs_music_train_cut, wavs_voice_train_cut)
            #送入神经网络，开始训练

            self.data.acq_batch_min_batch_freq(self.sample_frames, self.n_fft, self.hop_length)
            train_loss = self.mdl.train(x_mixed_src = self.data.x_mixed_src, y_music_src = self.data.y_music_src, y_voice_src = self.data.y_voice_src,
                                    learning_rate = self.learning_rate, dropout_rate = self.dropout_rate)
 
        # 每10步输出一次训练结果的损失值
            if i % 10 == 0:
                print('Step: %d Train Loss: %f' %(i, train_loss))
        # 每200步输出一次测试结果
            if i % 200 == 0:
                print('==============================================')
                data_mono_batch, data_music_batch, data_voice_batch = get_next_batch(
                    stfts_mono = self.data.val_cbatch_specs_mono, stfts_music = self.data.val_cbatch_specs_music,
                    stfts_voice = self.data.val_cbatch_specs_voice, batch_size = self.batch_size, sample_frames = self.sample_frames)
                x_mixed_src, _ = separate_magnitude_phase(data = data_mono_batch)
                y_music_src, _ = separate_magnitude_phase(data = data_music_batch)
                y_voice_src, _ = separate_magnitude_phase(data = data_voice_batch)
                y_music_src_pred, y_voice_src_pred, validate_loss = self.mdl.validate(x_mixed_src = x_mixed_src,
                        y_music_src = y_music_src, y_voice_src = y_voice_src, dropout_rate = self.dropout_rate)
                print('Step: %d Validation Loss: %f' %(i, validate_loss))
                print('==============================================')
        # 每200步保存一次模型
            if i % 200 == 0:
                self.mdl.save(directory = self.model_dir, filename = self.model_filename, global_step=i)


    def gen_xy_train_data(self, i, wavs_mono_train_cut, wavs_music_train_cut, wavs_voice_train_cut):
        print(f"gen x,y train data for iteration {i}")
        for seed in range(self.batch_size):
            index = np.random.randint(0, len(self.data.train_wavs_mono))
            wavs_mono_train_cut.append(self.data.train_wavs_mono[index])
            wavs_music_train_cut.append(self.data.train_wavs_music[index])
            wavs_voice_train_cut.append(self.data.train_wavs_voice[index])
            
            #短时傅里叶变换，将选取的音频数据转到频域
        
        stfts_mono_train_cut, stfts_music_train_cut, stfts_voice_train_cut = wavs2specs(
            _wavs_mono = wavs_mono_train_cut, _wavs_music = wavs_music_train_cut, _wavs_voice = wavs_voice_train_cut,
            n_fft = self.n_fft, hop_length = self.hop_length)

        data_mono_batch, data_music_batch, data_voice_batch = get_next_batch(
        stfts_mono = stfts_mono_train_cut, stfts_music = stfts_music_train_cut, stfts_voice = stfts_voice_train_cut,
        batch_size = self.batch_size, sample_frames = self.sample_frames)
            #获取频率值
        self.x_mixed_src, _ = separate_magnitude_phase(data = data_mono_batch)
        self.y_music_src, _ = separate_magnitude_phase(data = data_music_batch)
        self.y_voice_src, _ = separate_magnitude_phase(data = data_voice_batch)




    def main(self):
        self.main_de()
        self.start_mdl()
        self.start_train()d