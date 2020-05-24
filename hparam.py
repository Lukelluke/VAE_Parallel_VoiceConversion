from os import path
d = path.dirname(__file__)
d = '/'.join(d.split('/')[:-2])

audio_num_mel_bins = 80
audio_sample_rate = 16000
num_freq = 513
symbol_size = 256
n_fft = 1024
rescale = True
rescaling_max = 0.999
hop_size = 256
win_size = 1024
frame_shift_ms = None
preemphasize = True
preemphasis = 0.97
min_level_db = -100
ref_level_db = 20
fmin = 55
fmax = 7600
signal_normalization = True
allow_clipping_in_normalization = True
symmetric_mels = True
max_abs_value = 4
power = 1.1
magnitude_power = 1.3
# griffin_lim_iters = 60
griffin_lim_iters = 3
trim_fft_size = 1200
trim_hop_size = 300
trim_top_db = 23
use_lws = False
silence_threshold = 2
trim_silence = True
max_mel_frames = 2048
wavenet_pad_sides = 1
predict_linear = True

phone_list_file = "data/phone_set.json"
bin_data_dir = "liqiao" # 生成特征名字，需要跟train.sh一致
metadata_csv = "meta/liqiao.csv"
#metadata_csv = "linnan.csv"
test_num = 1

text_data_dir = "meta"
#wav_data_dir = "Wave16k/7000000000-7100002500"
wav_data_dir = "wav/train"

#train_csv = "meta/train.csv"
#test_csv = "meta/test.csv"
#phone_set = "res/phone_set.json"
#wav_dir = "wav/train/final"
#test_wav_dir = "wav/test"
#train_feat_dir = "feat/train"
#test_feat_dir = "feat/test"
#data_dir = '%s/1.pub-data'%(d) #"/fast/lxd_room/bjfu-ailab/1.pub-data"

