class hyperparams:
    def __init__(self):
        #################################################################################
        #                                                                               #
        #                            Preprocess Hyperparams                             #
        #                                                                               #
        #################################################################################

        # ------------------------ Paths And Directory ------------------------ #
        self.CSV_PATH = './data/dataset.csv'
        self.SOURCE_CSV_PATH = './paralleldata/source.csv'  # 平行中的发起者：source
        self.TOTAL_CSV_PATH = './paralleldata/total.csv'  # 保存着所有的训练数据（source 和 target）
        # Format: spk|fpath
        # Example: xyb|0001.wav
        self.TRAIN_DATASET_PATH = './paralleldata/train_data'
        self.PARALLEL_TRAIN_DATASET_PATH = './paralleldata/train_data'
        self.TEST_DATASET_PATH = './paralleldata/test_data'
        self.WAVS_PATH = './data/wavs'

        self.TOTAL_WAVS_PATH = './paralleldata/TOTAL'
        self.SOURCE_WAVS_PATH = './paralleldata/VCC2SF1'  # 新增加的：source语音路径
        self.TARGET_WAVS_PATH = './paralleldata/VCC2TM1'  # 新增加的：target语音路径
        self.TRAIN_RATE = 1
        # ------------------------ Setting And Hyperparams -------------------- #
        self.MULTI_PROCESS = True
        self.CPU_RATE = 1
        self.SR = 16000
        self.N_FFT = 1024
        self.CODED_DIM = 80  # 压缩成80维 mel
        self.SPK_NUM = 14
        self.PARALLEL_SPK_NUM = 2

        #################################################################################
        #                                                                               #
        #                               Train Hyperparams                               #
        #                                                                               #
        #################################################################################

        # ------------------------ Paths And Directory ------------------------ #
        self.LOG_DIR = './logs'
        self.MODEL_DIR = './models'
        # ------------------------ Setting And Hyperparams -------------------- #
        self.NUM_EPOCHS = 150
        self.BATCH_SIZE = 16
        self.EMBED_SIZE = 256
        self.G_LR = 0.001
        self.D_LR = 0.001
        # VAE LOSS WEIGHT
        self.X0 = 2500
        self.K = 0.0025
        self.DROPOUT_RATE = 0.5
        self.VAE_GAUSSION_UNITS = 16
        self.PER_STEPS = 100
