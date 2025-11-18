from yacs.config import CfgNode as CN

_C = CN()
# MODEL
_C.MODEL = CN()
_C.MODEL.DEVICE = "cuda" # Using cuda or cpu for training
_C.MODEL.DEVICE_ID = '0' # ID number of GPU
_C.MODEL.DIST_TRAIN = False # If train with multi-gpu ddp mode, options: 'True', 'False'

_C.MODEL.IMG = CN()
_C.MODEL.IMG.NAME = 'resnet50' # Name of backbone
_C.MODEL.IMG.LAST_STRIDE = 1 # Last stride of backbone
_C.MODEL.IMG.PRETRAIN_PATH = '' # Path to pretrained model of backbone
_C.MODEL.IMG.PRETRAIN_CHOICE = 'imagenet'
_C.MODEL.IMG.NECK = 'bnneck' # If train with BNNeck, options: 'bnneck' or 'no'
_C.MODEL.IMG.IF_WITH_CENTER = 'no' # If train loss include center loss, options: 'yes' or 'no'. Loss with center loss has different optimizer configuration
_C.MODEL.IMG.ID_LOSS_TYPE = 'softmax'
_C.MODEL.IMG.ID_LOSS_WEIGHT = 1.0
_C.MODEL.IMG.TRIPLET_LOSS_WEIGHT = 1.0
_C.MODEL.IMG.I2T_LOSS_WEIGHT = 1.0
_C.MODEL.IMG.METRIC_LOSS_TYPE = 'triplet'
_C.MODEL.IMG.NO_MARGIN = False # If train with soft triplet loss, options: 'True', 'False'
_C.MODEL.IMG.IF_LABELSMOOTH = 'on' # If train with label smooth, options: 'on', 'off'
_C.MODEL.IMG.COS_LAYER = False # If train with arcface loss, options: 'True', 'False'
_C.MODEL.IMG.DROP_PATH = 0.1
_C.MODEL.IMG.DROP_OUT = 0.0
_C.MODEL.IMG.ATT_DROP_RATE = 0.0
_C.MODEL.IMG.TRANSFORMER_TYPE = 'None'
_C.MODEL.IMG.STRIDE_SIZE = [16, 16]
_C.MODEL.IMG.SIE_COE = 3.0
_C.MODEL.IMG.SIE_CAMERA = False
_C.MODEL.IMG.SIE_VIEW = False

_C.MODEL.SKE = CN()
_C.MODEL.SKE.HIDDEN_DIM = 128
_C.MODEL.SKE.JOINT_NUM = 17
_C.MODEL.SKE.TIME_STEP = 6
_C.MODEL.SKE.K_POS_ENC = 10
_C.MODEL.SKE.NUM_LAYERS = 2
_C.MODEL.SKE.N_HEADS = 8
_C.MODEL.SKE.DROPOUT = 0.3
_C.MODEL.SKE.USE_POS_ENC = True  # pos enc
_C.MODEL.SKE.USE_S_PROMPT = True 
_C.MODEL.SKE.USE_T_PROMPT = True
_C.MODEL.SKE.S_TYPE = "l2"
_C.MODEL.SKE.T_TYPE = "l2"
_C.MODEL.SKE.PROMPT_LAMBDA = 0.5
_C.MODEL.SKE.TEMP_SEQ = 0.07  # Sequence-level temperature
_C.MODEL.SKE.TEMP_SKE = 0.07  # Skeleton-level temperature
_C.MODEL.SKE.SEQ_LAMBDA = 0.5
_C.MODEL.SKE.GPC_LAMBDA = 0.5


# INPUT
_C.INPUT = CN()
_C.INPUT.SIZE_TRAIN = [384, 128] # Size of the image during training
_C.INPUT.SIZE_TEST = [384, 128] # Size of the image during test
_C.INPUT.PROB = 0.5 # Random probability for image horizontal flip
_C.INPUT.RE_PROB = 0.5 # Random probability for random erasing
_C.INPUT.PIXEL_MEAN = [0.485, 0.456, 0.406] # Values to be used for image normalization
_C.INPUT.PIXEL_STD = [0.229, 0.224, 0.225] # Values to be used for image normalization
_C.INPUT.PADDING = 10 # Value of padding size
_C.INPUT.SEQ_LEN = 4
_C.INPUT.SAMPLE_STRIDE = 4
_C.INPUT.TEST_FRAMES = 4
_C.INPUT.SEQ_SRD = 4


# Dataset
_C.DATASETS = CN()
_C.DATASETS.NAMES = ('market1501') # dataset names
_C.DATASETS.ROOT_DIR = ('../data') # Root directory where datasets should be used (and downloaded if not found)
_C.DATASETS.SPLIT = 0
_C.DATASETS.SEQ_SRD = 4


# DataLoader
_C.DATALOADER = CN()
_C.DATALOADER.NUM_WORKERS = 8 # Number of data loading threads
_C.DATALOADER.SAMPLER = 'softmax' # Sampler for data loading
_C.DATALOADER.NUM_INSTANCE = 4 # Number of instance for one batch


# Solver
_C.SOLVER = CN()
_C.SOLVER.SEED = 1234
_C.SOLVER.MARGIN = 0.3

_C.SOLVER.STAGE1 = CN()
_C.SOLVER.STAGE1.IMS_PER_BATCH = 64
_C.SOLVER.STAGE1.OPTIMIZER_NAME = "Adam"
_C.SOLVER.STAGE1.MAX_EPOCHS = 100 # Number of max epoches
_C.SOLVER.STAGE1.BASE_LR = 3e-4 # Base learning rate
_C.SOLVER.STAGE1.LARGE_FC_LR = False
_C.SOLVER.STAGE1.BIAS_LR_FACTOR = 1
_C.SOLVER.STAGE1.MOMENTUM = 0.9 # Momentum
_C.SOLVER.STAGE1.WEIGHT_DECAY = 0.0005
_C.SOLVER.STAGE1.WEIGHT_DECAY_BIAS = 0.0005
_C.SOLVER.STAGE1.WARMUP_FACTOR = 0.01
_C.SOLVER.STAGE1.WARMUP_EPOCHS = 5
_C.SOLVER.STAGE1.WARMUP_LR_INIT = 0.01
_C.SOLVER.STAGE1.LR_MIN = 0.000016
_C.SOLVER.STAGE1.WARMUP_ITERS = 500
_C.SOLVER.STAGE1.WARMUP_METHOD = "linear"
_C.SOLVER.STAGE1.COSINE_MARGIN = 0.5
_C.SOLVER.STAGE1.COSINE_SCALE = 30
_C.SOLVER.STAGE1.CHECKPOINT_PERIOD = 10 # epoch number of saving checkpoints
_C.SOLVER.STAGE1.LOG_PERIOD = 100 # iteration of display training log
_C.SOLVER.STAGE1.EVAL_PERIOD = 10

_C.SOLVER.STAGE2 = CN()
_C.SOLVER.STAGE2.IMS_PER_BATCH = 64
_C.SOLVER.STAGE2.OPTIMIZER_NAME = "Adam"
_C.SOLVER.STAGE2.MAX_EPOCHS = 100 # Number of max epoches
_C.SOLVER.STAGE2.BASE_LR = 3e-4 # Base learning rate
_C.SOLVER.STAGE2.LARGE_FC_LR = False
_C.SOLVER.STAGE2.LARGE_Prompt_LR = False
_C.SOLVER.STAGE2.BIAS_LR_FACTOR = 1
_C.SOLVER.STAGE2.MOMENTUM = 0.9 # Momentum
_C.SOLVER.STAGE2.CENTER_LR = 0.5
_C.SOLVER.STAGE2.CENTER_LOSS_WEIGHT = 0.0005
_C.SOLVER.STAGE2.WEIGHT_DECAY = 0.0005
_C.SOLVER.STAGE2.WEIGHT_DECAY_BIAS = 0.0005
_C.SOLVER.STAGE2.GAMMA = 0.1
_C.SOLVER.STAGE2.STEPS = (40, 70)
_C.SOLVER.STAGE2.WARMUP_FACTOR = 0.01
_C.SOLVER.STAGE2.WARMUP_EPOCHS = 5
_C.SOLVER.STAGE2.WARMUP_LR_INIT = 0.01
_C.SOLVER.STAGE2.LR_MIN = 0.000016
_C.SOLVER.STAGE2.WARMUP_ITERS = 500
_C.SOLVER.STAGE2.WARMUP_METHOD = "linear"
_C.SOLVER.STAGE2.COSINE_MARGIN = 0.5
_C.SOLVER.STAGE2.COSINE_SCALE = 30
_C.SOLVER.STAGE2.CHECKPOINT_PERIOD = 10
_C.SOLVER.STAGE2.LOG_PERIOD = 100
_C.SOLVER.STAGE2.EVAL_PERIOD = 10


# TEST
_C.TEST = CN()
_C.TEST.IMS_PER_BATCH = 128 # Number of images per batch during test
_C.TEST.RE_RANKING = False # If test with re-ranking, options: 'True','False'# If test with re-ranking, options: 'True','False'
_C.TEST.WEIGHT = "" # Path to trained model
_C.TEST.NECK_FEAT = 'after' # Which feature of BNNeck to be used for test, before or after BNNneck, options: 'before' or 'after'
_C.TEST.FEAT_NORM = 'yes' # Whether feature is nomalized before test, if yes, it is equivalent to cosine distance
_C.TEST.DIST_MAT = "dist_mat.npy" # Name for saving the distmat after testing.
_C.TEST.EVAL = False # Whether calculate the eval score option: 'True', 'False'


# OUTPUT_DIR
_C.OUTPUT_DIR = ""
