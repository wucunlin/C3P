from easydict import EasyDict as edict

config = edict()

config.OUTPUT_DIR = 'output'
config.LOG_DIR = 'log'
config.DATA_DIR = ''
config.GPUS = '0'
config.WORKERS = 4
config.PRINT_FREQ = 20

# Cudnn related params
config.CUDNN = edict()
config.CUDNN.BENCHMARK = True
config.CUDNN.DETERMINISTIC = False
config.CUDNN.ENABLED = True

# pose_pointnet related params
POSE_POINTNET = edict()
POSE_POINTNET.STACK_NUM = 1
POSE_POINTNET.SA_MLPS = [[64, 64, 128], [128, 128, 256], [256, 512, 1024]]
POSE_POINTNET.FP_MLPS = [[256, 256], [256, 128], [128, 128, 128]]
POSE_POINTNET.REG_MLP = [1024, 512, 128]
POSE_POINTNET.NETG_MLP = [128, 128]
POSE_POINTNET.LEVEL_INPUT_NUM = [2048, 512, 128]


MODEL_EXTRAS = {
    'pose_pointnet': POSE_POINTNET
}

# common params for NETWORK
config.MODEL = edict()
config.MODEL.INIT_WEIGHTS = True
config.MODEL.NUM_JOINTS = 17


config.MODEL.POINTNET = edict()
config.MODEL.POINTNET.INPUT_NUM = 2048
config.MODEL.POINTNET.EXTRA = MODEL_EXTRAS['pose_pointnet']

config.LOSS = edict()
config.LOSS.USE_TARGET_WEIGHT = True

# DATASET related params
config.DATASET = edict()
config.DATASET.ROOT = 'data/CMU_Panoptic'

# train
config.TRAIN = edict()

config.TRAIN.LR_FACTOR = 0.1
config.TRAIN.LR_STEP = [90, 110]
config.TRAIN.LR = 0.0001

config.TRAIN.OPTIMIZER = 'adam'
config.TRAIN.MOMENTUM = 0.9
config.TRAIN.WD = 0.001
config.TRAIN.NESTEROV = False
config.TRAIN.GAMMA1 = 0.99
config.TRAIN.GAMMA2 = 0.0

config.TRAIN.BEGIN_EPOCH = 0
config.TRAIN.END_EPOCH = 140

config.TRAIN.RESUME = False
config.TRAIN.CHECKPOINT = ''

config.TRAIN.BATCH_SIZE = 2
config.TRAIN.SHUFFLE = True

# point cloud hyperparameters
config.POINT_CLOUD = edict()

config.POINT_CLOUD.GROUP_NUM = [512, 128]
config.POINT_CLOUD.KNN = 64
config.POINT_CLOUD.BALL_RADIUS = [0.015, 0.04]
config.POINT_CLOUD.INSERT_K = 3

# testing
config.TEST = edict()

# size of images for each device
config.TEST.BATCH_SIZE = 2
# Test Model Epoch
config.TEST.FLIP_TEST = False
config.TEST.POST_PROCESS = True
config.TEST.SHIFT_HEATMAP = True
config.TEST.USE_GT_BBOX = False
# nms
config.TEST.OKS_THRE = 0.5
config.TEST.IN_VIS_THRE = 0.0
config.TEST.COCO_BBOX_FILE = ''
config.TEST.BBOX_THRE = 1.0
config.TEST.MODEL_FILE = ''
config.TEST.IMAGE_THRE = 0.0
config.TEST.NMS_THRE = 1.0

# debug
config.DEBUG = edict()
config.DEBUG.DEBUG = True
config.DEBUG.SAVE_BATCH_IMAGES_GT = False
config.DEBUG.SAVE_BATCH_IMAGES_PRED = True
config.DEBUG.SAVE_HEATMAPS_GT = False
config.DEBUG.SAVE_HEATMAPS_PRED = False

