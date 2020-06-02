from detectron2.config.defaults import _C
from detectron2.config import CfgNode as CN


# ---------------------------------------------------------------------------- #
# Additional Configs
# ---------------------------------------------------------------------------- #
_C.MODEL.MOBILENET = False
_C.MODEL.RESNEST =False
_C.MODEL.BACKBONE.ANTI_ALIAS = False
_C.MODEL.RESNETS.DEFORM_INTERVAL = 1
_C.INPUT.HFLIP_TRAIN = True
_C.INPUT.CROP.CROP_INSTANCE = True

# ---------------------------------------------------------------------------- #
# FCOS Head
# ---------------------------------------------------------------------------- #
_C.MODEL.FCOS = CN()

# This is the number of foreground classes.
_C.MODEL.FCOS.NUM_CLASSES = 80
_C.MODEL.FCOS.IN_FEATURES = ["p3", "p4", "p5", "p6", "p7"]
_C.MODEL.FCOS.FPN_STRIDES = [8, 16, 32, 64, 128]
_C.MODEL.FCOS.PRIOR_PROB = 0.01
_C.MODEL.FCOS.INFERENCE_TH_TRAIN = 0.6
_C.MODEL.FCOS.INFERENCE_TH_TEST = 0.6
_C.MODEL.FCOS.NMS_TH = 0.6
_C.MODEL.FCOS.PRE_NMS_TOPK_TRAIN = 1000
_C.MODEL.FCOS.PRE_NMS_TOPK_TEST = 1000
_C.MODEL.FCOS.POST_NMS_TOPK_TRAIN = 100
_C.MODEL.FCOS.POST_NMS_TOPK_TEST = 100
_C.MODEL.FCOS.TOP_LEVELS = 2
_C.MODEL.FCOS.NORM = "GN"  # Support GN or none
_C.MODEL.FCOS.USE_SCALE = True

# Multiply centerness before threshold
# This will affect the final performance by about 0.05 AP but save some time
_C.MODEL.FCOS.THRESH_WITH_CTR = False

# Focal loss parameters
_C.MODEL.FCOS.LOSS_ALPHA = 0.25
_C.MODEL.FCOS.LOSS_GAMMA = 2.0
_C.MODEL.FCOS.SIZES_OF_INTEREST = [64, 128, 256, 512]
_C.MODEL.FCOS.USE_RELU = True
_C.MODEL.FCOS.USE_DEFORMABLE = False

# the number of convolutions used in the cls and bbox tower
_C.MODEL.FCOS.NUM_CLS_CONVS = 4
_C.MODEL.FCOS.NUM_BOX_CONVS = 4
_C.MODEL.FCOS.NUM_SHARE_CONVS = 0
_C.MODEL.FCOS.CENTER_SAMPLE = True
_C.MODEL.FCOS.POS_RADIUS = 1.5
_C.MODEL.FCOS.LOC_LOSS_TYPE = 'giou'
_C.MODEL.FCOS.YIELD_PROPOSAL = False


# ---------------------------------------------------------------------------- #
# ATSS Options
# ---------------------------------------------------------------------------- #
_C.MODEL.ATSS = CN()
_C.MODEL.ATSS.NUM_CLASSES = 80  
_C.MODEL.ATSS.IN_FEATURES = ["p3", "p4", "p5", "p6", "p7"]
# Anchor parameter
_C.MODEL.ANCHOR_GENERATOR.ANCHOR_SIZES = (64, 128, 256, 512, 1024)
_C.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = (1.0,)
_C.MODEL.ANCHOR_GENERATOR.ANCHOR_STRIDES = (8, 16, 32, 64, 128)
_C.MODEL.ATSS.ANCHOR_STRIDES = (8, 16, 32, 64, 128)
_C.MODEL.ANCHOR_GENERATOR.STRADDLE_THRESH = 0
_C.MODEL.ANCHOR_GENERATOR.OCTAVE = 2.0
_C.MODEL.ANCHOR_GENERATOR.SCALES_PER_OCTAVE = 1

# Head parameter
_C.MODEL.ATSS.NUM_CONVS = 4
_C.MODEL.ATSS.USE_DCN_IN_TOWER = False

# Focal loss parameter
_C.MODEL.ATSS.LOSS_ALPHA = 0.25
_C.MODEL.ATSS.LOSS_GAMMA = 2.0

# how to select positves: ATSS (Ours) , SSC (FCOS), IoU (RetinaNet), TOPK
_C.MODEL.ATSS.POSITIVE_TYPE = 'ATSS'

# IoU parameter to select positves
_C.MODEL.ATSS.FG_IOU_THRESHOLD = 0.5
_C.MODEL.ATSS.BG_IOU_THRESHOLD = 0.4

# topk for selecting candidate positive samples from each level
_C.MODEL.ATSS.TOPK = 9

# regressing from a box ('BOX') or a point ('POINT')
_C.MODEL.ATSS.REGRESSION_TYPE = 'BOX'

# Weight for bbox_regression loss
_C.MODEL.ATSS.REG_LOSS_WEIGHT = 2.0

# Inference parameter
_C.MODEL.ATSS.PRIOR_PROB = 0.01
_C.MODEL.ATSS.INFERENCE_TH = 0.05
_C.MODEL.ATSS.NMS_TH = 0.6
_C.MODEL.ATSS.PRE_NMS_TOP_N = 1000

_C.MODEL.ATSS.PRIOR_PROB = 0.01
_C.MODEL.ATSS.INFERENCE_TH_TRAIN = 0.6
_C.MODEL.ATSS.INFERENCE_TH_TEST = 0.6
_C.MODEL.ATSS.NMS_TH = 0.6
_C.MODEL.ATSS.PRE_NMS_TOPK_TRAIN = 1000
_C.MODEL.ATSS.PRE_NMS_TOPK_TEST = 1000
_C.MODEL.ATSS.POST_NMS_TOPK_TRAIN = 100
_C.MODEL.ATSS.POST_NMS_TOPK_TEST = 100
_C.MODEL.ATSS.TOP_LEVELS = 2
_C.MODEL.ATSS.NORM = "GN"  # Support GN or none
_C.MODEL.ATSS.USE_SCALE = True

_C.MODEL.ATSS.THRESH_WITH_CTR = False

# Focal loss parameters
_C.MODEL.ATSS.LOSS_ALPHA = 0.25
_C.MODEL.ATSS.LOSS_GAMMA = 2.0

_C.MODEL.ATSS.USE_RELU = True
_C.MODEL.ATSS.USE_DEFORMABLE = False

# the number of convolutions used in the cls and bbox tower
_C.MODEL.ATSS.NUM_CLS_CONVS = 4
_C.MODEL.ATSS.NUM_BOX_CONVS = 4
_C.MODEL.ATSS.NUM_SHARE_CONVS = 0

# ---------------------------------------------------------------------------- #
# VoVNet backbone
# ---------------------------------------------------------------------------- #
_C.MODEL.VOVNET = CN()
_C.MODEL.VOVNET.CONV_BODY = "V-39-eSE"
_C.MODEL.VOVNET.OUT_FEATURES = ["stage2", "stage3", "stage4", "stage5"]

# Options: FrozenBN, GN, "SyncBN", "BN"
_C.MODEL.VOVNET.NORM = "FrozenBN"
_C.MODEL.VOVNET.OUT_CHANNELS = 256
_C.MODEL.VOVNET.BACKBONE_OUT_CHANNELS = 256


# ---------------------------------------------------------------------------- #
# ResNe[X]t options (ResNets = {ResNet, ResNeXt}
# Note that parts of a resnet may be used for both the backbone and the head
# These options apply to both
# ---------------------------------------------------------------------------- #
_C.MODEL.RESNETS = CN()

_C.MODEL.RESNETS.DEPTH = 50
_C.MODEL.RESNETS.OUT_FEATURES = ["res4"]  # res4 for C4 backbone, res2..5 for FPN backbone

# Number of groups to use; 1 ==> ResNet; > 1 ==> ResNeXt
_C.MODEL.RESNETS.NUM_GROUPS = 1

# Options: FrozenBN, GN, "SyncBN", "BN"
_C.MODEL.RESNETS.NORM = "FrozenBN"

# Baseline width of each group.
# Scaling this parameters will scale the width of all bottleneck layers.
_C.MODEL.RESNETS.WIDTH_PER_GROUP = 64

# Place the stride 2 conv on the 1x1 filter
# Use True only for the original MSRA ResNet; use False for C2 and Torch models
_C.MODEL.RESNETS.STRIDE_IN_1X1 = True

# Apply dilation in stage "res5"
_C.MODEL.RESNETS.RES5_DILATION = 1

# Output width of res2. Scaling this parameters will scale the width of all 1x1 convs in ResNet
# For R18 and R34, this needs to be set to 64
_C.MODEL.RESNETS.RES2_OUT_CHANNELS = 256
_C.MODEL.RESNETS.STEM_OUT_CHANNELS = 64

# Apply Deformable Convolution in stages
# Specify if apply deform_conv on Res2, Res3, Res4, Res5
_C.MODEL.RESNETS.DEFORM_ON_PER_STAGE = [False, False, False, False]
# Use True to use modulated deform_conv (DeformableV2, https://arxiv.org/abs/1811.11168);
# Use False for DeformableV1.
_C.MODEL.RESNETS.DEFORM_MODULATED = False
# Number of groups in deformable conv.
_C.MODEL.RESNETS.DEFORM_NUM_GROUPS = 1


# Apply deep stem 
_C.MODEL.RESNETS.DEEP_STEM = False
# Apply avg after conv2 in the BottleBlock
# When AVD=True, the STRIDE_IN_1X1 should be False
_C.MODEL.RESNETS.AVD = False
# Apply avg_down to the downsampling layer for residual path 
_C.MODEL.RESNETS.AVG_DOWN = False

# Radix in ResNeSt
_C.MODEL.RESNETS.RADIX = 1
# Bottleneck_width in ResNeSt
_C.MODEL.RESNETS.BOTTLENECK_WIDTH = 64


# ---------------------------------------------------------------------------- #
# DLA backbone
# ---------------------------------------------------------------------------- #

_C.MODEL.DLA = CN()
_C.MODEL.DLA.CONV_BODY = "DLA34"
_C.MODEL.DLA.OUT_FEATURES = ["stage2", "stage3", "stage4", "stage5"]

# Options: FrozenBN, GN, "SyncBN", "BN"
_C.MODEL.DLA.NORM = "FrozenBN"


# ---------------------------------------------------------------------------- #
# Basis Module Options
# ---------------------------------------------------------------------------- #
_C.MODEL.BASIS_MODULE = CN()
_C.MODEL.BASIS_MODULE.NAME = "ProtoNet"
_C.MODEL.BASIS_MODULE.NUM_BASES = 4
_C.MODEL.BASIS_MODULE.LOSS_ON = False
_C.MODEL.BASIS_MODULE.ANN_SET = "coco"
_C.MODEL.BASIS_MODULE.CONVS_DIM = 128
_C.MODEL.BASIS_MODULE.IN_FEATURES = ["p3", "p4", "p5"]
_C.MODEL.BASIS_MODULE.NORM = "SyncBN"
_C.MODEL.BASIS_MODULE.NUM_CONVS = 3
_C.MODEL.BASIS_MODULE.COMMON_STRIDE = 8
_C.MODEL.BASIS_MODULE.NUM_CLASSES = 80
_C.MODEL.BASIS_MODULE.LOSS_WEIGHT = 0.3
