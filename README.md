# Object-Detection.pytorch

## bdd100k Dataset Baseline
- we use `mmdetection` to train all models.
- All models were trained on `bdd100k_train`, and tested on the `bdd100k_val`.
- We use distributed training across 8 Nvdia-1080Ti GPUs. 





### Anchor-based:
|  Name    | backbone | tricks |  AP   |  AP50  |  AP75  |  APs   |  APm   |  APl   |
| :------: |:------:  |:------:|:------:|:------:|:------:|:------:|:------:|:------:|
|  FasterRCNN | R_50_FPN |   16   | 
|  PISA    | R_50_FPN |      | 
|  LibraRCNN| R_50_FPN|      | 
|  GA      | R_50_FPN |      | 


### Anchor-free
|  Name    | backbone | tricks |  AP   |  AP50  |  AP75  |  APs   |  APm   |  APl   |
| :------: |:------:  |:------:|:------:|:------:|:------:|:------:|:------:|:------:|
| FCOS     | R_50_FPN |        |
| ATSS     | R_50_FPN |      |
| CenterNet| 
| RepPoints| R_50_FPN |     |
