# Object-Detection.pytorch

## bdd100k Dataset Baseline
- we use `mmdetection` to train all models.
- All models were trained on `bdd100k_train`, and tested on the `bdd100k_val`.
- We use distributed training across 8 Nvdia-1080Ti GPUs. 





### Anchor-based:
|  Name    | backbone | tricks |  AP   |  AP50  |  AP75  |  APs   |  APm   |  APl   |
| :------: |:------:  |:------:|:------:|:------:|:------:|:------:|:------:|:------:|
|  FasterRCNN | R_50_FPN |      | 0.318 | 0.551 | 0.311 | 0.145 | 0.356 | 0.497|
|  FasterRCNN | R_101_FPN|    |  0.322 | 0.553 | 0.314 | 0.142 | 0.360 | 0.512 |
|  PISA    | R_50_FPN |      | 
|  LibraRCNN| R_50_FPN|      | 
|  GA      | R_50_FPN |      | 


### Anchor-free
|  Name    | backbone | tricks |  AP   |  AP50  |  AP75  |  APs   |  APm   |  APl   |
| :------: |:------:  |:------:|:------:|:------:|:------:|:------:|:------:|:------:|
| FCOS     | R_50_FPN |        |
| ATSS     | R_50_FPN |      | 0.329 | 0.562 | 0.323 | 0.141 | 0.367 | 0.517| 
| CenterNet| 
| RepPoints| R_50_FPN |     |
