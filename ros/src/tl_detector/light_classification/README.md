# Traffic light detection 

## Result 
[notebook](./tl_detection.ipynb)

## frozen models
1. [Download from google drive](https://drive.google.com/open?id=1-d8KBPTXgwxt2Dbsl2Uk89KuW_qR11dA)
1. Extracts to `frozen_models/`

## Commands for training and exporting for inference
For copy and paste. :)

Needs dataset of TFRecord format and label map of pbtxt type.

#### faster_rcnn_resnet101_coco_2018_01_28
```
python object_detection/train.py --pipeline_config_path config/faster_rcnn_resnet101_coco.config --train_dir train_dir/faster_rcnn_resnet101_coco_2018_01_28
```

```
python object_detection/export_inference_graph.py --pipeline_config_path config/faster_rcnn_resnet101_coco.config --trained_checkpoint_prefix train_dir/faster_rcnn_resnet101_coco_2018_01_28/model.ckpt-{#####} --output_directory frozen_models/faster_rcnn_resnet101_coco_2018_01_28
```

#### ssd_inception_v2_coco_2017_11_17
```
python object_detection/train.py --pipeline_config_path config/ssd_inception_v2_coco.config --train_dir train_dir/ssd_inception_v2_coco_2017_11_17
```

```
python object_detection/export_inference_graph.py --pipeline_config_path config/ssd_inception_v2_coco.config --trained_checkpoint_prefix train_dir/ssd_inception_v2_coco_2017_11_17/model.ckpt-{#####} --output_directory frozen_models/ssd_inception_v2_coco_2017_11_17
```

#### ssd_mobilenet_v2_coco_2018_03_29
```
python object_detection/train.py --pipeline_config_path config/ssd_mobilenet_v2_coco.config --train_dir train_dir/ssd_mobilenet_v2_coco_2018_03_29
```

```
python object_detection/export_inference_graph.py --pipeline_config_path config/ssd_mobilenet_v2_coco.config --trained_checkpoint_prefix train_dir/ssd_mobilenet_v2_coco_2018_03_29/model.ckpt-{#####} --output_directory frozen_models/ssd_mobilenet_v2_coco_2018_03_29
```