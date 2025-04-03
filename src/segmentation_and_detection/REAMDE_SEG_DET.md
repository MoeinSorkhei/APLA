## Segmentation and Detection with APLA
Code for APLA for segmentation and detection can be found in this folder.

### Segmentation on ADE20K with SETR-PUP
We used the [mmsegmentation](https://github.com/open-mmlab/mmsegmentation/tree/main/configs/setr) 
repository and directly added ViT with APLA as a new backbone.
Clone their repository and make these changes:

- ```mmseg/models/backbones```: add [```apla_vit.py```](./segmentation/apla_vit.py)
- Modify ```mmseg/models/backbones/__init__.py``` accordingly to inlucde this backbone


- In ```configs/setr``` add 
[```apla_setr_vit-l_pup_8xb2-160k_ade20k-512x512.py```](./segmentation/apla_setr_vit-l_pup_8xb2-160k_ade20k-512x512.py)



### Object detection and instance segmentation on MS COCO

For this please use the 
[Swin Transformer for Object Detection](https://github.com/SwinTransformer/Swin-Transformer-Object-Detection) 
repository, and make the follwoing changes:

- In ```mmdet/models/backbones```
add [```apla_swin_transformer.py```](./detection/apla_swin_transformer.py)

- Modify ```backbones/__init__.py``` accordingly to inlucde this new backbone

- In ```configs``` 
add [```apla_mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_adamw_1x_coco.py```](./detection/apla_mask_rcnn_swin_tiny_patch4_window7_mstrain_480-800_adamw_1x_coco.py)


