# Face Landmarks

## Main Framework
- SENet

## Dataset
- 300W-LP 
- Link: <http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/main.htm>

## Development procedures
1. Data reading: read_300wlp_data.py
2. Save as TFRecord files: write_data_tf.py
3. Model train: landmark_train.py
4. Model inference: inference_senet.py

## Freezed graph
- Link1: <https://drive.google.com/file/d/1hFugPu3MkNTfKLmC6CodY_YH_cuA2Xuh/view?usp=sharing>
- Link2: 68 landmarkd with dlib [<https://drive.google.com/file/d/1-08C9hEY-ZNWyoraMBlwofyE9dQ1KA9x/view?usp=sharing>]