# Face Recognition

## Main Framework
- facenet
- Link: <https://github.com/davidsandberg/facenet>

## Dataset
- LFW: <http://vis-www.cs.umass.edu/lfw/>
- Celeba: <http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html>
- 64_CASIA-FaceV5: <https://pan.baidu.com/s/1WS4nooNQgmQHR6EpmrW6dw  password: sc8b>
- CASIA-WebFace: <https://pan.baidu.com/s/162V9XzC_m66iNHLg44twmQ password: n5ru>

## Development procedures
1. Data preparation: facenet/src/lfw.py + facenet/dlib_facedetect.py
2. Model train: facenet/src/train_tripletloss.py
3. Model inference: facenet/src/compare.py

## Freezed graph
- Link: <https://drive.google.com/file/d/1ozRjZlhe3GWcGeQ7S8CY5J3FKXIikKsw/view?usp=sharing>