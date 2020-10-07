import tensorflow as tf
import cv2
import glob
import numpy as np
import dlib

pb_path = "/Users/chenxuzhuang/faceAI/faceAttribute/pb/face_attribute.pb"

sess = tf.Session()

with sess.as_default():
    with tf.gfile.FastGFile(pb_path, 'rb') as f:
        graph_def = sess.graph_def
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')

# pred_eyeglasses = sess.graph.get_tensor_by_name('Softmax:0')
# pred_young = sess.graph.get_tensor_by_name('Softmax_1:0')
# pred_male = sess.graph.get_tensor_by_name('Softmax_2:0')
# pred_smiling = sess.graph.get_tensor_by_name('Softmax_3:0')

# pred_eyeglasses = tf.argmax(pred_eyeglasses, axis=1)
# pred_young = tf.argmax(pred_young, axis=1)
# pred_male = tf.argmax(pred_male, axis=1)
# pred_smiling = tf.argmax(pred_smiling, axis=1)


pred_eyeglasses = sess.graph.get_tensor_by_name('ArgMax:0')
pred_young = sess.graph.get_tensor_by_name('ArgMax_1:0')
pred_male = sess.graph.get_tensor_by_name('ArgMax_2:0')
pred_smiling = sess.graph.get_tensor_by_name('ArgMax_3:0')

detector = dlib.get_frontal_face_detector()

im_list = glob.glob("/Users/chenxuzhuang/faceAI/faceAttribute/images/*")
for im_path in im_list:
    im_data = cv2.imread(im_path)
    rects = detector(im_data, 0)

    if len(rects) == 0:
        continue

    x1 = rects[0].left()
    y1 = rects[0].top()
    x2 = rects[0].right()
    y2 = rects[0].bottom()

    y1 = int(max(y1 - 0.3 * (y2 - y1), 0))

    im_data = im_data[y1:y2, x1:x2]

    im_data = cv2.resize(im_data, (128,128))

    [eye, young, male, smiling] = sess.run([pred_eyeglasses, pred_young, pred_male, pred_smiling], 
            {'Placeholder:0': np.expand_dims(im_data, 0)})

    print("eye, young, male, smiling", eye, young, male, smiling)
    cv2.imshow("11", im_data)
    cv2.waitKey(0)



