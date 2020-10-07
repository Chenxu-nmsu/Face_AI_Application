#from object_detection.utils import ops as utils_ops
import os, sys
import numpy as np
import cv2
import tensorflow as tf
import glob

PATH_TO_FROZEN_GRAPH = "/Users/chenxuzhuang/faceRecognition/widerface/resnet50v1-fpn/pb/frozen_inference_graph.pb"
PATH_TO_LABELS = "/Users/chenxuzhuang/faceRecognition/widerface/models/research/object_detection/data/face_label_map.pbtxt"


detection_sess = tf.Session()
with detection_sess.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
        
im_path_list = glob.glob('/Users/chenxuzhuang/faceRecognition/widerface/test_image/*')
IMAGE_SIZE = (256, 256)


def run_inference_for_single_image(model, image):
    image = np.asarray(image)
    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis,...]

    # Run inference
    model_fn = model.signatures['serving_default']
    output_dict = model_fn(input_tensor)

    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(output_dict.pop('num_detections'))
    output_dict = {key:value[0, :num_detections].numpy() 
                    for key,value in output_dict.items()}
    output_dict['num_detections'] = num_detections

    ############
    # detection_classes should be ints.
    output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)

    # Handle models with masks:
    if 'detection_masks' in output_dict:
    # Reframe the the bbox mask to the image size.
    detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                output_dict['detection_masks'], output_dict['detection_boxes'],
                image.shape[0], image.shape[1])      
    detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,
                                        tf.uint8)
    output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()

    output_dict = detection_sess.run(tensor_dict,
                                                feed_dict={image_tensor: np.expand_dims(image_np, 0)})

    # all outputs are float32 numpy arrays, so convert types as appropriate
    output_dict['num_detections'] = int(output_dict['num_detections'][0])
    output_dict['detection_classes'] = output_dict[
        'detection_classes'][0].astype(np.uint8)
    output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
    output_dict['detection_scores'] = output_dict['detection_scores'][0]

    return output_dict
        


for image_path in im_path_list:
    imdata = cv2.imread(image_path)
    sp = imdata.shape
    imdata = cv2.resize(imdata, IMAGE_SIZE)
    output_dict = run_inference_for_single_image(imdata, detection_graph)

    for i in range(len(output_dict['detection_scores'])):
        if output_dict['detection_scores'][i] > 0.6:
            bbox = output_dict['detection_boxes'][i]
            cate = output_dict['detection_classes'][i]
            y1 = bbox[0] * sp[0]
            x1 = bbox[1] * sp[1]
            y2 = bbox[2] * sp[0]
            x2 = bbox[3] * sp[1]
            cv2.rectangle(imdata, (x1, y1), (x2, y2), (0,255,0), 2)
    cv2.imshow('im', imdata)
    cv2.waitKey(0)

