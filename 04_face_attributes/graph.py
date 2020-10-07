import tensorflow as tf
from tensorflow.contrib.layers import *
from tensorflow.contrib.slim.python.slim.nets.inception_v3 import inception_v3_base
slim = tf.contrib.slim

def inception_v3(images, drop_out=0.5, is_training=True):
    batch_norm_params = {
        "is_training": is_training,
        "trainable": True,
        "decay": 0.9997,
        "epsilon": 0.00001,
        "variables_collections": {
            "beta":None,
            "gamma": None,
            "moving_mean": ["moving_vars"],
            "moving_variance": ["moving_var"]
        }
    }

    weights_regularizer = tf.contrib.layers.l2_regularizer(0.00004)
    with tf.contrib.slim.arg_scope(
        [tf.contrib.slim.conv2d, tf.contrib.slim.fully_connected],
        weights_regularizer = weights_regularizer,
        trainable = True):

        with tf.contrib.slim.arg_scope(
            [tf.contrib.slim.conv2d],
            weights_initializer = tf.truncated_normal_initializer(stddev=0.1),
            activation_fn = tf.nn.relu,
            normalizer_fn = batch_norm,
            normalizer_params = batch_norm_params):
            nets, endpoints = inception_v3_base(images)
            print(nets)
            print(endpoints)
            
            net = tf.reduce_mean(nets, axis = [1,2])
            net = tf.nn.dropout(net, drop_out, name="droplast")
            net = flatten(net, scope="flatten")
    
    net_eyeglasses = slim.fully_connected(net, 2, activation_fn=None)
    net_young = slim.fully_connected(net, 2, activation_fn=None)
    net_male = slim.fully_connected(net, 2, activation_fn=None)
    net_smiling = slim.fully_connected(net, 2, activation_fn=None)

    return net_eyeglasses, net_young, net_male, net_smiling

input_x = tf.placeholder(tf.float32, shape=[None, 128, 128, 3])
label_eyeglasses = tf.placeholder(tf.int64, shape=[None, 1]) 
label_young = tf.placeholder(tf.int64, shape=[None, 1]) 
label_male = tf.placeholder(tf.int64, shape=[None, 1]) 
label_smiling = tf.placeholder(tf.int64, shape=[None, 1]) 

logits_eyeglasses, logits_young, logits_male, logits_smiling = inception_v3(input_x, 1.0, False)

loss_eyeglasses = tf.losses.sparse_softmax_cross_entropy(labels = label_eyeglasses,logits=logits_eyeglasses)
loss_young = tf.losses.sparse_softmax_cross_entropy(labels = label_young,logits=logits_young)
loss_male = tf.losses.sparse_softmax_cross_entropy(labels = label_male,logits=logits_male)
loss_smiling = tf.losses.sparse_softmax_cross_entropy(labels = label_smiling,logits=logits_smiling)

logits_eyeglasses = tf.nn.softmax(logits_eyeglasses)
logits_young = tf.nn.softmax(logits_young)
logits_male = tf.nn.softmax(logits_male)
logits_smiling = tf.nn.softmax(logits_smiling)

logits_eyeglasses = tf.argmax(logits_eyeglasses, axis=1)
logits_young = tf.argmax(logits_young, axis=1)
logits_male = tf.argmax(logits_male, axis=1)
logits_smiling = tf.argmax(logits_smiling, axis=1)

print(logits_eyeglasses, logits_young, logits_male, logits_smiling)

loss = loss_eyeglasses + loss_young + loss_male + loss_smiling

reg_set = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
l2_loss = tf.add_n(reg_set)

# learn
global_step = tf.Variable(0, trainable=True)
lr = tf.train.exponential_decay(0.0001, global_step, 
                                decay_steps = 1000, 
                                decay_rate=0.98)
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    tf.train.AdamOptimizer(lr).minimize(loss + l2_loss, global_step)

def get_one_batch(batch_size, type):
    if type == 0:
        file_list = tf.gfile.Glob('train.tfrecords')
    else:
        file_list = tf.gfile.Glob('test.tfrecords')
    
    reader = tf.TFRecordReader()

    file_queue = tf.train.string_input_producer(
        file_list, num_epochs=None, shuffle=True
    )

    _, se = reader.read(file_queue)

    if type == 0:
        batch = tf.train.shuffle_batch([se], batch_size, 
                                        capacity=batch_size, 
                                        min_after_dequeue=batch_size // 2)
    else:
        batch = tf.train.batch([se], batch_size, capacity)
    
    features = tf.parse_example(batch, features={
        "image": tf.FixedLenFeature([], tf.string),
        "Eyeglasses": tf.FixedLenFeature([1], tf.int64),
        "Male": tf.FixedLenFeature([1], tf.int64),
        "Young": tf.FixedLenFeature([1], tf.int64),
        "Smiling": tf.FixedLenFeature([1], tf.int64)
    })

    batch_im = features["image"]
    batch_eye = (features["Eyeglasses"]+1) // 2
    batch_male = (features["Male"]+1) // 2
    batch_young = (features["Young"]+1) // 2
    batch_smiling = (features["Smiling"]+1) // 2
    batch_im = tf.decode_raw(batch_im, tf.uint8)

    batch_im = tf.cast(tf.reshape(batch_im, (batch_size, 128, 128, 3)), tf.float32)

    return batch_im, batch_eye, batch_male, batch_young, batch_smiling

tf_im_batch, tf.label_eye_batch, tf.label_male_batch, \
        tf.label_young_batch, tf.label_smiling_batch = get_one_batch(64, 0)

te_im_batch, te.label_eye_batch, te.label_male_batch, \
        te.label_young_batch, te.label_smiling_batch = get_one_batch(64, 1)
saver = tf.train.Saver(tf.global_variables())

### session
with tf.Session() as session:
    coord = tf.train.Coordinator()
    tf.train.start_queue_runners(sess = session, coord=coord)

    init_ops = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    session.run(init_ops)

    summary_writer = tf.summary.FileWriter('logs', session.graph)

    ### select the lastest model checkpoint
    ckpt = tf.train.get_checkpoint_state("models")
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        print(ckpt.model_checkpoint_path)
        saver.restore(session, ckpt.model_checkpoint_path)
    
    output_graph_def = tf.graph_util.convert_variables_to_constants(session, 
                                                                    session.graph.as_graph_def(), 
                                                                    ['ArgMax', 'ArgMax_1', 'ArgMax_2', 'ArgMax_3'])
    with tf.gfile.FastGFile('face_attribute.pb', 'wb') as f:
        f.write(output_graph_def.SerializeToString())
        f.close()

    









