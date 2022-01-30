import tensorflow as tf
import numpy as np
import cv2
import os
import sys
import tqdm
import lmdb
import six
from PIL import Image

from data_provider.data_utils import get_vocabulary
from utils.transcription_utils import idx2label, calc_metrics
from model import Model
from utils.visualization import heatmap_visualize

from config import get_args

def get_images(images_dir):
    files = []
    exts = ['jpg', 'png', 'jpeg', 'JPG']
    for parent, dirnames, filenames in os.walk(images_dir):
        for filename in filenames:
            for ext in exts:
                if filename.endswith(ext):
                    files.append(os.path.join(parent, filename))
                    break
    print('Find {} images'.format(len(files)))
    return files

def resize_pad_img(image, height, width, keep_ratio=True):
    H, W, C = image.shape
    # Rotate the vertical images
    if H > 4 * W:
        image = np.rot90(image)
        H, W = W, H
    if keep_ratio:
        new_width = int((1.0 * height / H) * W)
        new_width = new_width if new_width < width else width
        new_height = height
        img_resize = np.zeros((height, width, C), dtype=np.uint8)
        image = cv2.resize(image, (new_width, new_height))
        img_resize[:, :new_width, :] = image
    else:
        img_resize = cv2.resize(image, (width, height))
        new_width = width
    img_raw = img_resize
    img_resize = np.expand_dims(cv2.cvtColor(img_resize, cv2.COLOR_RGB2GRAY), axis=-1)
    return img_resize, new_width, img_raw

def data_preprocess(image, word, char2id, args):
    img_resize, new_width, img_raw = resize_pad_img(image, args.height, args.width, args.keep_ratio)
    label = np.full((args.max_len), char2id['EOS'], dtype=np.int)
    label_list = []
    for char in word:
        if char in char2id:
            label_list.append(char2id[char])
        else:
            continue
    if len(label_list) > (args.max_len - 1):
        label_list = label_list[:(args.max_len - 1)]
    label[:len(label_list)] = np.array(label_list)

    return img_resize, label, new_width, img_raw

def main_test_lmdb(args):
    voc, char2id, id2char = get_vocabulary(voc_type=args.voc_type)

    input_images = tf.placeholder(dtype=tf.float32, shape=[1, args.height, args.width, 1], name="input_images")
    input_labels = tf.placeholder(dtype=tf.int32, shape=[1, args.max_len], name="input_labels")
    model = Model(num_classes=len(voc), num_block=args.num_block, embed_dim=args.embed_dim, att_dim=args.att_dim, num_head=args.num_head, num_decoder=args.num_decoder, hidden_units=args.hidden_units, seq_len=args.max_len, is_training=False)

    enc_logits, at_logits,  enc_pred, at_pred, enc_attention_weights, at_attention_weights, _, _ = model(input_images, input_labels, reuse=False)
    global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False, dtype=tf.int32)
    variable_averages = tf.train.ExponentialMovingAverage(0.997, global_step)
    saver = tf.train.Saver(variable_averages.variables_to_restore())
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        ckpt_state = tf.train.get_checkpoint_state(args.checkpoints)
        model_path = os.path.join(args.checkpoints, os.path.basename(ckpt_state.model_checkpoint_path))
        print('Restore from {}'.format(model_path))
        saver.restore(sess, model_path)
        print("Checkpoints step: {}".format(global_step.eval(session=sess)))

        env = lmdb.open(args.test_data_dir, readonly=True)
        txn = env.begin()
        num_samples = int(txn.get(b"num-samples").decode())
        predicts = []
        labels = []

        for i in tqdm.tqdm(range(1, num_samples+1)):
            image_key = b'image-%09d' % i
            label_key = b'label-%09d' % i

            imgbuf = txn.get(image_key)
            buf = six.BytesIO()
            buf.write(imgbuf)
            buf.seek(0)

            img_pil = Image.open(buf).convert('RGB')
            img = np.array(img_pil)
            label = txn.get(label_key).decode()
            labels.append(label)
            img, la, width, img_ = data_preprocess(img, label, char2id, args)

            pred_value, attention_weights_value = sess.run([enc_pred, enc_attention_weights], feed_dict={input_images: [img], input_labels: [la]})

            pred_value_str = idx2label(pred_value, id2char, char2id)[0]

            predicts.append(pred_value_str)
            if args.vis_dir != None and args.vis_dir != "":
                os.makedirs(args.vis_dir, exist_ok=True)
                os.makedirs(os.path.join(args.vis_dir, "errors"), exist_ok=True)
                _ = heatmap_visualize(img_, attention_weights_value, pred_value_str, args.vis_dir, "{}.jpg".format(i))
                if pred_value_str.lower() != label.lower():
                    _ = heatmap_visualize(img_, attention_weights_value, pred_value_str,
                                       os.path.join(args.vis_dir, "errors"), "{}.jpg".format(i))
        metrics_value = calc_metrics(predicts, labels, args.metrics_type)
        print("Done, {}: {}".format(args.metrics_type, metrics_value))

if __name__ == "__main__":
    args = get_args(sys.argv[1:])
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    main_test_lmdb(args)