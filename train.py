import sys
import os
import time
import tensorflow as tf
from tensorflow.contrib import slim
import numpy as np

from model import Model
from data_provider import lmdb_data_generator
from data_provider import evaluator_data
from data_provider.data_utils import get_vocabulary
from utils.transcription_utils import idx2label, calc_metrics
from config import get_args

def get_data(image_dir, gt_path, voc_type, max_len, num_samples, height, width, batch_size, workers, keep_ratio, with_aug):
    data_list = []
    if isinstance(image_dir, list) and len(image_dir) > 1:
        # assert len(image_dir) == len(gt_path), "datasets and gt are not corresponding"
        assert batch_size % len(image_dir) == 0, "batch size should divide dataset num"
        per_batch_size = batch_size // len(image_dir)

        for i in image_dir:
            data_list.append(lmdb_data_generator.get_batch(workers, lmdb_dir=i, input_height=height, input_width=width, batch_size=per_batch_size, max_len=max_len, voc_type=voc_type, keep_ratio=keep_ratio, with_aug=with_aug))

    else:
        if isinstance(image_dir, list):
            data = lmdb_data_generator.get_batch(workers, lmdb_dir=image_dir[0], input_height=height, input_width=width, batch_size=batch_size, max_len=max_len, voc_type=voc_type, keep_ratio=keep_ratio, with_aug=with_aug)

        else:
            data = lmdb_data_generator.get_batch(workers, lmdb_dir=image_dir, input_height=height, input_width=width, batch_size=batch_size, max_len=max_len, voc_type=voc_type, keep_ratio=keep_ratio, with_aug=with_aug)
        data_list.append(data)

    return data_list

def get_batch_data(data_list, batch_size):
    batch_images = []
    batch_labels = []
    batch_labels_mask = []
    batch_labels_str = []
    # batch_widths = []
    for data in data_list:
        _data = next(data)
        batch_images.append(_data[0])
        batch_labels.append(_data[1])
        batch_labels_mask.append(_data[2])
        batch_labels_str.extend(_data[4])
        # batch_widths.append(_data[5])

    batch_images = np.concatenate(batch_images, axis=0)
    batch_labels = np.concatenate(batch_labels, axis=0)
    batch_labels_mask = np.concatenate(batch_labels_mask, axis=0)
    # batch_widths = np.concatenate(batch_widths, axis=0)

    assert len(batch_images) == batch_size, "concat data is not equal to batch size"

    return batch_images, batch_labels, batch_labels_mask, batch_labels_str

def main_train(args):
    voc, char2id, id2char = get_vocabulary(voc_type=args.voc_type)
    tf.set_random_seed(1)
    # Build graph
    input_train_images = tf.placeholder(dtype=tf.float32, shape=[args.train_batch_size, args.height, args.width, 1], name="input_train_images")
    input_train_labels = tf.placeholder(dtype=tf.int32, shape=[args.train_batch_size, args.max_len], name="input_train_labels")
    input_train_labels_mask = tf.placeholder(dtype=tf.int32, shape=[args.train_batch_size, args.max_len], name="input_train_labels_mask")

    input_val_images = tf.placeholder(dtype=tf.float32, shape=[args.val_batch_size, args.height, args.width, 1],name="input_val_images")
    input_val_labels = tf.placeholder(dtype=tf.int32, shape=[args.val_batch_size, args.max_len], name="input_val_labels")
    input_val_labels_mask = tf.placeholder(dtype=tf.int32, shape=[args.val_batch_size, args.max_len], name="input_val_labels_mask")

    train_model = Model(num_classes=len(voc), num_block=args.num_block, embed_dim=args.embed_dim, att_dim=args.att_dim, num_head=args.num_head, hidden_units=args.hidden_units, num_decoder=args.num_decoder, seq_len=args.max_len, is_training=True)
    eval_model  = Model(num_classes=len(voc), num_block=args.num_block, embed_dim=args.embed_dim, att_dim=args.att_dim, num_head=args.num_head, hidden_units=args.hidden_units, num_decoder=args.num_decoder, seq_len=args.max_len, is_training=False)

    train_enc_logits, train_at_logits, train_enc_pred, train_at_pred, train_enc_alphas, train_at_alphas, train_nat_glimpses, train_at_glimpses = train_model(input_train_images, input_train_labels, input_train_labels_mask, reuse=False)

    train_enc_loss = train_model.loss(train_enc_logits, input_train_labels, input_train_labels_mask, None)
    train_at_loss = train_model.loss(train_at_logits, input_train_labels, input_train_labels_mask, None)
    train_glimpse_loss = train_model.glimpse_mimic_loss(train_nat_glimpses, train_at_glimpses, input_train_labels_mask)

    train_loss = train_enc_loss + train_at_loss + train_glimpse_loss

    val_enc_logits, val_at_logits, val_enc_pred, val_at_pred, val_enc_alphas, val_at_alphas, val_at_glimpses, val_nat_glimpses = eval_model(input_val_images, input_val_labels, reuse=True)

    train_data_list = get_data(args.train_data_dir,
                         args.train_data_gt,
                         args.voc_type,
                         args.max_len,
                         args.num_train,
                         args.height,
                         args.width,
                         args.train_batch_size,
                         args.workers,
                         args.keep_ratio,
                         with_aug=args.aug)

    val_data_gen = evaluator_data.Evaluator(lmdb_data_dir=args.test_data_dir,
                                            batch_size=args.val_batch_size,
                                            height=args.height,
                                            width=args.width,
                                            max_len=args.max_len,
                                            keep_ratio=args.keep_ratio,
                                            voc_type=args.voc_type)
    val_data_gen.reset()

    global_step = tf.get_variable(name='global_step', initializer=tf.constant(0), trainable=False)

    learning_rate = tf.train.piecewise_constant(global_step, args.decay_bound, args.lr_stage)
    batch_norm_updates_op = tf.group(tf.get_collection(tf.GraphKeys.UPDATE_OPS))

    # Save summary
    os.makedirs(args.checkpoints, exist_ok=True)
    tf.summary.scalar(name='learning_rate', tensor=learning_rate)
    tf.summary.scalar(name='train_loss', tensor=train_loss)
    tf.summary.scalar(name='train_enc_loss', tensor=train_enc_loss)
    tf.summary.scalar(name='train_at_loss', tensor=train_at_loss)
    tf.summary.scalar(name='train_glimpse_loss', tensor=train_glimpse_loss)

    merge_summary_op = tf.summary.merge_all()

    train_start_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    model_name = 'model_{:s}.ckpt'.format(str(train_start_time))
    model_save_path = os.path.join(args.checkpoints, model_name)
    best_model_save_path = os.path.join(args.checkpoints, 'best_model', model_name)
    variable_averages = tf.train.ExponentialMovingAverage(0.997, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    with tf.control_dependencies([variables_averages_op, batch_norm_updates_op]):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate) # 0.0001 constant
        grads = optimizer.compute_gradients(train_loss)
        if args.grad_clip > 0:
            print("With Gradients clipped!")
            for idx, (grad, var) in enumerate(grads):
                grads[idx] = (tf.clip_by_norm(grad, args.grad_clip), var)
        train_op = optimizer.apply_gradients(grads, global_step=global_step)

    saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)
    best_saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)
    summary_writer = tf.summary.FileWriter(args.checkpoints)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    log_file = open(os.path.join(args.checkpoints, "log.txt"), "w")
    with tf.Session(config=config) as sess:
        summary_writer.add_graph(sess.graph)
        start_iter = 0
        if args.resume == True and args.pretrained != '':
            print('Restore model from {:s}'.format(args.pretrained))
            ckpt_state = tf.train.get_checkpoint_state(args.pretrained)
            model_path = os.path.join(args.pretrained, os.path.basename(ckpt_state.model_checkpoint_path))
            saver.restore(sess=sess, save_path=model_path)
            start_iter = sess.run(tf.train.get_global_step())
        elif args.resume == False and args.pretrained != '':
            print('Restore pretrained model from {:s}'.format(args.pretrained))
            ckpt_state = tf.train.get_checkpoint_state(args.pretrained)
            model_path = os.path.join(args.pretrained, os.path.basename(ckpt_state.model_checkpoint_path))
            init = tf.global_variables_initializer()
            sess.run(init)
            pretrained_var = tf.get_collection(tf.GraphKeys.VARIABLES, scope='model')
            variable_restore_op = slim.assign_from_checkpoint_fn(model_path, pretrained_var, ignore_missing_vars=True)
            variable_restore_op(sess)
            sess.run(tf.assign(global_step, 0))
        else:
            print('Training from scratch')
            init = tf.global_variables_initializer()
            sess.run(init)

        # Evaluate the model first
        val_enc_pred_value_all = []
        val_at_pred_value_all = []
        val_labels = []
        for eval_iter in range(val_data_gen.num_samples // args.val_batch_size):

            val_data = val_data_gen.get_batch()
            if val_data is None:
                break
            print("Evaluation: [{} / {}]".format(eval_iter, (val_data_gen.num_samples // args.val_batch_size)))
            val_enc_pred_value, val_at_pred_value = sess.run([val_enc_pred, val_at_pred], feed_dict={input_val_images: val_data[0], input_val_labels: val_data[1], input_val_labels_mask: val_data[2]})
            val_enc_pred_value_all.extend(val_enc_pred_value)
            val_at_pred_value_all.extend(val_at_pred_value)
            val_labels.extend(val_data[4])

        val_data_gen.reset()

        val_enc_metrics_result = calc_metrics(idx2label(np.array(val_enc_pred_value_all), char2id=char2id, id2char=id2char), val_labels, metrics_type="accuracy")
        val_at_metrics_result = calc_metrics(idx2label(np.array(val_at_pred_value_all), char2id=char2id, id2char=id2char), val_labels, metrics_type="accuracy")
        print("Evaluation Before training: Mask Test Accuracy: {:3f} AT Test Accuracy: {:3f}".format(val_enc_metrics_result, val_at_metrics_result))

        val_best_acc = val_enc_metrics_result

        while start_iter < args.iters:
            start_iter += 1
            train_data = get_batch_data(train_data_list, args.train_batch_size)
            _, train_loss_value, train_enc_loss_value, train_at_loss_value, train_glimpse_loss_value = sess.run([train_op, train_loss, train_enc_loss, train_at_loss, train_glimpse_loss], feed_dict={input_train_images: train_data[0],
                                                                                                                                                                                                      input_train_labels: train_data[1],
                                                                                                                                                                                                      input_train_labels_mask: train_data[2]})

            if start_iter % args.log_iter == 0:
                print("Iter {} train loss= {:3f} (train_mask_loss: {:3f} train_at_loss: {:3f} train_glimpse_loss: {:3f})".format(start_iter, train_loss_value, train_enc_loss_value, train_at_loss_value, train_glimpse_loss_value))
                log_file.write("Iter {} train loss= {:3f} (train_mask_loss: {:3f} train_at_loss: {:3f} train_glimpse_loss: {:3f})\n".format(start_iter, train_loss_value, train_enc_loss_value, train_at_loss_value, train_glimpse_loss_value))
            if start_iter % args.summary_iter == 0:
            # if start_iter % 100 == 0            :
                merge_summary_value, train_enc_pred_value, train_at_pred_value = sess.run([merge_summary_op,train_enc_pred, train_at_pred], feed_dict={input_train_images: train_data[0],
                                                                                                                                                       input_train_labels: train_data[1],
                                                                                                                                                       input_train_labels_mask: train_data[2]})

                summary_writer.add_summary(summary=merge_summary_value, global_step=start_iter)
                if start_iter % args.eval_iter == 0:
                    val_enc_pred_value_all = []
                    val_at_pred_value_all = []
                    val_labels = []
                    for eval_iter in range(val_data_gen.num_samples // args.val_batch_size):
                        val_data = val_data_gen.get_batch()
                        if val_data is None:
                            break
                        print("Evaluation: [{} / {}]".format(eval_iter, (val_data_gen.num_samples // args.val_batch_size)))
                        val_enc_pred_value, val_at_pred_value = sess.run([val_enc_pred, val_at_pred], feed_dict={input_val_images: val_data[0],
                                                                                                                 input_val_labels: val_data[1],
                                                                                                                 input_val_labels_mask: val_data[2]})
                        val_enc_pred_value_all.extend(val_enc_pred_value)
                        val_at_pred_value_all.extend(val_at_pred_value)
                        val_labels.extend(val_data[4])
                    val_data_gen.reset()

                    train_enc_metrics_result = calc_metrics(idx2label(train_enc_pred_value, char2id=char2id, id2char=id2char), train_data[3], metrics_type="accuracy")
                    val_enc_metrics_result = calc_metrics(idx2label(np.array(val_enc_pred_value_all), char2id=char2id, id2char=id2char), val_labels, metrics_type="accuracy")
                    train_at_metrics_result = calc_metrics(idx2label(train_at_pred_value, char2id=char2id, id2char=id2char), train_data[3], metrics_type="accuracy")
                    val_at_metrics_result = calc_metrics(idx2label(np.array(val_at_pred_value_all), char2id=char2id, id2char=id2char), val_labels, metrics_type="accuracy")

                    print("Best accuracy= {:3f}".format(val_best_acc))
                    print("Evaluation Iter {} train_mask_accuracu= {:3f} train_at_accuracy= {:3f} test_mask_accuracy= {:3f} test_at_accuracy= {:3f}".format(start_iter,
                                                                                                                                                            train_enc_metrics_result,
                                                                                                                                                            train_at_metrics_result,
                                                                                                                                                            val_enc_metrics_result,
                                                                                                                                                            val_at_metrics_result))

                    log_file.write("Evaluation Iter {} train_mask_accuracu= {:3f} train_at_accuracy= {:3f} test_mask_accuracy= {:3f} test_at_accuracy= {:3f}".format(start_iter,
                                                                                                                                                            train_enc_metrics_result,
                                                                                                                                                            train_at_metrics_result,
                                                                                                                                                            val_enc_metrics_result,
                                                                                                                                                            val_at_metrics_result))

                    if val_enc_metrics_result >= val_best_acc:
                        print("Better results! Save checkpoitns to {}".format(best_model_save_path))
                        val_best_acc = val_enc_metrics_result
                        best_saver.save(sess, best_model_save_path, global_step=global_step)

            if start_iter % args.save_iter == 0:
                print("Iter {} save to checkpoint".format(start_iter))
                saver.save(sess, model_save_path, global_step=global_step)
    log_file.close()
if __name__ == "__main__":
    args = get_args(sys.argv[1:])
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    main_train(args)