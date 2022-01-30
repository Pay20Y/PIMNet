import tensorflow as tf
from module.Backbone import Backbone
from module.iterative_decoder import IterativeDecoder
from module.at_decoder import AT_Decoder

class Model(object):
    def __init__(self,
                 num_classes,
                 num_block,
                 embed_dim,
                 att_dim,
                 num_head,
                 hidden_units,
                 num_decoder,
                 seq_len=25,
                 is_training=True):
        self.num_classes = num_classes
        self.seq_len = seq_len
        self.is_training = is_training
        self.embedding_dim = embed_dim
        transformer_params = {'num_block': num_block, "att_dim": att_dim, "num_head": num_head, "hidden_units": hidden_units}

        self.backbone = Backbone(is_training=self.is_training)
        self.iterative_decoder = IterativeDecoder(output_classes=self.num_classes, seq_len=self.seq_len, num_iter=num_decoder, embed_dim=embed_dim, transformer_params=transformer_params, is_training=self.is_training)
        self.at_decoder = AT_Decoder(output_classes=self.num_classes, embedding_dim=embed_dim, transformer_params=transformer_params, seq_len=self.seq_len, is_training=self.is_training)

    def position_embedding(self, resue=False):
        with tf.variable_scope("share_position_embedding", reuse=resue):
            pos_range = tf.range(self.seq_len, dtype=tf.int32)
            w = tf.get_variable('embed_w', [self.seq_len, self.embedding_dim], initializer=tf.random_uniform_initializer(minval=-1.0, maxval=1.0))
            pos_embed = tf.nn.embedding_lookup(w, pos_range, name='pos_embed')

            return pos_embed

    def __call__(self, input_images, input_labels, input_length_mask=None, reuse=False, decode_type='greed'):
        with tf.variable_scope(name_or_scope="model", reuse=reuse):
            feature_map, tu_feature_map = self.backbone(input_images)
            pos_embed = self.position_embedding() # Share the position embedding with two decoders
            at_logits, at_alphas, at_glimpses = self.at_decoder(tu_feature_map, input_labels, pos_embed)
            at_pred = tf.argmax(at_logits, axis=2)
            logits, pred, alphas, nat_glimpses = self.iterative_decoder(tu_feature_map, input_labels=input_labels, pos_embedding=pos_embed)

            return logits, at_logits, pred, at_pred, alphas, at_alphas, nat_glimpses, at_glimpses

    def glimpse_mimic_loss(self, nat_glimpse, at_glimpse, input_lengths_mask=None):
        def smooth_L1(inputs, targets):
            with tf.name_scope("smooth_l1"):
                inside = tf.subtract(inputs, targets)
                smooth_l1_sign = tf.cast(tf.less(tf.abs(inside), 1.0), tf.float32)
                smooth_l1_option1 = tf.multiply(tf.multiply(inside, inside), 0.5)
                smooth_l1_option2 = tf.subtract(tf.abs(inside), 0.5)

                smooth_l1_result = tf.add(tf.multiply(smooth_l1_option1, smooth_l1_sign), tf.multiply(smooth_l1_option2, tf.abs(tf.subtract(smooth_l1_sign, 1.0))))

                return smooth_l1_result
        def cos_sim(x, y):
            with tf.name_scope("cosin_similarity"):
                norm_x = tf.math.l2_normalize(x, 1)
                norm_y = tf.math.l2_normalize(y, 1)
                output = tf.squeeze(tf.squeeze(tf.matmul(tf.expand_dims(norm_x, axis=1), tf.transpose(tf.expand_dims(norm_y, axis=1), [0, 2, 1])), axis=2), axis=1)
                return output

        with tf.name_scope("attention_loss"):
            N, T, C = nat_glimpse.shape.as_list()
            nat_glimpse = tf.reshape(nat_glimpse, [(N * T), C])
            at_glimpse = tf.reshape(at_glimpse, [(N * T), C])

            at_glimpse = tf.stop_gradient(at_glimpse)

            loss = 1. - cos_sim(nat_glimpse, at_glimpse)
            # loss = tf.reduce_mean(smooth_L1(nat_glimpse, at_glimpse), axis=1)

            loss = tf.reshape(loss, [N, T])

            if input_lengths_mask is not None:
                loss = loss * tf.cast(input_lengths_mask, tf.float32)

                # Normalize for each example
                loss = tf.reduce_sum(loss, axis=1) # N
                valid_num = tf.reduce_sum(input_lengths_mask, axis=1)
                loss = loss / (tf.cast(valid_num, tf.float32) + 1e-8)

            loss = tf.reduce_sum(loss) / N
            return loss

    def loss(self, pred, input_labels, input_lengths_mask, train_random_mask=None):
        """
        cross-entropy loss
        :param pred: Decoder outputs N * L * C
        :param input_labels: N * L
        :param input_lengths_mask: N * L (0 & 1 like indicating the real length of the label)
        :param pred_mask: N * L
        :return:
        """
        with tf.name_scope(name="MaskCrossEntropyLoss"):
            input_labels = tf.one_hot(input_labels, self.num_classes, 1, 0) # N * L * C
            input_labels = tf.stop_gradient(input_labels) # since softmax_cross_entropy_with_logits_v2 will bp to labels
            loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=input_labels, logits=pred, dim=-1)
            mask_loss = loss * tf.cast(input_lengths_mask, tf.float32)

            if train_random_mask is not None:
                mask_loss = mask_loss * tf.cast(train_random_mask, tf.float32)

            loss = tf.reduce_sum(mask_loss) / tf.cast(tf.shape(pred)[0], tf.float32)

            return loss