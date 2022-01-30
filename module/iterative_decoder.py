import tensorflow as tf
import math

from module.parallel_decoder import Decoder

class IterativeDecoder():
    def __init__(self, output_classes, seq_len, num_iter, embed_dim, transformer_params, is_training=True):
        self.output_classes = output_classes
        self.seq_len = seq_len
        self.num_iter = num_iter
        self.is_training = is_training
        self.MASK_TOKEN = self.output_classes - 2
        self.EOS_TOKEN  = self.output_classes - 1
        self.top_k = math.ceil(self.seq_len / self.num_iter)

        self.decoder = Decoder(output_classes=self.output_classes, embedding_dim=embed_dim, transformer_params=transformer_params, seq_len=seq_len, is_training=self.is_training)

    def easy_first(self, feature_map, input_labels=None, pos_embedding=None):
        """
        Easy first decoding strategy
        :param feature_map: N * H * W * C
        :param input_labels: N * T
        :param input_length_mask: N * T
        :return:
        """
        print("easy first decoding")
        with tf.variable_scope("iterative_decoder"):
            N, _, _, C = feature_map.shape.as_list()

            tgt_tokens = tf.ones(dtype=tf.int32, shape=[N, self.seq_len]) * self.MASK_TOKEN
            pred_tgt_tokens = tf.ones(dtype=tf.int32, shape=[N, self.seq_len]) * self.MASK_TOKEN
            token_logits = tf.zeros(dtype=tf.float32, shape=[N, self.seq_len, self.output_classes])
            final_ffn = tf.zeros(dtype=tf.float32, shape=[N, self.seq_len, C])

            for i in range(self.num_iter):

                new_token_logits, alpha, token_probs, new_ffn = self.decoder(tgt_tokens, feature_map, pos_embedding=pos_embedding, reuse=(i != 0))

                new_tgt_tokens = tf.cast(tf.argmax(new_token_logits, axis=2), tf.int32)
                token_probs = tf.reduce_max(token_probs, axis=2)  # N * T
                token_probs = tf.where(tf.equal(tgt_tokens, self.MASK_TOKEN), token_probs, tf.zeros_like(token_probs)) # A character won't be updated two times

                top_tuple = tf.nn.top_k(token_probs, self.top_k)  # The top-k best position
                kth = tf.reduce_min(top_tuple.values, 1, keep_dims=True)
                update_idx = tf.greater_equal(token_probs, kth)  # positions to update

                logits_update_idx = tf.tile(tf.expand_dims(update_idx, axis=2), [1, 1, self.output_classes])
                ffn_update_idx = tf.tile(tf.expand_dims(update_idx, axis=2), [1, 1, C])

                if self.is_training:
                    tgt_tokens = tf.where(update_idx, input_labels, tgt_tokens)
                    pred_tgt_tokens = tf.where(update_idx, new_tgt_tokens, pred_tgt_tokens)
                else:
                    tgt_tokens = tf.where(update_idx, new_tgt_tokens, tgt_tokens)  # Update the low-confident tokens
                    pred_tgt_tokens = tgt_tokens
                token_logits = tf.where(logits_update_idx, new_token_logits, token_logits)
                final_ffn = tf.where(ffn_update_idx, new_ffn, final_ffn)

            return token_logits, pred_tgt_tokens, alpha, final_ffn

    def __call__(self, feature_map, input_labels=None, pos_embedding=None):
        return self.easy_first(feature_map, input_labels=input_labels, pos_embedding=pos_embedding)