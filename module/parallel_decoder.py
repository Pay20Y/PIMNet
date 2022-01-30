import tensorflow as tf
from tensorflow.contrib import slim
from module.Transformer_Modules import multihead_attention, positional_encoding, ff

class Decoder(object):
    def __init__(self, output_classes, embedding_dim, transformer_params, seq_len=25, is_training=True):
        self.output_classes = output_classes
        self.embedding_dim = embedding_dim
        self.seq_len = seq_len
        self.transformer_params = transformer_params
        self.is_training = is_training
        self.dropout_rate = 0.1
        self.MASK_TOKEN = self.output_classes - 2
        self.EOS_TOKEN = self.output_classes - 1
        self.weight_initializer = tf.contrib.layers.xavier_initializer()
        self.const_initializer = tf.constant_initializer(0.0)
        self.emb_initializer = tf.random_uniform_initializer(minval=-1.0, maxval=1.0)

    def _char_embedding(self, inputs, reuse=False):
        """
        Embedding the input character
        :param inputs: [N * self.output_classes] one-hot tensor
        :param reuse:
        :return: N * self.embedding_dim
        """
        with tf.variable_scope('word_embedding', reuse=reuse):
            w = tf.get_variable('embed_w', [self.output_classes, self.embedding_dim], initializer=self.emb_initializer)
            x = tf.nn.embedding_lookup(w, inputs, name='word_vector')  # (N, T, M) or (N, M)
            return x

    def mask_self_attention(self, inputs, mask=None, causality=False, backward=False, reuse=False):
        assert (causality ^ backward) or (causality == False and backward == False)
        with tf.variable_scope('mask_self_attention', reuse=reuse):
            # multi head mask attention
            C = inputs.shape.as_list()[-1]
            outputs, _ = multihead_attention(queries=inputs, keys=inputs, values=inputs, key_masks=mask, causality=causality, num_heads=self.transformer_params['num_head'], training=self.is_training, dropout_rate=self.dropout_rate)
            outputs = ff(outputs, num_units=[self.transformer_params['hidden_units'], C])
            return outputs

    def cross_attention(self, query, feature_map, reuse=False):
        with tf.variable_scope("cross_attention", reuse=reuse):
            N, H, W, C = feature_map.shape.as_list()
            key_vec = tf.reshape(feature_map, [N, (H * W), C])
            outputs, alphas = multihead_attention(queries=query, keys=key_vec, values=key_vec, num_heads=self.transformer_params['num_head'], training=self.is_training, dropout_rate=self.dropout_rate)
            outputs = ff(outputs, num_units=[self.transformer_params['hidden_units'], C])
            alphas = tf.reshape(tf.stack(tf.split(alphas, self.transformer_params['num_head'], axis=0), axis=1), [N, self.transformer_params['num_head'], self.seq_len, H, W])
            return outputs, alphas

    def decode_op(self, inputs):
        with tf.variable_scope("logits"):
            N, T, C = inputs.shape.as_list()
            output_w = tf.get_variable(name="output_w", shape=[C, self.output_classes], initializer=self.weight_initializer)
            output_b = tf.get_variable(name="output_b", shape=[self.output_classes], initializer=self.const_initializer)
            inputs = tf.reshape(inputs, [(N * T), C])
            logistic = tf.matmul(inputs, output_w) + output_b
            logistic = tf.reshape(logistic, [N, T, self.output_classes])
            return logistic

    def gen_eos_mask(self, inputs):
        """
        Mask the right and itself of the first EOS token
        :param inputs: N * T
        :return: N * T
        """
        with tf.name_scope("eos_mask"):
            N, T, = inputs.shape.as_list()
            eos_mask = tf.where(tf.equal(inputs, self.output_classes-1), tf.ones_like(inputs), tf.zeros_like(inputs))
            eos_mask = tf.transpose(tf.tile(tf.expand_dims(eos_mask, axis=1), [1, T, 1]), [0, 2, 1])

            diag_vals = tf.ones(dtype=tf.float32, shape=[T, T])
            tril = 1. - tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()  # Here we reserve the first eos token
            tril = tf.tile(tf.expand_dims(tril, axis=0), [N, 1, 1])
            tril = tf.cast(tril, tf.int32)

            eos_right_mask = tf.where(tf.reduce_any(tf.equal(tril * eos_mask, 1), axis=1), tf.ones_like(inputs), tf.zeros_like(inputs)) # N * T

            return eos_right_mask

    def bi_bert(self, predicts, feature_map, pos_embedding, reuse=False):
        with tf.variable_scope("encoder", reuse=reuse):

            mask_mask = tf.where(tf.equal(predicts, self.MASK_TOKEN), tf.ones_like(predicts), tf.zeros_like(predicts))  # N * T
            # Text Length post-process
            eos_mask = self.gen_eos_mask(predicts)  # N * T

            key_mask = mask_mask + eos_mask
            key_mask = tf.where(tf.equal(key_mask, 0), tf.zeros_like(key_mask), tf.ones_like(key_mask))

            predicts_embed = self._char_embedding(predicts)

            if pos_embedding is None:
                pos_embed = positional_encoding(predicts_embed, maxlen=self.seq_len)
            else:
                pos_embed = pos_embedding
            predicts_embed += pos_embed

            self_attention_outputs = self.mask_self_attention(predicts_embed, mask=key_mask, causality=False)
            outputs, alphas = self.cross_attention(self_attention_outputs, feature_map)
            logits = self.decode_op(outputs)
            probs = tf.nn.softmax(logits, axis=2)

            return logits, alphas, probs, outputs

    def __call__(self, predicts, feature_map, pos_embedding=None, reuse=False):
        print("Build non-autoregressive decoder!")
        return self.bi_bert(predicts, feature_map, pos_embedding, reuse)
