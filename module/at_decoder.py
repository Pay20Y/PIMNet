import tensorflow as tf
from module.Transformer_Modules import ff, multihead_attention, positional_encoding

class AT_Decoder(object):
    def __init__(self, output_classes, transformer_params, seq_len=25, embedding_dim=512, att_dim=512, is_training=True, keep_prob=0.9):
        self.output_classes = output_classes
        self.seq_len = seq_len
        self.embedding_dim = embedding_dim
        self.att_dim = att_dim
        self.is_training = is_training
        if self.is_training:
            self.keep_prob = keep_prob
        else:
            self.keep_prob = 1.0
        self.dropout_rate = 1 - self.keep_prob
        self.weight_initializer = tf.contrib.layers.xavier_initializer()
        self.const_initializer = tf.constant_initializer(0.0)
        self.emb_initializer = tf.random_uniform_initializer(minval=-1.0, maxval=1.0)
        self.START_TOKEN = output_classes - 1  # Same like EOS TOKEN
        self.MASK_TOKEN = output_classes - 2
        self.transformer_params = transformer_params

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

    def mask_self_attention(self, inputs, mask=None, causality=True, backward=False, reuse=False):
        assert (causality ^ backward) or (causality == False and backward == False)
        with tf.variable_scope('mask_self_attention', reuse=reuse):
            # multi head mask attention
            C = inputs.shape.as_list()[-1]
            outputs, _ = multihead_attention(queries=inputs, keys=inputs, values=inputs, key_masks=mask, causality=causality, num_heads=self.transformer_params['num_head'], training=self.is_training, dropout_rate=self.dropout_rate)
            outputs = ff(outputs, num_units=[self.transformer_params['hidden_units'], C])
            return outputs

    def transformer_attention_op(self, query, feature_map, reuse=False):
        with tf.variable_scope("attention_op", reuse=reuse):
            N, H, W, C = feature_map.shape.as_list()
            key_vec = tf.reshape(feature_map, [N, (H * W), C])
            outputs, alphas = multihead_attention(queries=query, keys=key_vec, values=key_vec,
                                                  num_heads=self.transformer_params['num_head'], training=self.is_training, dropout_rate=self.dropout_rate)
            outputs = ff(outputs, num_units=[self.transformer_params['hidden_units'], self.att_dim])
            return outputs, alphas

    def decode_op(self, inputs, reuse=False):
        with tf.variable_scope("logits", reuse=reuse):
            N, T, C = inputs.shape.as_list()
            output_w = tf.get_variable(name="output_w", shape=[C, self.output_classes], initializer=self.weight_initializer)
            output_b = tf.get_variable(name="output_b", shape=[self.output_classes], initializer=self.const_initializer)
            inputs = tf.reshape(inputs, [(N * T), C])
            logistic = tf.matmul(inputs, output_w) + output_b
            logistic = tf.reshape(logistic, [N, T, self.output_classes])
            return logistic

    def __call__(self, feature_map, input_labels, pos_embedding, scope="at_decoder"):
        """
        Transformer block for decoding based encoding outputs
        :param encoder_output: N * T * self.output_classes
        :param feature_map: N * H * W * C
        :param labels: N * T
        :return:
        """
        with tf.variable_scope(scope):
            print("Build auto-regressive decoder!")
            N, H, W, C = feature_map.shape.as_list()

            #########################################################################################################################
            ################################# Prepare the input label ###############################################################
            if self.is_training:
                input_labels = tf.unstack(input_labels, axis=1)[:-1]
                input_labels.insert(0, tf.fill([N], value=self.MASK_TOKEN)) # START_TOKEN
                input_labels = tf.stack(input_labels, axis=1)
                #########################################################################################################################

                #########################################################################################################################
                ################################# Character Embedding ###################################################################

                char_embedding = self._char_embedding(input_labels)

                #########################################################################################################################

                #########################################################################################################################
                ################################# Self Attention for Character Embedding ################################################

                if pos_embedding is None:
                    pos_embed = positional_encoding(char_embedding, maxlen=self.seq_len, masking=False)
                else:
                    pos_embed = pos_embedding

                char_embedding += pos_embed
                char_embedding = tf.nn.dropout(char_embedding, keep_prob=self.keep_prob)
                block_output = self.mask_self_attention(char_embedding, causality=True)

                #########################################################################################################################

                #########################################################################################################################
                ############################################# 2D Attention ##############################################################

                outputs, alphas = self.transformer_attention_op(block_output, feature_map)

                logits = self.decode_op(outputs)

            else:
                predicts = [tf.fill([N], value=self.MASK_TOKEN)] * self.seq_len # START_TOKEN
                pre_pred = tf.fill([N], value=self.MASK_TOKEN) # START_TOKEN
                for t in range(self.seq_len):
                    predicts[t] = pre_pred
                    predicts_tensor = tf.stack(predicts, axis=1)

                    char_embedding = self._char_embedding(predicts_tensor, reuse=(t != 0))

                    if pos_embedding is None:
                        pos_embed = positional_encoding(char_embedding, maxlen=self.seq_len, masking=False)
                    else:
                        pos_embed = pos_embedding
                    char_embedding += pos_embed

                    self_att_outputs = self.mask_self_attention(char_embedding, causality=True, reuse=(t != 0))  # N * T * ~

                    # 2D scale dot attention
                    outputs, alphas = self.transformer_attention_op(self_att_outputs, feature_map, reuse=(t != 0))

                    logits = self.decode_op(outputs, reuse=(t != 0))

                    current_logits = tf.unstack(logits, axis=1)[t]
                    pre_pred = tf.cast(tf.argmax(current_logits, axis=1), tf.int32)  # N

            return logits, alphas, outputs
