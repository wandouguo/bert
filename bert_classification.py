import modeling
import tensorflow as tf
import json


class Model(object):

    def __init__(self,
                 bert_checkpoint_path,
                 bert_config_path,
                 num_labels,
                 is_training=True, use_one_hot_embeddings=False):
        self.bert_checkpoint_path = bert_checkpoint_path
        self.bert_config_path = bert_config_path
        self.bert_config = json.load(open(self.bert_config_path))
        print("self.bert_config:", self.bert_config)
        self.num_labels = num_labels
        self.is_training = is_training
        self.use_one_hot_embeddings = use_one_hot_embeddings

    def build_model(self):
        # placeholders
        self.input_ids = tf.placeholder(dtype=tf.int32, shape=[None, None], name='input_ids')
        self.input_mask = tf.placeholder(dtype=tf.int32, shape=[None, None], name='input_mask')
        self.segment_ids = tf.placeholder(dtype=tf.int32, shape=[None, None], name='segment_ids')
        self.labels = tf.placeholder(dtype=tf.int32, shape=[None], name='labels')
        self.dropout_prob = tf.placeholder(dtype=tf.float32, name='Dropout')
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.temperature = tf.placeholder(dtype=tf.float32, name="temperature")
        ###初始化bert
        bert_config = modeling.BertConfig.from_json_file(self.bert_config_path)
        model = modeling.BertModel(
            config=bert_config,
            is_training=self.is_training,
            input_ids=self.input_ids,
            input_mask=self.input_mask,
            token_type_ids=self.segment_ids,
            use_one_hot_embeddings=self.use_one_hot_embeddings)
        output_layer = model.get_pooled_output()

        hidden_size = output_layer.shape[-1].value

        output_weights = tf.get_variable(
            "output_weights", [self.num_labels, hidden_size],
            initializer=tf.truncated_normal_initializer(stddev=0.02))
        output_bias = tf.get_variable(
            "output_bias", [self.num_labels], initializer=tf.zeros_initializer())

        if self.is_training:
            # I.e., 0.1 dropout
            output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)
        else:
            output_layer = tf.nn.dropout(output_layer, keep_prob=1)
        logits = tf.matmul(output_layer, output_weights, transpose_b=True)
        self.logits = tf.nn.bias_add(logits, output_bias)
        self.output = tf.arg_max(logits, -1, name="output")
        self.softmax = tf.nn.softmax(logits / self.temperature, axis=-1, name="probabilities")
        correct_prediction = tf.equal(tf.cast(self.output, tf.int32), self.labels)
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="accuracy")
        self.recall = tf.metrics.recall(self.labels, self.output)
        self.precision = tf.metrics.precision(self.labels, self.output)
        self.loss = tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(self.labels,self.num_labels), logits=logits)
        self.loss = tf.reduce_mean(self.loss)
        optimizer = tf.train.AdamOptimizer(3e-5)
        self.train_op = optimizer.minimize(self.loss, global_step=self.global_step)
        return (self.loss, self.logits, self.output, self.softmax, self.accuracy)

    def init_model(self):
        (loss, logits, output, probabilities, accuracy) = self.build_model()
        tvars = tf.trainable_variables()
        initialized_variable_names = {}
        (assignment_map, initialized_variable_names
         ) = modeling.get_assignment_map_from_checkpoint(tvars, self.bert_checkpoint_path)
        tf.train.init_from_checkpoint(self.bert_checkpoint_path, assignment_map)
        return (loss, logits, output, probabilities, accuracy)

