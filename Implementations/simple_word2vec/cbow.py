import tensorflow as tf


class CBOW(object):
    def __init__(self, args):
        self.__args = args
        self.__ngram_size = args.ngram_size
        self.__input_size = self.__ngram_size - 1
        self.__vocab_size = args.vocab_size + 1
        self.__embedding_dim = args.embedding_dim
        self.__learning_rate = args.learning_rate
        self.__activation_function = args.activation_function
        self.__optimizer = args.optimizer
        self.__loss_function = args.loss_function

    def init_session(self, restore=False):
        self.__session = tf.Session()

        if restore:
            self.__saver = tf.train.Saver()
            self.__saver.restore(self.__session, self.__args.model)

    def build(self):
        self.__input = tf.placeholder(tf.float32, [None, self.__input_size * self.__vocab_size])
        self.__output = tf.placeholder(tf.float32, [None, self.__vocab_size])

        self.__input_to_hidden_weights = tf.get_variable("ih_w", shape=[self.__input_size * self.__vocab_size, self.__embedding_dim],
                                                  initializer=tf.contrib.layers.xavier_initializer())
        self.__input_to_hidden_bias = tf.Variable(tf.ones(self.__embedding_dim))
        self.__hidden_to_output_weights = tf.get_variable("ho_w", shape=[self.__embedding_dim, self.__vocab_size], initializer=tf.contrib.layers.xavier_initializer())
        self.__hidden_to_output_bias = tf.Variable(tf.ones([self.__vocab_size]))

        if self.__optimizer.lower() == "sgd":
            self.__optimizer = tf.train.GradientDescentOptimizer(self.__learning_rate)
        elif self.__optimizer.lower() == "adam":
            self.__optimizer = tf.train.AdamOptimizer(self.__learning_rate)

        self.__embedding_layer = tf.matmul(self.__input, self.__input_to_hidden_weights) + self.__input_to_hidden_bias

        if self.__activation_function.lower() == "tanh":
            self.__embedding_layer = tf.nn.tanh(self.__embedding_layer)
        elif self.__activation_function.lower() == "relu":
            self.__embedding_layer = tf.nn.relu(self.__embedding_layer)

        self.__output_layer = tf.matmul(self.__embedding_layer, self.__hidden_to_output_weights) + self.__hidden_to_output_bias
        self.__output_layer = tf.nn.softmax(self.__output_layer)

        if self.__loss_function.lower() == "mse":
            self.__cost_function = 0.5 * tf.reduce_sum(tf.square(self.__output_layer - self.__output))
        elif self.__loss_function.lower() == "ce":
            self.__cost_function = -tf.reduce_mean((self.__output * tf.log(self.__output_layer)) + ((1 - self.__output) * tf.log(1 - self.__output_layer)))

        self.__train = self.__optimizer.minimize(self.__cost_function)

    def run(self, x_input, y_output):
        self.__session.run(tf.global_variables_initializer())
        error = self.__session.run(self.__cost_function, feed_dict={self.__input: x_input, self.__output: y_output})

        return error