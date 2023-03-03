# reference: https://github.com/google-research/bert/
import copy, json, six, pickle
from typing import Optional, List
import tensorflow as tf
from tensorflow.keras import layers, Model, Input


### Multi CUDA GPU Selection ###
def SelectStrategy() -> tf.distribute.Strategy:
    devices: List[tf.config.PhysicalDevice] = tf.config.list_physical_devices("GPU")
    if len(devices) == 0:
        raise RuntimeError("No GPU found in CUDA_VISIBLE_DEVICE")
    elif len(devices) > 1:
        # choice: HierarchicalCopyAllReduce / ReductionToOneDevice / NcclAllReduce (default)
        strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.ReductionToOneDevice())
    else:
        strategy = tf.distribute.OneDeviceStrategy("/gpu:0")
    return strategy


### Bert Settings ###
class BertConfig(object):
    """Configuration for `BertModel`."""
    def __init__(
        self,
        vocab_size, 
        pos_scale=365.2425,
        num_layers=12, 
        hidden_size=768, 
        intermediate_size=3072, 
        num_heads=12, 
        hidden_dropout=0.1, 
        attention_dropout=0.1, 
        hidden_act="gelu",
        epsilon=1e-12, 
        initializer="truncated_normal",
        ):
        self.vocab_size = vocab_size
        self.pos_scale = pos_scale
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_heads = num_heads
        self.hidden_dropout = hidden_dropout
        self.attention_dropout = attention_dropout
        self.hidden_act = hidden_act
        self.epsilon = epsilon
        self.initializer = initializer

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `BertConfig` from a Python dictionary of parameters."""
        config = BertConfig(vocab_size=None)
        for (key, value) in six.iteritems(json_object):
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `BertConfig` from a json file of parameters."""
        with tf.io.gfile.GFile(json_file, "r") as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


### Activation Functions ###
def get_activation(activation_string):
    if not isinstance(activation_string, six.string_types):
        return activation_string
    if not activation_string:
        return None

    act = activation_string.lower()
    if act == "linear":
        return None
    elif act == "relu":
        return tf.nn.relu
    elif act == "gelu":
        return tf.nn.gelu
    elif act == "tanh":
        return tf.nn.tanh
    elif act == "swish":
        return tf.nn.swish
    else:
        raise ValueError("Unsupported activation: %s" % act)


### Initializers ###
def get_initializers(initializer_string):
    """
    Truncated normal initializer with "initializer_range=0.02" 
    is the default setting in BertConfig
    """
    if not isinstance(initializer_string, six.string_types):
        return initializer_string
    if not initializer_string:
        return None

    act = initializer_string.lower()
    if act == "truncated_normal":
        return tf.initializers.truncated_normal(stddev=0.02)
    elif act == "glorot_normal":
        return tf.initializers.glorot_normal()
    elif act == "glorot_uniform":
        return tf.initializers.glorot_uniform()
    elif act == "he_normal":
        return tf.initializers.he_normal()
    elif act == "he_uniform":
        return tf.initializers.he_uniform()
    else:
        raise ValueError("Unsupported activation: %s" % act)


### Custom Absolute Position Encoding ###
class PositionEncoding(layers.Layer):
    def __init__(self, depth, scale):
        super(PositionEncoding, self).__init__()
        assert depth % 2 == 0, "The number of hidden size is NOT a multiple of 2."
        self.depth = depth // 2
        self.scale = scale

    @tf.function
    def call(self, inputs):
        assert len(inputs.shape) == 2, "Position Input shape must be 2D."
        positions = inputs[:,:,tf.newaxis]
        depths = tf.cast(tf.range(self.depth)[tf.newaxis,:]/self.depth, dtype=inputs.dtype)
        angle_rates = 1 / (10000**depths) * self.scale # 1 for ordinary sin_cos position
        angle_rads = tf.einsum('ijk,kl->ijl', positions, angle_rates) # (bs,pos,depth)
        sin_part = tf.math.sin(angle_rads) # sin to even indices in the array; 2i
        cos_part = tf.math.cos(angle_rads) # cos to  odd indices in the array; 2i+1
        return tf.reshape(tf.stack([sin_part, cos_part], axis=-1), [-1, inputs.shape[-1], self.depth*2])


### Transformer Encoder Layer ###
class TransformerEncoderLayer(layers.Layer):
    def __init__(self, hidden_size, intermediate_size, num_heads, hidden_dropout, attention_dropout, hidden_act, epsilon, initializer):
        super(TransformerEncoderLayer, self).__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_heads = num_heads
        self.hidden_dropout = hidden_dropout
        self.attention_dropout = attention_dropout
        self.hidden_act = hidden_act
        self.epsilon = epsilon
        self.initializer = initializer
        self.mha = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=int(hidden_size//num_heads), dropout=attention_dropout, kernel_initializer=initializer
        )
        self.ffn = tf.keras.Sequential([
            layers.Dense(units=intermediate_size, activation=hidden_act, kernel_initializer=initializer), 
            layers.Dense(units=hidden_size, kernel_initializer=initializer)
        ])
        self.do1 = layers.Dropout(rate=hidden_dropout)
        self.do2 = layers.Dropout(rate=hidden_dropout)
        self.ln1 = layers.LayerNormalization(epsilon=epsilon)
        self.ln2 = layers.LayerNormalization(epsilon=epsilon)

    @tf.function
    def call(self, inputs, mask):
        mha_out = self.mha(inputs, inputs, inputs, attention_mask = mask)
        mha_out = self.do1(mha_out)
        mha_ln1 = self.ln1(mha_out + inputs)
        ffn_out = self.ffn(mha_ln1)
        ffn_out = self.do2(ffn_out)
        return self.ln2(ffn_out + mha_ln1)


### Matmul Weights and Add Bias Layers ###
class MatmulOutputWeights(layers.Layer):
    def __init__(self):
        super(MatmulOutputWeights, self).__init__()

    @tf.function
    def call(self, inputs, output_weights):
        return tf.matmul(inputs, output_weights, transpose_b=True)

class OutputWeights(layers.Layer):
    def __init__(self, b_units, h_units):
        super(OutputWeights, self).__init__()
        self.b_units = b_units
        self.h_units = h_units
        self.w_init = tf.initializers.truncated_normal(stddev=0.02)
        self.w = tf.Variable(initial_value=self.w_init(shape=(self.b_units, self.h_units,)))

    @tf.function
    def call(self, inputs):
        return tf.matmul(inputs, tf.cast(self.w, dtype=inputs.dtype), transpose_b=True)

class OutputBias(layers.Layer):
    def __init__(self, b_units):
        super(OutputBias, self).__init__(name="output_bias")
        self.b_units = b_units
        self.b_init = tf.zeros_initializer()
        self.b = tf.Variable(initial_value=self.b_init(shape=(self.b_units,)))

    @tf.function
    def call(self, inputs):
        return tf.nn.bias_add(inputs, tf.cast(self.b, dtype=inputs.dtype))


### BERT structure ###
class BertModel(Model):
    def __init__(
        self, 
        is_training: bool,
        vocab_size: int, 
        pos_scale: float,
        num_layers: int, 
        hidden_size: int, 
        intermediate_size: int, 
        num_heads: int, 
        hidden_dropout: float, 
        attention_dropout: float, 
        hidden_act: object,
        epsilon: float, 
        initializer: object,
        ):
        super(BertModel, self).__init__()
        self.is_training = is_training
        self.vocab_size = vocab_size
        self.pos_scale = pos_scale
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_heads = num_heads
        self.hidden_dropout = hidden_dropout
        self.attention_dropout = attention_dropout
        self.hidden_act = hidden_act
        self.epsilon = epsilon
        self.initializer = initializer
        if not is_training:
            self.hidden_dropout = 0.0
            self.attention_dropout = 0.0

        self.age_pos = PositionEncoding(depth=self.hidden_size, scale=self.pos_scale)
        self.emb = layers.Embedding(input_dim=self.vocab_size, output_dim=self.hidden_size, embeddings_initializer=self.initializer)
        self.ln = layers.LayerNormalization(epsilon=self.epsilon)
        self.do = layers.Dropout(rate=self.hidden_dropout)
        self.enc_layers = [
            TransformerEncoderLayer(
                self.hidden_size, self.intermediate_size, self.num_heads, self.hidden_dropout, self.attention_dropout, self.hidden_act, self.epsilon, self.initializer
                ) for _ in range(self.num_layers)
        ]
        self.pooler = layers.Dense(units=self.hidden_size, activation=tf.nn.tanh, kernel_initializer=self.initializer, name='pooled_output')

    @tf.function
    def call(self, inputs):
        ### prepare inputs -> omit segments(token_type_ids)
        inp, pos = inputs # order: sequence, position
        att_mask = tf.not_equal(inp, 0)
        att_mask = att_mask[:, tf.newaxis, tf.newaxis, :]
        inp = self.emb(inp)
        pos = self.age_pos(pos)
        ### combine inputs and ln/do
        x = inp + pos
        x = self.ln(x)
        x = self.do(x)
        ### BERT module
        for i in range(self.num_layers):
            x = self.enc_layers[i](x, att_mask)
        pooled_output = self.pooler(x[:, 0, :])
        return x, pooled_output, tf.cast(tf.squeeze(self.emb.weights), dtype=x.dtype)


### MLM task (next sentence prediction pre-training task is NOT included) ###
class GatherIndexes(layers.Layer):
    def __init__(self):
        super(GatherIndexes, self).__init__()

    @tf.function
    def call(self, sequence_tensor, positions):
        """Gathers the vectors at the specific positions over a minibatch."""
        assert len(sequence_tensor.shape) == 3, "The shape of sequences including dims must be 3D."
        batch_size = sequence_tensor.shape[0]
        seq_length = sequence_tensor.shape[1]
        width = sequence_tensor.shape[2]

        flat_offsets = tf.reshape(tf.range(0, batch_size, dtype=positions.dtype) * seq_length, [-1, 1])
        flat_positions = tf.reshape(positions + flat_offsets, [-1])
        flat_sequence_tensor = tf.reshape(sequence_tensor, [batch_size * seq_length, width])
        output_tensor = tf.gather(flat_sequence_tensor, tf.cast(flat_positions, dtype=tf.int32))
        return output_tensor


def MLMBert(
    is_training: bool,
    batch_size: int,
    seq_len: int,
    max_pred_per_seq: int,
    config: object,
    ):
    config = copy.deepcopy(config)
    ### inputs
    inp_seq = Input(batch_size=batch_size, shape=(seq_len,), dtype=tf.int64)
    inp_pos = Input(batch_size=batch_size, shape=(seq_len,), dtype=tf.float32)
    mlm_pos = Input(batch_size=batch_size, shape=(max_pred_per_seq,), dtype=tf.int64)
    ### BERT module
    sequence_output, _, embedding_table = BertModel(
        is_training, config.vocab_size, config.pos_scale, config.num_layers, config.hidden_size, config.intermediate_size, config.num_heads, 
        config.hidden_dropout, config.attention_dropout, get_activation(config.hidden_act), config.epsilon, get_initializers(config.initializer),
    )((inp_seq, inp_pos))
    print(sequence_output.dtype)
    ### the weights below this line are only used in pre-training
    x = GatherIndexes()(sequence_output, mlm_pos)
    x = layers.Dense(units=config.hidden_size, activation=get_activation(config.hidden_act), kernel_initializer=get_initializers(config.initializer))(x)
    x = layers.LayerNormalization(epsilon=config.epsilon)(x)
    logits = MatmulOutputWeights()(x, embedding_table)
    logits = OutputBias(b_units=config.vocab_size)(logits)
    log_probs = tf.cast(tf.nn.log_softmax(logits, axis=-1), dtype=tf.float32)
    ### model construction
    model = Model(inputs=[inp_seq, inp_pos, mlm_pos], outputs=log_probs)
    return model


### BERT Classifier structure ###
def ClassifierBert(
    is_training: bool, 
    seq_len: int, 
    num_labels: int,
    config: object,
    bert_module_weights_path: Optional[str] = None, 
    ):
    config = copy.deepcopy(config)
    if not is_training:
        config.hidden_dropout=0.0
        config.attention_dropout=0.0
    ### inputs
    inp_seq = Input(shape=(seq_len,), dtype=tf.int64)
    inp_pos = Input(shape=(seq_len,), dtype=tf.float32)
    ### BERT module
    _, pooled_output, _ = BertModel(
        is_training, config.vocab_size, config.pos_scale, config.num_layers, config.hidden_size, config.intermediate_size, config.num_heads, 
        config.hidden_dropout, config.attention_dropout, get_activation(config.hidden_act), config.epsilon, get_initializers(config.initializer),
    )((inp_seq, inp_pos))
    ### Add layers
    pooled_output = layers.Dropout(rate=config.hidden_dropout)(pooled_output)
    logits = OutputWeights(b_units=num_labels, h_units=config.hidden_size)(pooled_output)
    logits = OutputBias(b_units=num_labels)(logits)
    logits = tf.cast(logits, dtype=tf.float32)
    ### model construction
    model = Model(inputs=[inp_seq, inp_pos], outputs=logits)
    ### Load pre-trained weights of BERT module from MLM
    if bert_module_weights_path is not None:
        with open(bert_module_weights_path, "rb") as f:
            bert_module_weights = pickle.load(f)
        model.layers[2].set_weights(bert_module_weights)
        del bert_module_weights
        print("Pre-trained Weights of BERT Module Loaded")
    return model
