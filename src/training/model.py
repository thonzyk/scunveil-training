import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, LayerNormalization, Add, Layer
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import categorical_crossentropy


def c_xent(y_true, y_pred):
    losss = categorical_crossentropy(tf.cast(y_true, 'float32'), tf.cast(y_pred, 'float32'), from_logits=True)
    return tf.reduce_mean(losss)


class TiedOutput(Layer):
    def __init__(self, input_projection, **kwargs):
        super().__init__(**kwargs)
        self.input_projection = input_projection

    def call(self, inputs):
        return tf.matmul(inputs, self.input_projection.kernel, transpose_b=True)


class RNABagModel:
    def __init__(self, n_vars, n_layers, emb_dim, ff_dim=None):
        
        if ff_dim is None:
            ff_dim = emb_dim * 4

        self.n_vars = n_vars
        self.n_layers = n_layers
        self.emb_dim = emb_dim

        input_x = Input(shape=(n_vars,))

        input_projection = Dense(emb_dim, use_bias=False,
                                    kernel_initializer='glorot_uniform')
        
        x = input_projection(input_x)

        for layer_i in range(n_layers):
            x_add = LayerNormalization(epsilon=1e-6)(x)
            x_add = Dense(ff_dim, activation='relu')(x_add)
            x_add = Dense(emb_dim, kernel_initializer='zeros')(x_add)

            x = Add()([x, x_add])

        x = LayerNormalization(epsilon=1e-6, name='out_emb')(x)
        output_layer = TiedOutput(input_projection, name='out_logits')(x)

        self.model = Model(
            inputs=input_x, 
            outputs=output_layer
        )


    def compile_pretrain(self, lr=1e-4):
        optimizer = Adam(learning_rate=lr)

        self.model.compile(
            optimizer=optimizer,
            loss=c_xent,
        )

        print(f'Number of parameters: {self.model.count_params():,}')
