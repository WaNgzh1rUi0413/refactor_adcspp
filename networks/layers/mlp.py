import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, BatchNormalization, Dropout, Concatenate, Lambda


class MLP(Layer):
    def __init__(
        self,
        hidden_units,
        activation='relu',
        output_activation=None,
        l2_reg=0,
        dropout_rate=0,
        use_bn=False,
        name=None,
        gate_units=None,
        **kwargs
    ):
        super(MLP, self).__init__(name=name)
        if isinstance(hidden_units, int):
            hidden_units = [hidden_units]
        if isinstance(activation, str):
            activation = [activation] * len(hidden_units)
        if output_activation is not None:
            activation[-1] = output_activation
        if isinstance(dropout_rate, (int, float)):
            dropout_rate = [dropout_rate] * len(hidden_units)
        if isinstance(l2_reg, (int, float)):
            l2_reg = [l2_reg] * len(hidden_units)
        if len(set([len(hidden_units), len(activation), len(dropout_rate), len(l2_reg)])) != 1:
            raise ValueError('hidden_units, activation, dropout_rate and l2_reg should have the same length')

        self.hidden_units = hidden_units
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg
        self.use_bn = use_bn

        # 初始化Dense层
        self.dense_layers = []
        for i, units in enumerate(hidden_units):
            reg = tf.keras.regularizers.l2(self.l2_reg[i]) if self.l2_reg[i] > 0 else None
            self.dense_layers.append(
                Dense(
                    units,
                    activation=None,
                    kernel_regularizer=reg,
                    name=f'{self.name}_dense_{i}',
                    **kwargs
                )
            )

        # 初始化BatchNormalization层
        if self.use_bn:
            self.bn_layers = [BatchNormalization(name=f'{self.name}_bn_{i}') for i in range(len(hidden_units))]
        else:
            self.bn_layers = [None] * len(hidden_units)

        # 初始化Dropout层
        self.dropout_layers = []
        for i, rate in enumerate(dropout_rate):
            layer = Dropout(rate, name=f'{self.name}_dropout_{i}') if rate > 0 else None
            self.dropout_layers.append(layer)

        # 初始化激活函数
        self.activation_fns = [tf.keras.activations.get(activation[i]) for i in range(len(hidden_units))]

        # 初始化门控MLP
        self.gate_mlps = [None] * len(hidden_units)
        if gate_units:
            if isinstance(gate_units, int):
                gate_units = [gate_units] * len(hidden_units)
            for i, (gate_unit, units) in enumerate(zip(gate_units, hidden_units)):
                if not gate_unit:
                    continue
                self.gate_mlps[i] = (
                    tf.keras.Sequential([
                        Dense(gate_unit, activation='relu', name=f'{self.name}_gate_{i}_dense_1'),
                        Dense(units, activation='sigmoid', name=f'{self.name}_gate_{i}_dense_2'),
                        Lambda(lambda x: x * 2.0, name=f'{self.name}_gate_{i}_scale')
                    ])
                )

    def call(self, inputs, training=None, **kwargs):
        if isinstance(inputs, (list, tuple)):
            if self.gate_mlps[0]:
                x = Concatenate(axis=-1)(inputs[:-1])
                gate_inputs = inputs[-1]
            else:
                x = Concatenate(axis=-1)(inputs)
        elif isinstance(inputs, dict):
            if self.gate_mlps[0]:
                gate_inputs = inputs.get('gate')
                tensors = [inputs[key] for key in sorted(inputs.keys()) if key != 'gate']
            else:
                tensors = [inputs[key] for key in sorted(inputs.keys())]
            x = Concatenate(axis=-1)(tensors)
        else:
            x = inputs

        for dense, bn, activation, dropout, gate_mlp in zip(
            self.dense_layers,
            self.bn_layers,
            self.activation_fns,
            self.dropout_layers,
            self.gate_mlps
        ):
            x = dense(x)
            if bn:
                x = bn(x, training=training)
            x = activation(x)
            if dropout:
                x = dropout(x, training=training)

            if gate_mlp:
                gate = gate_mlp(gate_inputs)
                x = gate * x

        return x

    def get_config(self):
        config = super(MLP, self).get_config()
        config.update({
            'hidden_units': self.hidden_units,
            'activation': self.activation,
            'output_activation': self.activation[-1],
            'l2_reg': self.l2_reg,
            'dropout_rate': self.dropout_rate,
            'use_bn': self.use_bn,
            'name': self.name
        })
        return config


class MultiMLPTower(Layer):
    def __init__(
        self,
        num_towers,
        hidden_units,
        activation='relu',
        output_activation=None,
        l2_reg=0,
        dropout_rate=0,
        use_bn=False,
        name=None,
        **kwargs
    ):
        super(MultiMLPTower, self).__init__(name=name)
        self.num_towers = num_towers
        self.mlp_layers = [
            MLP(
                hidden_units,
                activation,
                output_activation,
                l2_reg,
                dropout_rate,
                use_bn,
                name=f'{self.name}_mlp_{i}',
                **kwargs
            )
            for i in range(num_towers)
        ]

    def call(self, inputs, training=None, **kwargs):
        x = [self.mlp_layers[i](inputs[i], training=training) for i in range(self.num_towers)]
        # 返回一个长为num_towers的list
        return x

    def get_config(self):
        config = super(MultiMLPTower, self).get_config()
        config.update({
            'num_towers': self.num_towers,
            'hidden_units': self.mlp_layers[0].hidden_units,
            'activation': self.mlp_layers[0].activation,
            'output_activation': self.mlp_layers[0].activation[-1],
            'l2_reg': self.mlp_layers[0].l2_reg,
            'dropout_rate': self.mlp_layers[0].dropout_rate,
            'use_bn': self.mlp_layers[0].use_bn,
            'name': self.name
        })
        return config
