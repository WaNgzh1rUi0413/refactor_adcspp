from tensorflow import keras
class ToyFeatureOp(keras.layers.Layer):
    def __init__(self, cfg, **kwargs):
        super().__init__()
        self.num_sparse = cfg.num_sparse
        self.features_op = {
            f'sparse_{i}': keras.layers.Embedding(
                1000,
                cfg.output_dim
            )
            for i in range(self.num_sparse)
        }

    def call(self, inputs, **kwargs):
        outputs = {}
        for i in range(self.num_sparse):
            key = f'sparse_{i}'
            outputs[key] = self.features_op[key](inputs[key])
        return outputs