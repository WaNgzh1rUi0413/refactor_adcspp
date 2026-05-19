class DatasetBuilder(object):
    @classmethod
    def build(cls, cfg):
        reader_type = cfg.type
        if reader_type == "FlowTFRecordDataset":
            from .flow_tfrecord_dataset import FlowTFRecordDataset

            return FlowTFRecordDataset(cfg)
        if reader_type == "TFRecordDataset":
            from .tfrecord_dataset import TFRecordDataset

            return TFRecordDataset(cfg)
        if reader_type == "ToyDataset":
            from .toy_dataset import ToyDataset

            return ToyDataset(cfg)
        raise ValueError(f"Unknown reader: `{reader_type}`")