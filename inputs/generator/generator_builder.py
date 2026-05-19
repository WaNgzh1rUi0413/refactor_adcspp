from utils.easydict import EasyDict
from .tfrecord_generator import TFRecordGenerate

mapping = {
    "TFRecordGenerate": TFRecordGenerate
}


class GeneratorBuilder(object):
    @classmethod
    def build(cls, cfg: EasyDict, data_size=1000, files_per_split: int = 5, date: str = "",
              generate_type: str = "TFRecordGenerate"):
        if generate_type not in mapping:
            raise ValueError(f'Unknown reader: {generate_type}')
        return mapping.get(generate_type)(cfg, data_size=data_size, files_per_split=files_per_split, date=date)
