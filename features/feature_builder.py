from utils.easydict import EasyDict
from .feature_process import FeatureProcessLayer
from .toy_feature_op import ToyFeatureOp

mapping = {
    "ToyFeatureOp": ToyFeatureOp,
    "FeatureProcessLayer": FeatureProcessLayer
}

class FeatureBuilder(object):
    @classmethod
    def build(cls, cfg: EasyDict):
        """
        :param cfg:
        :return:
        """
        feature_process_type = cfg.get('type')
        if feature_process_type not in mapping:
            raise ValueError(f'Unknown feature processor: `{feature_process_type}`')
        return mapping.get(feature_process_type)(cfg)