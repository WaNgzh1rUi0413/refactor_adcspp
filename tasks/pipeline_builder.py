from utils.easydict import EasyDict
from .pipeline import AdsRecPipeline


mapping = {
    "AdsRecPipeline": AdsRecPipeline,
    "FlowGenerationPipeline": AdsRecPipeline,
}


class PipelineBuilder(object):
    @classmethod
    def build(cls, cfg: EasyDict):
        pipeline_type = cfg.run.get("pipeline_type", "AdsRecPipeline")
        if pipeline_type not in mapping:
            raise ValueError(f"Unknown pipeline type: {pipeline_type}")
        return mapping.get(pipeline_type)(cfg, cfg.run.steps)
