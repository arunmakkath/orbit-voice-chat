from transformers import Wav2Vec2Processor

class AutoProcessor(Wav2Vec2Processor):
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        return super().from_pretrained("facebook/wav2vec2-base-960h", *args, **kwargs)