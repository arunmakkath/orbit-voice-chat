from transformers import Wav2Vec2Model, Wav2Vec2PreTrainedModel
import torch.nn as nn

class CSMModel(Wav2Vec2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.wav2vec2 = Wav2Vec2Model(config)
        self.projection = nn.Identity()  # Optional: replace with your projection layer
        self.post_init()

    def forward(self, input_values, attention_mask=None):
        outputs = self.wav2vec2(input_values, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        return torch.nn.utils.rnn.pad_sequence([self.projection(h) for h in hidden_states], batch_first=True)