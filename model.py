import torch
import torch.nn as nn

from transformers import AutoConfig,AutoModel

class CustomModel(nn.Module):
    def __init__(self,model_name,num_output=1,pooling='clf'):
        super().__init__()
        self.pooling = pooling

        self.config = AutoConfig.from_pretrained(model_name)
        self.config.update(
            {
                "output_hidden_states": True,
                "add_pooling_layer": True,
                "num_labels": 1,
            }
        )
        self.transformer = AutoModel.from_pretrained(model_name, config=self.config)

        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.fc = nn.Linear(self.config.hidden_size, num_output)
        self._init_weights(self.fc)

        if self.pooling == 'attention':
            self.attention = nn.Sequential(
                nn.Linear(self.config.hidden_size, 512),
                nn.Tanh(),
                nn.Linear(512, 1),
                nn.Softmax(dim=1)
            )
            self._init_weights(self.attention)
        
    def feature(self, input_ids, attention_mask):
        outputs = self.transformer(input_ids,attention_mask)

        if self.pooling == 'attention':
            last_hidden_states = outputs.last_hidden_state
            weights = self.attention(last_hidden_states)
            feature = torch.sum(weights * last_hidden_states, dim=1)
            return feature
        elif self.pooling == 'clf':
            return outputs.pooler_output
        elif self.pooling == 'mean':
            return self.mean_pooling(outputs,attention_mask)        

    def forward(self, input_ids, attention_mask):
        feature = self.feature(input_ids,attention_mask)
        output = self.fc(self.dropout(feature))
        return output

    def mean_pooling(self,model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


if __name__ == "__main__":
    model = CustomModel(
        model_name = 'cardiffnlp/twitter-roberta-base-sentiment',
        pooling = 'mean'
    )

    input_ids = torch.ones((2,96),dtype=torch.long)
    attention_mask = torch.ones((2,96),dtype=torch.long)
    print(model(input_ids,attention_mask).shape)