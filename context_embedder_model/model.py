import torch
import torch.nn as nn

from transformers import AutoConfig,AutoModel

class USSModel(nn.Module):
    def __init__(self,model_name,num_output=1,pooling='clf'):
        super().__init__()
        self.pooling = pooling

        self.config = AutoConfig.from_pretrained(model_name)
        self.config.update(
            {
                "hidden_dropout_prob":0.0,
                "output_hidden_states": False,
                "add_pooling_layer": True,
                "num_labels": 1,
            }
        )

        self.transformer = AutoModel.from_pretrained(model_name, config=self.config)

        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.fc = nn.Linear(self.config.hidden_size*2, num_output)

        if self.pooling == 'attention':
            self.attention = nn.Sequential(
                nn.Linear(self.config.hidden_size*2, 512),
                nn.Tanh(),
                nn.Linear(512, 1),
                nn.Softmax(dim=1)
            )
            self._init_weights(self.attention)
        
    def feature(self, input_encoding, context_encoding):
        out_1 = self.transformer(input_encoding['input_ids'],input_encoding['attention_mask']).last_hidden_state
        out_2 = self.transformer(context_encoding['input_ids'],context_encoding['attention_mask']).last_hidden_state

        out = torch.cat([out_1,out_2],axis=-1)
        #out_2 = self.mean_pooling(out_2,context_encoding['attention_mask'])  ## bs,768

        if self.pooling == 'attention':
            #last_hidden_states = out.last_hidden_state ## bs,seq_len,768
            weights = self.attention(out)
            feature = torch.sum(weights * out, dim=1)
            return feature
        else:
            return out_1.pooler_output    

    def forward(self, input_encoding, context_encoding):
        feature = self.feature(input_encoding,context_encoding)
        output = self.fc(self.dropout(feature))
        return output

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

    def mean_pooling(self,model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
