import torch
import transformers
import config
from torch import nn

class roberta_model(nn.Module):
    
    def __init__(self, n_classes):
        super(roberta_model, self).__init__()
        self.roberta = transformers.XLMRobertaForSequenceClassification.from_pretrained(config.ROBERTA_PATH, num_labels = 2)
#         config = transformers.XLMRobertaConfig.from_pretrained("xlm-roberta-base")
#         self.roberta = transformers.XLMRobertaForSequenceClassification.from_pretrained("best_model_finetuned_lm.pt",config=config)
        self.drop = nn.Dropout(p = 0.3)
        self.out = nn.Linear(2,2)
        
        
    def forward(self, input_ids, attention_mask, labels = None): 
        if labels != None:
            _, pooled_output = self.roberta(
                **dict(input_ids = input_ids,attention_mask = attention_mask,labels = labels),return_dict=False)
            output = self.drop(pooled_output)
            return self.out(output)
        else:
            outputs = self.roberta(
                **dict(input_ids = input_ids,attention_mask = attention_mask),return_dict=False)
            return outputs[0]
        