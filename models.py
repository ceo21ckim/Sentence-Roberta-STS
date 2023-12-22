from torch import nn 
import torch 
from transformers import AutoModelForSequenceClassification, AutoModel

class Model(nn.Module):
    def __init__(self, model_name='jhgan/ko-sroberta-sts', num_labels=1):
        super(Model, self).__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        self.num_labels = num_labels
        
        self._init_weight()
    
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.uniform_(m.bias)
    
    def forward(self, input_ids, attention_mask):
        pooler = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return pooler['logits']
    
    

class Model(nn.Module):
    def __init__(self, model_name, in_dim=768, num_labels=1, dr_rate=0.1):
        super(Model, self).__init__()
        self.model = AutoModel.from_pretrained(model_name, num_labels=num_labels)
        self.num_labels = num_labels
        self.dr_rate = dr_rate
        self.classifier = nn.Sequential(
            self.block(in_dim, in_dim//2), 
            self.block(in_dim//2, in_dim//4), 
            self.block(in_dim//4, num_labels)
        )
        
        self._init_weight()
    
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.uniform_(m.bias)
    
    
    def block(self, in_dim, out_dim):
        outs = nn.Sequential(
            nn.Linear(in_dim, out_dim), 
            nn.Dropout(self.dr_rate), 
            nn.ReLU()
        )
        return outs 
    
    def forward(self, input_ids, attention_mask):
        pooler = self.model(input_ids=input_ids, attention_mask=attention_mask)
        outs = self.classifier(pooler['pooler_output'])
        return outs
