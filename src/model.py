import torch
import torch.nn as nn
import torch.nn.functional as F

from src.backbones import InceptionV3
from src.attention import Linear_global, SelfAttention
from src.conformer import Conformer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FG_SBIR(nn.Module):
    def __init__(self, args):
        super(FG_SBIR, self).__init__()
        self.args = args
        self.sample_embedding_network = InceptionV3(args=args).to(device)
        self.attention = Conformer(dim=2048).to(device)
        self.linear = Linear_global(feature_num=self.args.output_size).to(device)
        
        self.sketch_embedding_network = InceptionV3(args=args).to(device)
        self.sketch_attention = Conformer(dim=2048).to(device)
        self.sketch_linear = Linear_global(feature_num=self.args.output_size).to(device)
    
        def init_weights(m):
            if type(m) == nn.Linear or type(m) == nn.Conv1d or type(m) == nn.Parameter:
                nn.init.kaiming_normal_(m.weight)
                
        if self.args.use_kaiming_init:
            self.attention.apply(init_weights)
            self.linear.apply(init_weights)
            
            self.sketch_attention.apply(init_weights)
            self.sketch_linear.apply(init_weights)
            
    def forward(self, batch):        
        sketch_img = batch['sketch_img'].to(device)
        positive_img = batch['positive_img'].to(device)
        negative_img = batch['negative_img'].to(device)
            
        positive_feature = self.sample_embedding_network(positive_img)
        negative_feature = self.sample_embedding_network(negative_img)
        sketch_feature = self.sketch_embedding_network(sketch_img)
        
        positive_feature = self.attention(positive_feature)
        negative_feature = self.attention(negative_feature)
        sketch_feature = self.sketch_attention(sketch_feature)
        
        positive_feature = self.linear(positive_feature)
        negative_feature = self.linear(negative_feature)
        sketch_feature = self.sketch_linear(sketch_feature)
                
        return positive_feature, negative_feature, sketch_feature
    
    def test_forward(self, batch):        
        sketch_img = batch['sketch_img'].to(device)
        positive_img = batch['positive_img'].to(device)
            
        positive_feature = self.sample_embedding_network(positive_img)
        positive_feature = self.attention(positive_feature)
        positive_feature = self.linear(positive_feature)
        
        sketch_feature = self.sketch_embedding_network(sketch_img)
        sketch_feature = self.sketch_attention(sketch_feature)
        sketch_feature = self.sketch_linear(sketch_feature)
        
        return sketch_feature, positive_feature
    
    