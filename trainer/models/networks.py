import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_hidden_dim, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, ff_hidden_dim),
            nn.ReLU(),
            nn.Linear(ff_hidden_dim, embed_dim)
        )
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Attention block
        attn_output, _ = self.attention(x, x, x)
        x = self.layer_norm1(x + self.dropout(attn_output))
        
        # Feed forward block
        ff_output = self.feed_forward(x)
        x = self.layer_norm2(x + self.dropout(ff_output))
        
        return x

class ClassificationTransformer(nn.Module):
    def __init__(self, num_classes=1, embed_dim=256, num_heads=8, ff_hidden_dim=512, num_layers=1, dropout=0.4):
        super(ClassificationTransformer, self).__init__()
        self.transformers = [TransformerBlock(embed_dim=embed_dim, hidden_dim=ff_hidden_dim) for _ in num_layers]
        self.project = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        pass


class MLP_Classifier(nn.Module):
    def __init__(self, input_dim, num_layers=4, use_distance=False):
        super(MLP_Classifier, self).__init__()
        self.layers = []
        if num_layers > 0:
            if use_distance:
                delta = 1
            else:
                delta = 0
            input_dim = input_dim + delta
            self.layers.append(nn.Linear(input_dim, input_dim//2))
        for idx in range(1, num_layers):
            self.layers.append(nn.Linear(int(input_dim/(2**idx)), int(input_dim/(2**(idx+1)))))
        #self.layer1 = nn.Linear(input_dim, 512)
        #self.layer2 = nn.Linear(512, 256)
        #self.layer3 = nn.Linear(256, 128)
        #self.layer4 = nn.Linear(128, 64)
        self.layers = nn.Sequential(*self.layers)
        self.output = nn.Linear(input_dim//(2**num_layers), 1)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        #x = self.dropout(x)
        for lidx, layer in enumerate(self.layers):
            x = torch.relu(self.layers[lidx](x))
            x = self.dropout(x)

        x = self.output(x)
        x = self.sigmoid(x)
        return x


class old_MLP_Classifier(nn.Module):
    def __init__(self, input_dim, num_layers=4):
        super(old_MLP_Classifier, self).__init__()
        self.layer1 = nn.Linear(input_dim, 512)
        self.layer2 = nn.Linear(512, 256)
        self.layer3 = nn.Linear(256, 128)
        self.layer4 = nn.Linear(128, 64)
        self.output = nn.Linear(input_dim//(2**num_layers), 1)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.dropout(x)
        x = torch.relu(self.layer1(x))
        x = self.dropout(x)
        x = torch.relu(self.layer2(x))
        x = self.dropout(x)
        x = torch.relu(self.layer3(x))
        x = self.dropout(x)
        x = torch.relu(self.layer4(x))
        x = self.dropout(x)
        x = self.output(x)
        x = self.sigmoid(x)
        return x




class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(MLP, self).__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.output_layer = nn.Linear(hidden_dim, input_dim)
        self.initialize_weights()

    def initialize_weights(self):
        with torch.no_grad():
            nn.init.eye_(self.input_layer.weight)
            nn.init.constant_(self.input_layer.bias, 0)
            
            nn.init.eye_(self.output_layer.weight)
            nn.init.constant_(self.output_layer.bias, 0)    
    
    def forward(self, x):
        x = self.input_layer(x)
        x = self.relu(x)
        x = self.output_layer(x)
        return x

class CNNHead(nn.Module):
    def __init__(self, in_channels, num_classes=1):
        super(CNNHead, self).__init__()
        # Additional convolutional layers
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=512, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(512)
        self.conv2 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(1024)

        # Global average pooling layer
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.sigmoid = nn.Sigmoid()
        # Fully connected layer for classification
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        #x = self.bn1(x)
        x = torch.relu(x)

        x = self.conv2(x)
        #x = self.bn2(x)
        x = torch.relu(x)

        x = self.global_pool(x)
        x = torch.flatten(x, 1)

        x = self.fc(x)
        x = self.sigmoid(x)
        return x



class SelfAttentionBinaryClassification(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, output_dim):
        super(SelfAttentionBinaryClassification, self).__init__()
        
        self.self_attention = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads)
        self.fc1 = MLP_Classifier(input_dim=input_dim, num_layers=1)
        #self.fc1 = nn.Linear(input_dim, output_dim)
        #self.relu = nn.ReLU()
        #self.fc2 = nn.Linear(hidden_dim, output_dim)
        #self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # Assuming input shape is (batch_size, seq_length, input_dim)
        #x = x.permute(1, 0, 2)  # Permute to (seq_length, batch_size, input_dim)
        attn_output, _ = self.self_attention(x, x, x)
        attn_output = attn_output.mean(dim=0)  # Mean pooling over the sequence length
        out = self.fc1(attn_output)
        #out = self.relu(out)
        #out = self.fc2(out)
        #out = self.sigmoid(out)
        return out
