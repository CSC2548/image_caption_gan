import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
from model import EncoderCNN

class discriminator(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(discriminator, self).__init__()
        # self.embed = nn.Embedding(vocab_size, embed_size)
        # self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)        
        self.image_feature_encoder = EncoderCNN(embed_size)
        self.sentence_feature_encoder = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.cnn_fine_tune_linear = nn.Linear(embed_size, hidden_size)

    def forward(self, images, captions, lengths):
        """Calculate reward score: r = logistic(dot_prod(f, h))"""
        features = self.image_feature_encoder(images) #(batch_size=128, embed_size=256)
        fine_tuned_features = self.cnn_fine_tune_linear(features)

        embeddings = self.embed(captions) # (batch_size, embed_size)
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True)
        hiddens, _ = self.lstm(packed)

        padded = pad_packed_sequence(hiddens, batch_first=True)
        # padded[0] # (batch_size, T_max, hidden_size)
        last_padded_indices = [index-1 for index in padded[1]]
        hidden_outputs = padded[0][range(128), last_padded_indices, :]
        
        dot_prod = torch.dot(fine_tuned_features, hidden_outputs)
        return nn.Sigmoid(dot_prod)


