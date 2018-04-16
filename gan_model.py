import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
from gan_encoder_decoder_model import EncoderCNN
from gan_encoder_decoder_model import DecoderRNN
import pdb

class Discriminator(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(Discriminator, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.image_feature_encoder = EncoderCNN(embed_size)
        self.sentence_feature_encoder = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.hidden_fine_tune_linear = nn.Linear(hidden_size, embed_size)

    def forward(self, images, captions, lengths):
        """Calculate reward score: r = logistic(dot_prod(f, h))"""
        # print(captions)
        features = self.image_feature_encoder(images) #(batch_size=128, embed_size=256)

        embeddings = self.embed(captions) # (batch_size, embed_size)
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True)
        hiddens, _ = self.sentence_feature_encoder(packed)

        padded = pad_packed_sequence(hiddens, batch_first=True)
        # padded[0] # (batch_size, T_max, hidden_size)
        last_padded_indices = [index-1 for index in padded[1]]
        hidden_outputs = padded[0][range(captions.size(0)), last_padded_indices, :]
        hidden_outputs = self.hidden_fine_tune_linear(hidden_outputs)
        
        dot_prod = torch.bmm(features.unsqueeze(1), hidden_outputs.unsqueeze(1).transpose(2,1)).squeeze()
        return nn.Sigmoid()(dot_prod)


class Generator(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(Generator, self).__init__()
        self.encoder = EncoderCNN(embed_size)
        self.decoder = DecoderRNN(embed_size, hidden_size, vocab_size, num_layers)

    def forward(self, images, captions, lengths):
        """Getting captions"""
        features = self.encoder(images)
        outputs = self.decoder(features, captions, lengths, noise=True) # (packed_size, vocab_size)
        outputs = pad_packed_sequence(outputs, batch_first=True) # (b, T, V)
        return outputs


