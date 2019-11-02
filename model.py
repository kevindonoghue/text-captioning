import numpy as np
import torch
import torch.nn as nn
import json
import matplotlib.pyplot as plt
import time



tokens = list('abcdefghijklmnopqrstuvwxyz') + ['<START>', '<END>', ' ']
ch_to_ix = {ch: i for i, ch in enumerate(tokens)}
ix_to_ch = {i: ch for i, ch in enumerate(tokens)}


if torch.cuda.is_available():
    FloatTensor = torch.cuda.FloatTensor
    LongTensor = torch.cuda.LongTensor
else:
    FloatTensor = torch.Tensor
    LongTensor = torch.LongTensor


embedding_size = 15
hidden_size = 10
num_tokens = len(tokens)
num_layers = 2
batch_size = 32
cnn_output_depth = 15


class ImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(nn.Conv2d(1, 10, 3, stride=(2, 5), padding=1), # (20, 135) --> (10, 27)
                                 nn.BatchNorm2d(10),
                                 nn.ReLU(),
                                 nn.Conv2d(10, 20, 3, stride=(2, 1), padding=1), # (10, 27) --> (5, 27)
                                 nn.BatchNorm2d(20),
                                 nn.ReLU(),
                                 nn.Conv2d(20, 30, 3, stride=1, padding=1), # (5, 27) --> (5, 27)
                                 nn.BatchNorm2d(30),
                                 nn.ReLU(),
                                 nn.Conv2d(30, 40, 3, stride=1, padding=1), # (5, 27) --> (5, 27)
                                 nn.BatchNorm2d(40),
                                 nn.ReLU(),
                                 nn.Conv2d(40, cnn_output_depth, 3, stride=(5, 1), padding=1), # (5, 27) --> (1, 27)
        )
        
    def forward(self, x):
        return self.seq(x)
    
    
class EncodedImageToHidden(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = nn.Sequential(nn.Linear(27*cnn_output_depth, 25),
                                 nn.BatchNorm1d(25),
                                 nn.ReLU(),
                                 nn.Linear(25, hidden_size))
        
    def forward(self, x):
        x = x.view(-1, 27*cnn_output_depth)
        x = self.seq(x)
        return x
    
    
        
    
class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(num_tokens, embedding_size)
        self.gru = nn.GRU(input_size=cnn_output_depth+embedding_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.affine = nn.Linear(hidden_size, num_tokens)
        
    def forward(self, token, context, hidden):
        embedded_token = self.embed(token)
        gru_input_vector = torch.cat([context, embedded_token], -1)
        gru_output_vector, hidden = self.gru(gru_input_vector)
        gru_output_vector = self.affine(gru_output_vector)
        return gru_output_vector, hidden
    

class AttentionScoreNet(nn.Module):
    def __init__(self):
        super().__init__()
        # want to combine the encoded image input, of shape (batch_size, 1, 27, cnn_output_depth)
        # with the hidden layer, of shape (batch_size, num_layers, hidden_size)
        # at the end, return a tensor of shape (batch_size, 27)

        self.W1 = nn.Linear(cnn_output_depth, 10)
        self.W2 = nn.Linear(num_layers*hidden_size, 10)
        self.seq = nn.Sequential(nn.Linear(10, 10),
                                 nn.ReLU(),
                                 nn.Linear(10, 10),
                                 nn.ReLU())
        self.V = nn.Linear(10, 1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, enc_image_inp, hidden):
        enc_image_inp = enc_image_inp.view(batch_size, 27, cnn_output_depth)
        hidden = hidden.view(batch_size, 1, num_layers*hidden_size)
        unnorm_scores = self.V(self.seq(self.W1(enc_image_inp) + self.W2(hidden)))
        unnorm_scores = unnorm_scores.view(batch_size, 27)
        norm_scores = self.softmax(unnorm_scores)

        return norm_scores

        

       

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_encoder = ImageEncoder()
        self.encoded_image_to_hidden = EncodedImageToHidden()
        self.decoder = Decoder()
        self.attention_score_net = AttentionScoreNet()
        
    def fit(self, images, captions, num_iterations, optimizer, loss_fn):
        losses = []
        start_time = time.time()
        for i in range(num_iterations):
            batch_indices = np.random.choice(images.shape[0], size=batch_size)
            image_batch = FloatTensor(images[batch_indices]).view(batch_size, 1, 20, 135) # batch size, channel depth, height, width
            caption_batch = []
            for index in batch_indices:
                caption_batch.append(captions[index])
            caption_batch = LongTensor(np.array(caption_batch)).view(batch_size, 1, -1) # batch size, seq len, dimension of caption
            
            encoded_image = self.image_encoder(image_batch)
            hidden = self.encoded_image_to_hidden(encoded_image)
            context = encoded_image.view(batch_size, cnn_output_depth, 27).mean(2).view(batch_size, 1, cnn_output_depth)
            
            
            
            
            output_sequence = []
            attention_score_archive = []
            for j in range(caption_batch.shape[2]-1):
                gru_out, hidden = self.decoder(caption_batch[:, :, j], context, hidden)
                loss = loss_fn(gru_out.view(batch_size, num_tokens), caption_batch[:, :, j+1].view(-1))
                loss.backward(retain_graph=True)
                losses.append(loss.item())
                optimizer.step()
                optimizer.zero_grad()
                
                attention_scores = self.attention_score_net(encoded_image, hidden)
                attention_score_archive.append(attention_scores[0].detach().cpu().numpy().reshape(27))
                with torch.no_grad():
                    output_sequence.append(ix_to_ch[gru_out.view(batch_size, num_tokens).detach().cpu().numpy().argmax(axis=1)[0]])
                context = (attention_scores.view(batch_size, 1, 1, 27)*encoded_image).view(batch_size, cnn_output_depth, 27).sum(2).view(batch_size, 1, cnn_output_depth)

            if i % 100 == 0:
                print(f'iteration: {i}, time elapsed: {time.time() - start_time}')
                print('loss: ', np.mean(losses[-100:]))
                print('pred   : ' + ''.join(output_sequence))
                print('correct: ' + ''.join([ix_to_ch[k] for k in caption_batch[0, 0].detach().cpu().numpy()]))
                attention_scores = attention_scores.detach().cpu().numpy().reshape(-1, 27)
                plt.imshow(np.array(attention_score_archive))
                plt.show()
                print('')


# model = Net()
# if torch.cuda.is_available():
#     model.cuda()
    
# optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
# loss_fn = nn.CrossEntropyLoss()

# images = np.load('images.npy')
# with open('captions.json') as f:
#     captions = json.load(f)
    
# model.fit(images, captions, 100000, optimizer, loss_fn)