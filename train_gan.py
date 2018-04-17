import argparse
import torch
import torch.nn as nn
import numpy as np
import os
import pickle
from data_loader_gan import get_loader
from build_vocab import Vocabulary
from torch.autograd import Variable 
from torch.nn.utils.rnn import *
from torchvision import transforms
from gan_model import Discriminator
from gan_model import Generator
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pdb
from tqdm import tqdm

"""
note: 
    vocab.idx2word[0] = <pad>
    vocab.idx2word[1] = <start>
    vocab.idx2word[2] = <end>
    vocab.idx2word[3] = <unk>
"""

def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)
    
def main(args):
    # Create model directory
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    
    if not os.path.exists(args.figure_path):
        os.makedirs(args.figure_path)
    
    # Image preprocessing
    # For normalization, see https://github.com/pytorch/vision#models
    transform = transforms.Compose([ 
        transforms.RandomCrop(args.crop_size),
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))])
    
    # Load vocabulary wrapper.
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    
    # Build data loader
    data_loader = get_loader(args.image_dir, args.caption_path, vocab, 
                             transform, args.batch_size,
                             shuffle=True, num_workers=args.num_workers) 

    # Build the models (Gen)
    generator = Generator(args.embed_size, args.hidden_size, len(vocab), args.num_layers)

    # Build the models (Disc)
    discriminator = Discriminator(args.embed_size, args.hidden_size, len(vocab), args.num_layers)
    
    if torch.cuda.is_available():
        generator.cuda()
        discriminator.cuda()

    # Loss and Optimizer (Gen)
    mle_criterion = nn.CrossEntropyLoss()
    params_gen = list(generator.parameters())
    optimizer_gen = torch.optim.Adam(params_gen)

    # Loss and Optimizer (Disc)
    params_disc = list(discriminator.parameters())
    optimizer_disc = torch.optim.Adam(params_disc)

    if args.pretraining:
        # Pre-training: train generator with MLE and discriminator with 3 losses (real + fake + wrong)
        total_steps = len(data_loader)
        disc_losses = []
        for epoch in tqdm(range(max([int(args.gen_pretrain_num_epochs), int(args.disc_pretrain_num_epochs)]))):
            for i, (images, captions, lengths, wrong_captions, wrong_lengths) in enumerate(data_loader):            
                
                images = to_var(images, volatile=True)
                captions = to_var(captions)
                wrong_captions = to_var(wrong_captions)
                targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]

                if epoch < int(args.gen_pretrain_num_epochs):
                    generator.zero_grad()
                    outputs, _ = generator(images, captions, lengths)
                    loss = mle_criterion(outputs, targets)
                    loss.backward()
                    optimizer_gen.step()

                if epoch < int(args.disc_pretrain_num_epochs):
                    discriminator.zero_grad()
                    rewards_real = discriminator(images, captions, lengths)
                    # rewards_fake = discriminator(images, sampled_captions, sampled_lengths) 
                    rewards_wrong = discriminator(images, wrong_captions, wrong_lengths)
                    real_loss = -torch.mean(torch.log(rewards_real))
                    # fake_loss = -torch.mean(torch.clamp(torch.log(1 - rewards_fake), min=-1000))
                    wrong_loss = -torch.mean(torch.clamp(torch.log(1 - rewards_wrong), min=-1000))
                    loss_disc = real_loss + wrong_loss # + fake_loss, no fake_loss because this is pretraining

                    disc_losses.append(loss_disc.cpu().data.numpy()[0])
                    loss_disc.backward()
                    optimizer_disc.step()
        
        # Save pretrained models
        torch.save(discriminator.state_dict(), os.path.join(args.model_path, 'pretrained-discriminator-%d.pkl' %int(args.disc_pretrain_num_epochs)))
        torch.save(generator.state_dict(), os.path.join(args.model_path, 'pretrained-generator-%d.pkl' %int(args.gen_pretrain_num_epochs)))

        # Plot pretraining figures
        plt.plot(disc_losses, label='pretraining_disc_loss')
        plt.savefig(args.figure_path + 'pretraining_disc_losses.png')
        plt.clf()

    # # Skip the rest for now
    # return

    # Train the Models
    total_step = len(data_loader)
    disc_losses = []
    for epoch in range(args.num_epochs):
        for i, (images, captions, lengths, wrong_captions, wrong_lengths) in enumerate(data_loader):

            # Set mini-batch dataset
            images = to_var(images, volatile=True)
            captions = to_var(captions)
            wrong_captions = to_var(wrong_captions)

            # Forward, Backward and Optimize
            # decoder.zero_grad()
            # encoder.zero_grad()
            # features = encoder(images)
            # outputs = decoder(features, captions, lengths)
            
            # sampled_captions = decoder.sample(features)
            # sampled_captions = torch.zeros_like(sampled_ids)
            # sampled_lengths = []

            # for row in range(sampled_captions.size(0)):
            #     for index, word_id in enumerate(sampled_captions[row,:]):
            #         # pdb.set_trace()
            #         word = vocab.idx2word[word_id.cpu().data.numpy()[0]]
            #         # sampled_captions[row, index].data = word
            #         if word == '<end>':
            #             sampled_lengths.append(index+1)
            #             break
            #         elif index == sampled_captions.size(1)-1:
            #             sampled_lengths.append(sampled_captions.size(1))
            #             break
            # sampled_lengths = np.array(sampled_lengths)
            # sampled_lengths[::-1].sort()
            # sampled_lengths = sampled_lengths.tolist()
            # loss = criterion(outputs, targets)
            # loss.backward()
            # optimizer.step()

            generator.zero_grad()
            outputs, packed_lengths = generator(images, captions, lengths)
            outputs = PackedSequence(outputs, packed_lengths)
            outputs = pad_packed_sequence(outputs, batch_first=True) # (b, T, V)

            Tmax = outputs[0].size(1)
            gen_samples = torch.zeros((args.batch_size, Tmax))
            
            # getting rewards from disc
            for t in range(2, Tmax):
                if t >= min(lengths): # TODO this makes things easier, but could min(lengths) could be too short
                    break

                # part 1: taken from real caption
                gen_samples[:,:t-1] = captions[:,:t-1].data
                for v in range(4, len(vocab)):
                    # part 2: taken from all possible vocabs
                    gen_samples[:,t] = v
                    # part 3: taken from rollouts
                    gen_samples[:,t:], sampled_lengths = generator.rollout(gen_samples, t)
                    rewards = discriminator(images, gen_samples, sampled_lengths)
            
            loss_gen = -outputs * rewards
            loss_gen.backward()
            optimizer_gen.step()

            # TODO get sampled_captions

            # Train discriminator
            discriminator.zero_grad()
            rewards_real = discriminator(images, captions, lengths)
            rewards_fake = discriminator(images, sampled_captions, sampled_lengths)
            rewards_wrong = discriminator(images, wrong_captions, wrong_lengths)
            real_loss = -torch.mean(torch.log(rewards_real))
            fake_loss = -torch.mean(torch.clamp(torch.log(1 - rewards_fake), min=-1000))
            wrong_loss = -torch.mean(torch.clamp(torch.log(1 - rewards_wrong), min=-1000))
            loss_disc = real_loss + fake_loss + wrong_loss

            disc_losses.append(loss_disc.cpu().data.numpy()[0])
            loss_disc.backward()
            optimizer_disc.step()

            # Print log info
            if i % args.log_step == 0:
                print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f, Perplexity: %5.4f'
                      %(epoch, args.num_epochs, i, total_step, 
                        loss.data[0], np.exp(loss.data[0]))) 
                
            # Save the models
            # if (i+1) % args.save_step == 0:
            if (i+1) % total_step == 0: # jm: saving at the last iteration instead
                torch.save(decoder.state_dict(), 
                           os.path.join(args.model_path, 
                                        'decoder-%d-%d.pkl' %(epoch+1, i+1)))
                torch.save(encoder.state_dict(), 
                           os.path.join(args.model_path, 
                                        'encoder-%d-%d.pkl' %(epoch+1, i+1)))
                torch.save(discriminator.state_dict(), 
                           os.path.join(args.model_path, 
                                        'discriminator-%d-%d.pkl' %(epoch+1, i+1)))

                # plot at the end of every epoch
                plt.plot(disc_losses, label='disc loss')
                plt.savefig(args.figure_path + 'disc_losses.png')
                plt.clf()
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--model_path', type=str, default='./models/' ,
    parser.add_argument('--model_path', type=str, default='./birds_gan_models/' ,
                        help='path for saving trained models')
    parser.add_argument('--crop_size', type=int, default=224 ,
                        help='size for randomly cropping images')
    # parser.add_argument('--vocab_path', type=str, default='./data/vocab.pkl',
    parser.add_argument('--vocab_path', type=str, default='./data/birds_vocab.pkl',
                        help='path for vocabulary wrapper')
    # parser.add_argument('--image_dir', type=str, default='./data/resized2014' ,
    parser.add_argument('--image_dir', type=str, default='./data/resized_CUB/' ,
                        help='directory for resized images')
    parser.add_argument('--caption_path', type=str,
                        # default='./data/annotations/captions_train2014.json',
                        default='./data/birds_captions/',
                        help='path for train annotation json file')
    parser.add_argument('--log_step', type=int , default=10,
                        help='step size for prining log info')
    # jm: changed code in main to save at the last iteration instead of taking manual inputs
    # parser.add_argument('--save_step', type=int , default=90,
    #                     help='step size for saving trained models')
    
    # Model parameters
    parser.add_argument('--embed_size', type=int , default=256 ,
                        help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int , default=512 ,
                        help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int , default=1 ,
                        help='number of layers in lstm')
    
    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    
    # Disc hyperparameters
    # jm: not mentioned in paper what they should be...
    parser.add_argument('--disc_alpha', type=float, default=0)
    parser.add_argument('--disc_beta', type=float, default=0.5)
    parser.add_argument('--gen_pretrain_num_epochs', type=int, default=20)
    parser.add_argument('--disc_pretrain_num_epochs', type=int, default=5)

    # dirs
    parser.add_argument('--figure_path', type=str, default='./figures/' ,
                        help='path for figures')

    # debuggin
    parser.add_argument('--pretraining', type=bool, default=False)
    args = parser.parse_args()
    print(args)
    main(args)
