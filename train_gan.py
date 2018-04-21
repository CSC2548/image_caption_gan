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
        gen_losses = []
        print('pre-training')
        for epoch in tqdm(range(max([int(args.gen_pretrain_num_epochs), int(args.disc_pretrain_num_epochs)]))):
        # for epoch in range(max([int(args.gen_pretrain_num_epochs), int(args.disc_pretrain_num_epochs)])):
            for i, (images, captions, lengths, wrong_captions, wrong_lengths) in enumerate(data_loader):            
                
                images = to_var(images, volatile=False)
                captions = to_var(captions)
                wrong_captions = to_var(wrong_captions)
                targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]

                if epoch < int(args.gen_pretrain_num_epochs):
                    generator.zero_grad()
                    outputs, _ = generator(images, captions, lengths)
                    loss = mle_criterion(outputs, targets)
                    gen_losses.append(loss.cpu().data.numpy()[0])
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

        plt.plot(losses, label='pretraining_gen_loss')
        plt.savefig(args.figure_path + 'pretraining_gen_losses.png')
        plt.clf()
        
    else:
        generator.load_state_dict(torch.load(args.pretrained_gen_path))
        discriminator.load_state_dict(torch.load(args.pretrained_disc_path))

    # # Skip the rest for now
    # return

    # Train the Models
    total_step = len(data_loader)
    disc_gan_losses = []
    gen_gan_losses= []
    for epoch in range(args.num_epochs):
        for i, (images, captions, lengths, wrong_captions, wrong_lengths) in enumerate(data_loader):

            # Set mini-batch dataset
            images = to_var(images, volatile=True)
            captions = to_var(captions)
            wrong_captions = to_var(wrong_captions)

            generator.zero_grad()
            outputs, packed_lengths = generator(images, captions, lengths)
            outputs = PackedSequence(outputs, packed_lengths)
            outputs = pad_packed_sequence(outputs, batch_first=True) # (b, T, V)

            Tmax = outputs[0].size(1)
            if torch.cuda.is_available():
                rewards = torch.zeros_like(outputs[0]).type(torch.cuda.FloatTensor)
            else:
                rewards = torch.zeros_like(outputs[0]).type(torch.FloatTensor)

            # getting rewards from disc
	    # for t in tqdm(range(2, Tmax, 4)):
            for t in range(2, Tmax, 2):
            # for t in range(2, 4):
                if t >= min(lengths): # TODO this makes things easier, but could min(lengths) could be too short
                    break

                gen_samples = to_var(torch.zeros((captions.size(0), Tmax)).type(torch.FloatTensor), volatile=True)
                # part 1: taken from real caption
                gen_samples[:,:t] = captions[:,:t].data

                predicted_ids, saved_states = generator.pre_compute(gen_samples, t)
                for v in range(predicted_ids.size(1)):
                    # part 2: taken from all possible vocabs
                    gen_samples[:,t] = predicted_ids[:,v]
                    # part 3: taken from rollouts
                    gen_samples[:,t:] = generator.rollout(gen_samples, t, saved_states)
                    
                    sampled_lengths = []
                    # finding sampled_lengths
                    for batch in range(int(captions.size(0))):
                        for b_t in range(Tmax):
                            if gen_samples[batch, b_t].cpu().data.numpy()[0] == 2: # <end>
                                sampled_lengths.append(b_t+1)
                                break
                            elif b_t == Tmax-1:
                                sampled_lengths.append(Tmax)

                    # sort sampled_lengths
                    sampled_lengths = np.array(sampled_lengths)
                    sampled_lengths[::-1].sort()
                    sampled_lengths = sampled_lengths.tolist()
                    
                    # get rewards from disc
                    rewards[:,t,v] = discriminator(images, gen_samples.detach(), sampled_lengths)

                # for v in tqdm(range(4, len(vocab))):
                # # for v in range(4, 5):
                #     # part 2: taken from all possible vocabs
                #     gen_samples[:,t] = v
                #     # part 3: taken from rollouts
                #     gen_samples[:,t+1:] = generator.rollout(gen_samples, t)

                #     sampled_lengths = []
                #     # finding sampled_lengths
                #     for batch in range(int(args.batch_size)):
                #         for b_t in range(Tmax):
                #             if gen_samples[batch, b_t].cpu().data.numpy()[0] == 2: # <end>
                #                 sampled_lengths.append(b_t+1)
                #                 break
                #             elif b_t == Tmax-1:
                #                 sampled_lengths.append(Tmax)

                #     # sort sampled_lengths
                #     sampled_lengths = np.array(sampled_lengths)
                #     sampled_lengths[::-1].sort()
                #     sampled_lengths = sampled_lengths.tolist()
                    
                #     # get rewards from disc
                #     rewards[:,t,v] = discriminator(images, gen_samples.detach(), sampled_lengths)

            # rewards = rewards.detach()
            # pdb.set_trace()
            rewards_detached = rewards.data
            rewards_detached = to_var(rewards_detached)
            loss_gen = torch.dot(outputs[0], -rewards_detached)
            gen_gan_losses.append(loss_gen.cpu().data.numpy()[0])
            loss_gen.backward()
            optimizer_gen.step()

            # TODO get sampled_captions
            sampled_ids = generator.sample(images)
            # sampled_captions = torch.zeros_like(sampled_ids).type(torch.LongTensor)
            sampled_lengths = []
            # finding sampled_lengths
            for batch in range(int(captions.size(0))):
                for b_t in range(20):
                    #pdb.set_trace()
                    #sampled_captions[batch, b_t].data = sampled_ids[batch, b_t].cpu().data.numpy()[0]
                    if sampled_ids[batch, b_t].cpu().data.numpy()[0] == 2: # <end>
                        sampled_lengths.append(b_t+1)
                        break
                    elif b_t == 20-1:
                        sampled_lengths.append(20)
            # sort sampled_lengths
            sampled_lengths = np.array(sampled_lengths)
            sampled_lengths[::-1].sort()
            sampled_lengths = sampled_lengths.tolist()

            # Train discriminator
            discriminator.zero_grad()
            images.volatile = False
            captions.volatile = False
            wrong_captions.volatile = False
            rewards_real = discriminator(images, captions, lengths)
            rewards_fake = discriminator(images, sampled_ids, sampled_lengths)
            rewards_wrong = discriminator(images, wrong_captions, wrong_lengths)
            real_loss = -torch.mean(torch.log(rewards_real))
            fake_loss = -torch.mean(torch.clamp(torch.log(1 - rewards_fake), min=-1000))
            wrong_loss = -torch.mean(torch.clamp(torch.log(1 - rewards_wrong), min=-1000))
            loss_disc = real_loss + fake_loss + wrong_loss

            disc_gan_losses.append(loss_disc.cpu().data.numpy()[0])
            loss_disc.backward()
            optimizer_disc.step()

            # Print log info
            if epoch % 2 == 0 and i % args.log_step == 0:
                print('Epoch [%d/%d], Step [%d/%d], Disc Loss: %.4f, Gen Loss: %.4f'
                      %(epoch, args.num_epochs, i, total_step, 
                        loss_disc.data[0], loss_gen.data[0])) 
                
            # Save the models
            # if (i+1) % args.save_step == 0:
            if (i+1) % total_step == 0: # jm: saving at the last iteration instead
                torch.save(generator.state_dict(), 
                           os.path.join(args.model_path, 
                                        'generator-gan-%d-%d.pkl' %(epoch+1, i+1)))
                torch.save(discriminator.state_dict(), 
                           os.path.join(args.model_path, 
                                        'discriminator-gan-%d-%d.pkl' %(epoch+1, i+1)))

                # plot at the end of every epoch
                plt.plot(disc_gan_losses, label='disc gan loss')
                plt.savefig(args.figure_path + 'disc_gan_losses.png')
                plt.clf()

                plt.plot(gen_gan_losses, label='gen gan loss')
                plt.savefig(args.figure_path + 'gen_gan_losses.png')

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
    
    parser.add_argument('--num_epochs', type=int, default=10)
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
    parser.add_argument('--pretraining', type=bool, default=True)
    parser.add_argument('--pretrained_gen_path', type=str, default='./birds_gan_models/pretrained-generator-20.pkl')
    parser.add_argument('--pretrained_disc_path', type=str, default='./birds_gan_models/pretrained-discriminator-5.pkl')
    args = parser.parse_args()
    print(args)
    main(args)
