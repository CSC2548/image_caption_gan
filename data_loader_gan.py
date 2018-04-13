import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import os
import pickle
import numpy as np
import nltk
from PIL import Image
from build_vocab import Vocabulary
import pdb
import pandas as pd

class CUBDataset(data.Dataset):
    """CUB Custom Dataset compatible with torch.utils.data.Dataloader."""
    def __init__(self, root, captionpath, vocab, transform=None):
        """Set the path for images, captions and vocabulary wrapper.
        
        Args:
            root: image directory.
            captionpath: CUB caption file path.
            vocab: vocabulary wrapper.
            transform: image transformer.
        """
        self.root = root
        
        # self.image_paths = open(os.path.join(self.root, 'images.txt'), 'r').readlines()
        image_paths_df = pd.read_table(os.path.join(self.root, 'images.txt'), sep=' ', header=None)
        image_paths_df.columns = ['index', 'img_name']
        train_test_split_df = pd.read_table(os.path.join(self.root, 'train_test_split.txt'), sep=' ', header=None)
        train_test_split_df.columns = ['index', 'split']
        self.image_paths = image_paths_df[train_test_split_df['split'] == 1] # select training set
        self.caption_root = captionpath
        self.vocab = vocab
        self.transform = transform

    def __getitem__(self, index):
        """Returns one data pair (image and caption)."""
        vocab = self.vocab
        # image_path = self.image_paths[index].split()[1]
        # print(self.image_paths.head())
        # print(self.image_paths.loc[index, 'img_name'])
        # image_path = self.image_paths.loc[index, 'img_name']
        image_path = self.image_paths.iloc[[index]].img_name.values[0]
        caption_path = image_path[:-3] + 'txt'

        # get image
        image = Image.open(os.path.join(self.root, 'images/' + image_path)).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        # get caption
        # TODO: only taking first caption for now
        caption = open(os.path.join(self.caption_root, caption_path), 'r').readlines()[0]
        # tokens = nltk.tokenize.word_tokenize(str(caption).lower())
        # Convert caption (string) to word ids.
        tokens = nltk.tokenize.word_tokenize(str(caption).lower())
        caption = []
        caption.append(vocab('<start>'))
        caption.extend([vocab(token) for token in tokens])
        caption.append(vocab('<end>'))
        target = torch.Tensor(caption)

        # getting a wrong target (caption)
        wrong_prob = [1/(len(self.image_paths)-1) if i != index else 0 for i in range(len(self.image_paths))]
        wrong_index = np.random.choice(range(len(self.image_paths)), 1, wrong_prob)[0]
        
        wrong_image_path = self.image_paths.iloc[[wrong_index]].img_name.values[0]
        wrong_caption_path = wrong_image_path[:-3] + 'txt'
        wrong_caption = open(os.path.join(self.caption_root, wrong_caption_path), 'r').readlines()[0]
        wrong_tokens = nltk.tokenize.word_tokenize(str(wrong_caption).lower())
        wrong_caption = []
        wrong_caption.append(vocab('<start>'))
        wrong_caption.extend([vocab(token) for token in wrong_tokens])
        wrong_caption.append(vocab('<end>'))
        wrong_target = torch.Tensor(wrong_caption)

        return image, target, wrong_target

    def __len__(self):
        return len(self.image_paths)


def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (image, caption).
    
    We should build custom collate_fn rather than using default collate_fn, 
    because merging caption (including padding) is not supported in default.

    Args:
        data: list of tuple (image, caption). 
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.

    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length (descending order).
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions, wrong_captions = zip(*data)

    # Merge images (from tuple of 3D tensor to 4D tensor).
    images = torch.stack(images, 0)

    # Merge captions (from tuple of 1D tensor to 2D tensor).
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]

    # Getting wrong targets(captions)
    wrong_lengths = [len(cap) for cap in wrong_captions]
    wrong_targets = torch.zeros(len(wrong_captions), max(wrong_lengths)).long()
    for i, cap in enumerate(wrong_captions):
        end = wrong_lengths[i]
        wrong_targets[i, :end] = cap[:end]
    
    return images, targets, lengths, wrong_targets, wrong_lengths


def get_loader(root, captionpath, vocab, transform, batch_size, shuffle, num_workers):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    # CUB caption dataset
    cub = CUBDataset(root=root,
                       captionpath=captionpath,
                       vocab=vocab,
                       transform=transform)
    
    # Data loader for CUB dataset
    # This will return (images, captions, lengths) for every iteration.
    # images: tensor of shape (batch_size, 3, 224, 224).
    # captions: tensor of shape (batch_size, padded_length).
    # lengths: list indicating valid length for each caption. length is (batch_size).
    data_loader = torch.utils.data.DataLoader(dataset=cub, 
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn)
    return data_loader
