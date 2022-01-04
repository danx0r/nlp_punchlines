import os
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from datasets import load_dataset

import pandas as pd
import argparse
from datetime import datetime

import data_tools as dtools
import model_tools as mtools

global tokenizer

# Tokenization function convert joke strings to tokenized 
def tokenize_function(example):
    global tokenizer
    
    # Reformat the jokes strings into the "Question: XX Answer: YY" format
    full_qa = dtools.joke_as_qa(example['setup'], example['punchline'])
    # Split the questions from the answers (these are our two sequences)
    q = [x[:x.find('Answer:')].strip() for x in full_qa]
    a = [x[x.find('Answer:'):].strip() for x in full_qa]
    # Tokenize the sequences
    #  - pad and truncate to make all the same length for happy PyTorch tensors
    output = tokenizer(q, a, padding="max_length", max_length=60, truncation=True)
    # Give attention to the first pad token
    for am in output['attention_mask']:
        if 0 in am:
            pad_start = am.index(0)
            am[pad_start] = 1
    return output


def fine_tune(train_files, use_model="gpt2", downsample=1, nepochs=3):
    global tokenizer
    
    # Load the specified training dataset
    dataset = load_dataset('csv', data_files={'train':train_files})
    
    # Remove any badly-formatted data and downsample, if requested
    dataset = dataset.filter(lambda ex,j: ((type(ex['setup'])==str) & (type(ex['punchline'])==str) & 
                                           (j%downsample==0)),                         
                             with_indices=True)    
    added_text = ' ({}x downsampled)'.format(downsample) if downsample!=1 else ''
    print('{} rows in the train dataset'.format(dataset['train'].num_rows)+added_text+'.')
    
    # Load the pre-trained model, checkpoint, and corresponding tokenizer we want to use
    checkpoint, tokenizer, model = mtools.load_model(use_model)    

    # Tokenize the dataset
    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    # Clean up / reformat data to fit into a PyTorch DataLoader
    # We don't need the text strings themselves anymore
    tokenized_datasets = tokenized_datasets.remove_columns(["setup", "punchline", "score"])
    tokenized_datasets.set_format("torch")
    
    train_dataset = tokenized_datasets['train']['input_ids']
#     print('{:8d} jokes encoded in the training set'.format(train_dataset.joke_count))
    
    model = mtools.train_generator(train_dataset, model, tokenizer, epochs=nepochs)

    if os.path.exists('models') is False: os.mkdir('models')
    filename = 'models/JokeGen_{}_{:4.2f}subset_{}epochs_{}.pt'.format(use_model,1./downsample,nepochs,datetime.now().date())
    print('Saving model as {}'.format(filename))
    torch.save(model,filename)
    
    return



if __name__ == "__main__":
    
    # Get command-line arguments
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train_files', type=str, default=None, required=True,
                        help='comma-separated list of CSV files with training data')
    parser.add_argument('--downsample', type=int, default=1, 
                        help='downsample by this factor (int)')
    parser.add_argument('--model', type=str, default='gpt2', 
                        help='Name of pre-trained model to use as starting point ("gpt2","bert","bart","t5")')
    parser.add_argument('--nepochs', type=int, default=3, 
                        help='Number of training epochs (iterations) to train through (default=3)')
    args = parser.parse_args()

    train_files = args.train_files.split(',')    
    print('Using these files for training data: {}'.format(train_files))
    
    fine_tune(train_files=train_files, use_model=args.model, 
              downsample=args.downsample, nepochs=args.nepochs)
    
