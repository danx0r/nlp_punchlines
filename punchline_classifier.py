import os.path
from datetime import datetime
import argparse

import torch
from datasets import load_dataset
import data_tools as dtools
import model_tools as mtools

global tokenizer
tokenizer=None

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
    return output


def train_punchline_classifier(train_files, test_files, downsample=1):
    global tokenizer

    # Check that a data/ dir exists for output and make one if needed
    if os.path.exists('models') is False:
        os.mkdir('models')
    
    # Load the specified train and test datasets
    dataset = load_dataset('csv', data_files={'train':train_files,'test':test_files})
    
    # Remove any badly-formatted data and downsample, if requested
    dataset = dataset.filter(lambda ex,j: ((type(ex['setup'])==str) & (type(ex['punchline'])==str) & 
                                           (j%downsample==0)),                         
                             with_indices=True)    
    added_text = ' ({}x downsampled)'.format(downsample) if downsample!=1 else ''
    print('{} rows in the train dataset'.format(dataset['train'].num_rows)+added_text+'.')
    print('{} rows in the test dataset'.format(dataset['test'].num_rows)+added_text+'.')

    # Load the pre-trained BERT model checkpoint and associated tokenizer
    checkpoint, tokenizer, model = mtools.load_model('bert')    

    # Tokenize the dataset
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    # Add 'label' based on 'score' (score=0, label=0 for "fake", score>0, label=1 for "real")
    tokenized_datasets = tokenized_datasets.map(lambda batch: {"labels": [int(x > 0) for x in batch["score"]]}, batched=True)

    # Clean up / reformat data to fit into a PyTorch DataLoader
    # We don't need the text strings themselves anymore
    tokenized_datasets = tokenized_datasets.remove_columns(["setup", "punchline", "score"])
    tokenized_datasets.set_format("torch")
    
    # Train the model!
    model = mtools.train_classifier(tokenized_datasets, model, epochs=3)
    
    # Save the trained model
    filename = 'models/ClassifyJokes_{}.pt'.format(checkpoint.split('/')[-1].split('-')[0])
    print('Saving model as {}'.format(filename))
    torch.save(model,filename)
        
    return model


if __name__ == "__main__":
    
    # Get command-line arguments
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train_files', type=str, default=None, required=True,
                        help='comma-separated list of CSV files with training data')
    parser.add_argument('--test_files', type=str, default=None, required=True,
                        help='comma-separated list of CSV files with test data')
    parser.add_argument('--downsample', type=int, default=1, 
                        help='downsample by this factor (int)')
    args = parser.parse_args()

    train_files = args.train_files.split(',')
    test_files = args.test_files.split(',')
    
    print('Using these files for training data: {}'.format(train_files))
    print(' Using these files for testing data: {}'.format(test_files))
    
    train_punchline_classifier(train_files, test_files, downsample=args.downsample)
