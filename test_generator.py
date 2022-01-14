import torch 
from datasets import Dataset 
import numpy as np
import model_tools as mtools
import data_tools as dtools

global classifier_tokenizer, classifier_model
global generator_tokenizer, gpt2_model, ft_model

eot = '<|endoftext|>'

def load_all_models(generator_filename=None, classifier_filename=None, use_gpu=True):
    global classifier_tokenizer, classifier_model
    global generator_tokenizer, gpt2_model, ft_model

    if generator_filename is None:
        generator_filename = 'models/JokeGen_gpt2_1.00subset_10epochs_2022-01-07.pt'
    if classifier_filename is None:
        classifier_filename = 'models/ClassifyJokes_bert_1.00subset_2021-12-16.pt'

    # Load the vanilla BERT model, plus its tokenizer
    print('Load the classifier...')
    classifier_checkpoint, classifier_tokenizer, temp_model = mtools.load_model('bert')
    # Load our trained classifier and put it onto the GPU
    classifier_model = torch.load(classifier_filename, map_location=torch.device('cpu'))
    classifier_model, device = mtools.set_device(classifier_model, use_gpu=use_gpu)

    # Load the original GPT-2 model and its tokenizer, and put it on the GPU
    print('Load vanilla GPT-2...')
    checkpoint, generator_tokenizer, gpt2_model = mtools.load_model('gpt2')    
    gpt2_model, device = mtools.set_device(gpt2_model, use_gpu=use_gpu)
    # Load our fine-tuned model and put it on the GPU
    print('Load fine-tuned generator...')
    ft_model = torch.load(generator_filename, map_location=torch.device('cpu'))
    ft_model, device = mtools.set_device(ft_model, use_gpu=use_gpu)
    
    print('Models are ready.')    
    return

def generate_punchlines(input_data, use_gpu=True):
    global generator_tokenizer, gpt2_model, ft_model    

    # Reformat joke setup + punchline into a prompt for the generators
    input_data['full_qa'] = [dtools.joke_as_qa(row['setup'],row['punchline']) for (i,row) in input_data.iterrows()]
    input_data['prompt'] = input_data['full_qa'].apply(lambda x: x[:x.find('Answer:')+len('Answer:')].strip())

    # Generate punchlines using vanilla GPT2 and the Fine-tuned version
    print('Generating punchlines using vanilla GPT-2...')
    generated_gpt2 = mtools.generate(gpt2_model, generator_tokenizer, list(input_data['prompt']))
    print('Generating punchlines using fine-tuned GPT-2...')
    generated_ft = mtools.generate(ft_model, generator_tokenizer, list(input_data['prompt']))
        
    return generated_gpt2, generated_ft


def get_class_predictions(input_data, generated_gpt2, generated_ft, use_gpu=True):

    global classifier_tokenizer, classifier_model

    # Apply tokenization and classification to each joke dataset
    preds_gpt2 = get_predictions(generated_gpt2)
    preds_ft = get_predictions(generated_ft)
    preds_human = get_predictions(input_data['full_qa'])

    n_tot = input_data.shape[0]
    n_human = np.sum(preds_human)
    n_gpt2 = np.sum(preds_gpt2)
    n_ft = np.sum(preds_ft)
    print('Of the {} jokes we checked:'.format(n_tot))
    print('{:>5} ({:4.1f}%) of the Human punchlines get classified as "real" punchlines'.format(n_human,n_human/n_tot*100))
    print('{:>5} ({:4.1f}%) of the GPT-2 punchlines get classified as "real" punchlines'.format(n_gpt2,n_gpt2/n_tot*100))
    print('{:>5} ({:4.1f}%) of the Fine-tuned punchlines get classified as "real" punchlines'.format(n_ft,n_ft/n_tot*100))   
    
    return preds_human, preds_gpt2, preds_ft


# Tokenize the jokes
def class_tokenize_function(example):
    global classifier_tokenizer
    q = [x[:x.find('Answer:')].strip() for x in example['text']]
    a = [x[x.find('Answer:'):].strip() for x in example['text']]
    return classifier_tokenizer(q, a, padding="max_length", max_length=60, truncation=True)

# Get predictions as to which jokes are "real"
def get_predictions(input_texts):
    global classifier_model
    texts = [x.replace(eot,'').replace('\n',' ').strip() for x in input_texts]
    text_dataset = Dataset.from_dict({'text': texts})
    text_tokenized = text_dataset.map(class_tokenize_function, batched=True)
    text_tokenized = text_tokenized.remove_columns(['text'])
    text_tokenized.set_format('torch')
    # Use the classifier to get predictions (1 = real joke, 0 = fake joke) 
    #     and probability of being a "real" joke (from 0.00 to 1.00)
    preds = mtools.classify_punchlines(text_tokenized, classifier_model, 
                                       return_prob=False, use_gpu=True,
                                       quiet=True)
    return preds


def strip_extras(text):
    text = text.replace('\n','')
    while text.count('Question:') > 1:
        text = text[:text.rfind('Question:')]
    while text.count('Answer:') > 1:
        text = text[:text.rfind('Answer:')]
    return text


def tell_a_joke(setup, vanilla_gpt2=False):

    # Format as Q/A joke
    prompt = 'Question: ' + setup + ' Answer:'

    if vanilla_gpt2:
        use_model = gpt2_model
    else:
        use_model = ft_model
    # Keep generating punchlines until one convinces the classifier it's real
    joke_class = 0; counter = 0
    while joke_class==0:
        counter += 1
        ai_joke = strip_extras(mtools.generate(use_model, generator_tokenizer, prompt, quiet=True))
        joke_class = get_predictions([ai_joke])[0]

    final_joke = ai_joke.replace(eot,'').strip()

    print('=======================================================')
    print('Here is our final, AI-generated, AI-approved joke! (N_tries = {})'.format(counter))
    print()
    print('  {}'.format(final_joke[:final_joke.find('Answer:')]))
    print('    {}'.format(final_joke[final_joke.find('Answer:'):]))
    print('=======================================================')
    
    return final_joke