import torch
from datasets import Dataset
import numpy as np

import model_tools as mtools

USE_GPU = False

generator_filename = 'models/JokeGen_gpt2_1.00subset_3epochs_2022-01-05.pt'
classifier_filename = 'models/ClassifyJokes_bert_1.00subset_2021-12-16.pt'

#-------------------------------------
# Load the NLP Models
#-------------------------------------

# Load the vanilla generator model, plus its tokenizer
gen_checkpoint, gen_tokenizer, gen_model = mtools.load_model('gpt2')  
# Load the fine-tuned generator
gen_model_ft = torch.load(generator_filename, map_location=torch.device('cpu'))

# Load the vanilla BERT model, plus its tokenizer
class_checkpoint, class_tokenizer, temp_model = mtools.load_model('bert')
# Load our trained classifier
class_model = torch.load(classifier_filename, map_location=torch.device('cpu'))

# Put all models on the specified device (GPU or CPU)
gen_model, device = mtools.set_device(gen_model, use_gpu=USE_GPU)
gen_model_ft, device = mtools.set_device(gen_model_ft, use_gpu=USE_GPU)
class_model, device = mtools.set_device(class_model, use_gpu=USE_GPU)


def class_tokenize_function(example):
    q = [x[:x.find('Answer:')].strip() for x in example['gentext']]
    a = [x[x.find('Answer:'):].strip() for x in example['gentext']]
    return class_tokenizer(q, a, padding="max_length", max_length=60, truncation=True)


def get_punchline(input_text, vanilla_gpt2=False, best_of=5):
    '''
    Given a text prompt (setup) for a joke, provide an NLP-generated punchline.
    '''
    # Format the prompts as "Question: XX Answer: "
    prompt = 'Question: ' + input_text + ' Answer:'
    # Duplicate the prompt as many times as requested
    prompts = [prompt] * best_of

    # Choose between vanilla GPT-2 or fine-tuned model
    gen_mod = gen_model if vanilla_gpt2 else gen_model_ft

    # Generate n versions of the punchline to choose between
    raw_gentext = mtools.generate(gen_mod, gen_tokenizer, prompts)
    gentext = [x.replace(gen_tokenizer.eos_token,'').replace('\n',' ').strip() for x in raw_gentext]
    punchlines = [x[x.find('Answer:')+len('Answser:'):] for x in gentext]
    
    # Tokenize them for the classifier
    gentext_dataset = Dataset.from_dict({'gentext': gentext})
    tokenized_gentext = gentext_dataset.map(class_tokenize_function, batched=True)
    tokenized_gentext = tokenized_gentext.remove_columns(["gentext"])
    tokenized_gentext.set_format("torch")

    # Use the classifier to get predictions (1 = real joke, 0 = fake joke) 
    #     and probability of being a "real" joke (from 0.00 to 1.00)
    preds, probs = mtools.classify_punchlines(tokenized_gentext, class_model, return_prob=True,
                                              batch_size=best_of, use_gpu=USE_GPU)
    
    # Return the punchline that has the highest probability
    return punchlines[np.argmax(probs)]


if __name__ == "__main__":
    print ("------------------------------------------------------------------")
    setup = "Why did the chicken cross the road?"
    for i in range(111):
        print ("Q:", setup)
        punchline = get_punchline(setup, best_of=3)
        print ("A:", punchline)
        print ()
        # setup = input("Q:")
        if i < 2:
            setup = "Knock knock. Who's there?"
        else:
            setup = input("Type joke here (or <enter> to quit)")
            if not setup.strip():
                break