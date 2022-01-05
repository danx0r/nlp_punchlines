from torch import load
from datasets import Dataset
import numpy as np

import model_tools as mtools

USE_GPU = False

# Load the punchline generator model
gen_checkpoint, gen_tokenizer, gen_model = mtools.load_model('gpt2')
gen_model_ft = load('models/JokeGen_gpt2_1.00subset_3epochs_2022-01-05.pt')
gen_model, device = mtools.set_device(gen_model, use_gpu=USE_GPU)
gen_model_ft, device = mtools.set_device(gen_model_ft, use_gpu=USE_GPU)

# Load the classification model
class_checkpoint, class_tokenizer, temp_model = mtools.load_model('bert')
class_model = load('models/ClassifyJokes_bert_1.00subset_2021-12-16.pt')
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

#     print('======')
#     print(tokenized_gentext)
#     print('------')
    preds, probs = mtools.classify_punchlines(tokenized_gentext, class_model, return_prob=True,
                                              batch_size=best_of, use_gpu=USE_GPU)
    
#     for i in range(best_of):
#         print(probs[i])
#         print(punchlines[i])

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