import time

import torch
from datasets import Dataset
from burst_tools import gpumem

import model_tools as mtools

def init(use_gpu = "AUTOMATIC",
         generator_filename = 'models/JokeGen_gpt2.pt',
         classifier_filename = 'models/ClassifyJokes_bert.pt'):
    global gen_model, gen_model_ft, gen_tokenizer, class_model, class_tokenizer, USE_GPU

    if use_gpu == "AUTOMATIC":
        USE_GPU = torch.cuda.is_available()
    else:
        USE_GPU = use_gpu

    if USE_GPU:
        try:
            gpumem.mem()
        except:
            raise Exception("no GPU found, aborting init (run with --cpu)")

    #-------------------------------------
    # Load the NLP Models
    #-------------------------------------

    # Load the vanilla generator model, plus its tokenizer
    gen_checkpoint = mtools.load_checkpoint('gpt2')
    gen_tokenizer = mtools.load_tokenizer(gen_checkpoint)
    gen_model = mtools.load_model(gen_checkpoint)
    # Load the fine-tuned generator, always load to CPU first
    gen_model_ft = torch.load(generator_filename, map_location=torch.device('cpu'))

    # Load the vanilla BERT tokenizer
    class_checkpoint = mtools.load_checkpoint('bert')
    class_tokenizer = mtools.load_tokenizer(class_checkpoint)
    # Load our trained BERT classifier
    class_model = torch.load(classifier_filename, map_location=torch.device('cpu'))

    # Put all models on the specified device (GPU or CPU)
    gen_model.to(mtools.get_device(use_gpu=USE_GPU))
    gen_model_ft.to(mtools.get_device(use_gpu=USE_GPU))
    class_model.to(mtools.get_device(use_gpu=USE_GPU))


def class_tokenize_function(example):
    q = [x[:x.find('Answer:')].strip() for x in example['gentext']]
    a = [x[x.find('Answer:'):].strip() for x in example['gentext']]
    return class_tokenizer(q, a, padding="max_length", max_length=60, truncation=True)


def get_punchline(input_text, vanilla_gpt2=False, max_tries=None, threshold=None):
    '''
    Given a text prompt (setup) for a joke, provide an NLP-generated punchline.
    '''
    if USE_GPU:
        print ("USE_GPU")
        if max_tries == None:
            max_tries = 10
        if threshold == None:
            threshold = 0.985
    else:
        print ("USE_CPU")
        if max_tries == None:
            max_tries = 3
        if threshold == None:
            threshold = 0.9
    
    # Format the prompts as "Question: XX Answer: "
    prompts = ['Question: ' + input_text + ' Answer:']

    # Choose between vanilla GPT-2 or fine-tuned model
    gen_mod = gen_model if vanilla_gpt2 else gen_model_ft

    # Generate punchlines until max tries or > thresh
    best_score = best_punch = -1
    for i in range(max_tries):
        print (f"---------------ITERATION: {i}-----------------")
        raw_gentext = mtools.generate(gen_mod, gen_tokenizer, prompts, 
                                      use_gpu=USE_GPU,leave_on_gpu=True)
        gentext = [x.replace(gen_tokenizer.eos_token,'').replace('\n',' ').strip() for x in raw_gentext]
        punchlines = [x[x.find('Answer:')+len('Answser:'):] for x in gentext]

        # Tokenize them for the classifier
        gentext_dataset = Dataset.from_dict({'gentext': gentext})
        tokenized_gentext = gentext_dataset.map(class_tokenize_function, batched=True)
        tokenized_gentext = tokenized_gentext.remove_columns(["gentext"])
        tokenized_gentext.set_format("torch")

        # Use the classifier to get predictions (1 = real joke, 0 = fake joke) 
        #     and probability of being a "real" joke (from 0.00 to 1.00)
        preds, probs = mtools.classify_punchlines(tokenized_gentext, class_model, 
                                                  return_prob=True, batch_size=1, 
                                                  use_gpu=USE_GPU, leave_on_gpu=True)
        score = probs[0]
        print(f"Score: {score} Punchline: {punchlines[0]}")
        if score > best_score:
            best_score = score
            best_punch = punchlines[0]
        if score >= threshold:
            print ("threshold reached, returning")
            break
        # Return the punchline that has the highest probability
    return best_punch


if __name__ == "__main__":
    import sys
    if "--cpu" in sys.argv:
        init(use_gpu=False)
    else:
        init(use_gpu=True)
    print ("USE_GPU:", USE_GPU)
    print ("------------------------------------------------------------------")
    setup = "Why did frogs eat the cheese?"
    print ("Q:", setup)
    t0 = time.time()
    punchline = get_punchline(setup, max_tries=3, threshold=10)
    t1 = time.time() - t0
    i = punchline.find("Answer:")
    if i > 0:
        punchline = punchline[:i]
    print ("A:", punchline)
    print (f"3 iterations in {t1} seconds ({t1/3} seconds per iteration)")
    print ()
