import os
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from burst_tools import gpumem
from datasets import load_metric
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM
from transformers import AdamW, get_scheduler, get_linear_schedule_with_warmup
from tqdm import tqdm, trange


def load_checkpoint(model_name):
    # Map model shortcut names onto full HuggingFace model names
    model_map = {'gpt2':'gpt2', 
                 'bart':'facebook/bart-base', 
                 'bert':'bert-base-uncased', 
                 't5':'t5-base'}
    if model_name in model_map.values():
        checkpoint = model_name
    elif model_name in model_map.keys():
        checkpoint = model_map[model_name]
    else:
        print('Model not in recognized set.  Please choose a model from this list:')
        for k,v in model_map.items():
            print('   {} (uses "{}" checkpoint)'.format(k,v))
        return None
    print('Using checkpoint "{}"'.format(checkpoint))
    return checkpoint


def load_tokenizer(checkpoint):
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    # If no End-Of-String token, create one
    # If no Beginning-Of-String or Pad token, make them match EOS
    if tokenizer.eos_token is None:
        print('Warning: no EOS token detected.  Adding an EOS token')
        tokenizer.eos_token = '<eos>'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.bos_token is None:
        tokenizer.bos_token = tokenizer.eos_token
    return tokenizer


def load_model(checkpoint):
    # Map checkpoint names to the AutoModel type they will work with
    automodel_map = {'gpt2': AutoModelForCausalLM,
                     'facebook/bart-base': AutoModelForSeq2SeqLM,
                     'bert-base-uncased': AutoModelForSequenceClassification,
                     't5-base': AutoModelForSeq2SeqLM}    
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)    
    AutoModel = automodel_map[checkpoint]
    model = AutoModel.from_pretrained(checkpoint)
    # If no End-Of-String token, create one
    # If no Beginning-Of-String or Pad token, make them match EOS
    if tokenizer.eos_token is None:
        print('Warning: no EOS token detected.  Adding an EOS token')
        model.config.eos_token_id = tokenizer.eos_token_id
    if tokenizer.pad_token is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    if tokenizer.bos_token is None:
        model.config.bos_token_id = tokenizer.eos_token_id
    return model


# Accumulated batch size (since GPT2 is so big)
def pack_tensor(new_tensor, packed_tensor, max_seq_len):
    if packed_tensor is None:
        return new_tensor, True, None
    if new_tensor.size()[1] + packed_tensor.size()[1] > max_seq_len:
        return packed_tensor, False, new_tensor
    else:
        packed_tensor = torch.cat([new_tensor, packed_tensor[:, 1:]], dim=1)
        return packed_tensor, True, None
    
    
def get_device(use_gpu=True):
    # If GPU was requested, find the one with most free memory
    if use_gpu:
        best, free = gpumem.least_used()
        if best is None:
            raise Exception("GPU unavailable")
        else:
            device = torch.device("cuda:{}".format(best))
    else:
        device = torch.device("cpu")
    return device
    
    
def train_classifier(dataset, model, use_gpu=True, 
                     batch_size=8, epochs=3, lr=5e-5,
                     warmup_steps=0,
                     output_dir="./checkpoints/", output_prefix="temp_classifier",
                     save_model_on_epoch=True):
    
    # Make sure output directory exists
    if os.path.exists(output_dir) is False:
        os.mkdir(output_dir)

    train_dataloader = DataLoader(dataset["train"], shuffle=True, batch_size=8)
    eval_dataloader = DataLoader(dataset["test"], batch_size=8)
    
    optimizer = AdamW(model.parameters(), lr=lr)
    nsteps = epochs * len(train_dataloader)
    lr_scheduler = get_scheduler("linear",optimizer=optimizer,
                                 num_warmup_steps=warmup_steps,
                                 num_training_steps=nsteps)
    metric= load_metric("glue", "mrpc")

    model.to(get_device(use_gpu=use_gpu))   

    progress_bar = tqdm(range(nsteps))
    losses = []
    for epoch in range(epochs):
        
        # Trainining pass 
        model.train()  # Put the model into "training" mode
        for batch in train_dataloader:
            batch = {k: v.to(model.device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

        # Evaluate this epoch
        model.eval()   # Put the model into "eval" mode
        for batch in eval_dataloader:
            batch = {k: v.to(model.device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            metric.add_batch(predictions=predictions, references=batch["labels"])

        # Output stats so far
        temp = metric.compute()
        print('Epoch: {}, Loss: {:4.3f}, Accuracy: {:4.3f}, F1: {:4.3f}'.format(epoch,loss,temp['accuracy'],temp['f1']))
        losses.append(loss)        
        if save_model_on_epoch:
            torch.save(model.state_dict(),os.path.join(output_dir, f"{output_prefix}-{epoch}.pt"))

    print('Done training model.  Final model has:')
    print('       Accuracy: {}'.format(temp['accuracy']))
    print('             F1: {}'.format(temp['f1']))

    # Move model off GPU and free up GPU memory
    model.to(device='cpu')   # Put model back on CPU
    del(batch,outputs,logits,predictions,metric,loss,losses)
    del(optimizer,lr_scheduler)
    torch.cuda.empty_cache()

    return model


def classify_punchlines(dataset, model, quiet=False,
                        use_gpu=True, batch_size=8,
                        return_prob=False,
                        leave_on_gpu=False):
    
    # Put model on the correct device
    model.to(get_device(use_gpu=use_gpu))
    print('CP start: model is on {}'.format(model.device))
    if use_gpu and leave_on_gpu and model.device != 'cpu':
        pass
    else:
        print('CP moving the model...')
        model.to(get_device(use_gpu=use_gpu))
    print('CP use model: model is on {}'.format(model.device))
    model.eval()   # Put the model into "eval" mode

    eval_dataloader = DataLoader(dataset, batch_size=batch_size)
    if not quiet:
        print('{} batches to process (batch_size={})'.format(len(eval_dataloader),batch_size))
        progress_bar = tqdm(range(len(eval_dataloader)))

    predictions = []; probs = []
    for batch in eval_dataloader:
        batch = {k: v.to(model.device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        logits = outputs.logits
        predictions_i = torch.argmax(logits, dim=-1).to(device='cpu') # output data on CPU
        predictions.extend(list(predictions_i.numpy()))
        if return_prob==True:
            probs_i = F.softmax(outputs.logits, dim=-1).to(device='cpu')
            probs.extend(list([p[-1] for p in probs_i.numpy()]))
        if not quiet:
            progress_bar.update(1)
            
    # Move model off GPU and free up GPU memory
    if use_gpu and not(leave_on_gpu):
        model.to(device='cpu')   # Put model back on CPU to free up GPU
        del(batch,outputs,logits)
        torch.cuda.empty_cache()
    print('CP end: model is on {}'.format(model.device))
            
    if return_prob:
        return predictions, probs
    else:
        return predictions

    
def train_generator(train_dataset, model, use_gpu=True,
                    batch_size=16, epochs=5, lr=2e-5,
                    max_seq_len=400, warmup_steps=200,
                    output_dir="./checkpoints/", output_prefix="temp",
                    test_mode=False,save_model_on_epoch=True):
    
    # Make sure output directory exists
    if os.path.exists(output_dir) is False:
        os.mkdir(output_dir)

    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)

    # Put model on GPU, if available
    model.to(get_device(use_gpu=use_gpu))
    model.train()  # Put model in "train" mode
 
    acc_steps = 100
    optimizer = AdamW(model.parameters(), lr=lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps=warmup_steps, 
                                                num_training_steps=-1)

    loss=0
    accumulating_batch_count = 0
    input_tensor = None

    for epoch in range(epochs):

        print(f"Training epoch {epoch}")
        progress_bar = tqdm(range(len(train_dataloader)), position=0, leave=True)
        for idx, entry in enumerate(train_dataloader):
            progress_bar.update(1)
            (input_tensor, carry_on, remainder) = pack_tensor(entry, input_tensor, 768)
            if carry_on and idx != len(train_dataloader) - 1:
                continue
            input_tensor = input_tensor.to(model.device)
            # For a generator, we use the input text itself as the labels --> the generator
            #     at each word is trying to predict what word comes next.  
            outputs = model(input_tensor, labels=input_tensor)
            loss = outputs[0]
            loss.backward()
            if (accumulating_batch_count % batch_size) == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                model.zero_grad()
            accumulating_batch_count += 1
            input_tensor = None
        if save_model_on_epoch:
            torch.save(
                model.state_dict(),
                os.path.join(output_dir, f"{output_prefix}-{epoch}.pt"),
            )
            
    # Move model off GPU and free up GPU memory    
    model.to(device='cpu')  # Put model back on CPU to free up GPU
    del(input_tensor,loss,outputs)
    del(optimizer,scheduler)
    torch.cuda.empty_cache()
    
    return model


def generate(model, tokenizer, prompts,
             use_gpu=True, quiet=False,
             maxlength=30, # maximum number of words in newly generated text
             top_p=0.8, temperature=1.,
             leave_on_gpu=False):
    '''
    Use the provided model and tokenizer to extend the input prompts with generated text.
    Adapted from an article and demo code by Francois St-Amant:
    https://towardsdatascience.com/how-to-fine-tune-gpt-2-for-text-generation-ae2ea53bc272 
    '''

    # Input is a list of strings with length n
    str_input = type(prompts)==str
    if str_input:
        prompts = [prompts]

    with torch.no_grad():
        
        # Use GPU device if requested (default: use_gpu=True) and it is available
        print('G start: model is on {}'.format(model.device))
        model.eval()        # Put model in "eval" mode
        if use_gpu and leave_on_gpu and model.device != 'cpu':
            pass
        else:
            print('moving the model...')
            model.to(get_device(use_gpu=use_gpu))
        print('G use model: model is on {}'.format(model.device))
                
        output_list = []
        for i in trange(len(prompts)):
            
            # Token string starts with tokenized input, on same device as the model
            gentokens = torch.tensor(encode_prompt(prompts[i],tokenizer)).unsqueeze(0)
            gentokens = gentokens.to(model.device)

            # Now generate additional tokens up to maxlength
            for j in range(maxlength):
                
                # Pass the tokens generated so far ("gentokens") through the model, 
                #       get loss & logits for the next token prediction
                loss, logits = model(gentokens, labels=gentokens)[:2]
                
                # Scale by the temperature (higher temperature ==> more original/unusual words)
                logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
                
                # Keep most likely predicted token possibilities up to a cumsum of top_p
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[:, indices_to_remove] = -float("Inf")

                # Next token - choose randomly from token possiblities selected above, 
                #      proportional to softmax distrib
                next_token = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
                gentokens = torch.cat((gentokens, next_token), dim=1)

                # If we got an EOS token, stop generating tokens, otherwise continue to maxlength
                if next_token in tokenizer.encode(tokenizer.eos_token):
                    break
                    
            # Move the data back to the CPU, decode, and store the generated text
            
            loss = loss.detach()
            logits = logits.detach()
            next_token = next_token.detach()
            gentokens = gentokens.cpu()
            output_tokens = list(gentokens.squeeze().numpy())
            output_text = tokenizer.decode(output_tokens)
            output_list.append(output_text)

    # Move model off GPU and free up GPU memory   
    if use_gpu and not(leave_on_gpu):
        model.to(device='cpu')  # Put model back onto CPU to free up GPU
        del(loss,logits,next_token,gentokens)
        torch.cuda.empty_cache()
    print('G end: model is on {}'.format(model.device))
    
    if str_input:
        output_list = output_list[0]
        
    return output_list


def encode_prompt(prompt, tokenizer):
    '''
    Standardize encoding to "[BOS] prompt" with no padding or EOS, because we want the string to continue
    '''
    encoded = tokenizer.encode(prompt)
    # Strip off BOS, EOS, and PAD tokens when the exist
    if encoded[0]==tokenizer.bos_token_id: encoded = encoded[1:]
    if encoded[-1]==tokenizer.eos_token_id: encoded = encoded[:-1]
    if encoded.count(tokenizer.pad_token_id) > 0: encoded = [x for x in encoded if x!=tokenizer.pad_token_id]
    # Add BOS token back to the front of the token list
    encoded = [tokenizer.bos_token_id] + encoded
    return encoded