import torch
from torch.utils.data import Dataset

def format_qa(string, string_type='q'):
    q_prefix = ['Q:','Question:']
    a_prefix = ['A:','A :','Answer:','Answers:','Answer :', 'Answers :',
                'Answer -', 'Answer =', 'Answer-', 'Answer=']
    prefix = q_prefix if string_type=='q' else a_prefix
    for p in prefix:
        if string.find(p) > -1:
            string = string[string.find(p)+len(p):]
    string = string.strip()
    prepend = 'Question: ' if string_type=='q' else 'Answer: '
    string = prepend + string
    return string.strip()


def joke_as_qa(setup, punchline, just_q=False, just_a=False):
    if type(setup)==str: setup = [setup]
    if type(punchline)==str: punchline = [punchline]
    jokes = []
    for i,s in enumerate(setup):
        p = punchline[i]
        if p.find(s) > -1:
            p = p[p.find(s)+len(s):].strip()
        joke = str(format_qa(s.strip(),string_type='q') + ' ' + 
                   format_qa(p.strip(),string_type='a'))
        # Keep only the question part, strip the answer
        if just_q:
            joke = joke[:joke.find(' Answer:')]
        # Keep only the answer part, strip the question
        if just_a:
            joke = joke[joke.find('Answer: '):]
        jokes.append(joke)
    return jokes


class Jokes(Dataset):  
    
    def __init__(self, inputlist, tokenizer=None):
        if tokenizer is None:
            print('Error: You must specify a tokenizer to encode this data.')
            return None
        self.tokenizer = tokenizer
        self.tokens = tokenizer(inputlist, padding=True, truncation=True, return_tensors="pt")
        self.count = len(inputlist)
        
    def __len__(self):
        return self.count

    def __getitem__(self, item):
        return self.tokens[item]
    
    
def trunc_at_last_stop(text):
    '''
    Clean up NLP-generated text because AI seems to be bad at knowing where to put an EOS.
    '''
    
    # Try to truncate at a full stop, otherwise truncate at last partial stop
    stops = ['.','!','?','\n']
    partial_stops = [',','"','-',':']
    text_list = text.split()
    tstop = [x for x in text_list if any(x.find(s) > -1 for s in stops)]
    tpstop = [x for x in text_list if any(x.find(ps) > -1 for ps in partial_stops)]
    if len(tstop)==0:  # If no 'stop' punctuation found, look for partial stop
        if len(tpstop)>0:
            output = text[:text.rfind(tpstop[-1])+len(tpstop[-1])]
        else:
            output = text
    else:              # If there are stops, return the text up to the last stop
        output = text[:text.rfind(tstop[-1])+len(tstop[-1])]
        
    # Get rid of all newlines (\n) 
    output = output.replace('\n','')
    # Strip off leading non-alpha text
    alphas = [x for x in output if x.isalpha()]
    if len(alphas) > 0:
        first_alpha = alphas[0]
        output = output[output.find(first_alpha):]
    # Get rid of unmatched quotes at end of generated text
    quote_count = output.count('"')
    if quote_count % 2 == 1 and output[-1]=='"':
        output = output[:-1]
    # Add closing parentheses if they are missing
    if output.find('(') > -1 and output.find(')')==-1:
        output += ')'
    if output[-1] in [',',':','-']:
        output = output[:-1] + '...'
    # Add closing or opening double quotes if they are missing
    if output.count('"')==1:
        if output.find('"') > 0 and ~output[output.find('"')-1].isalpha():
            output = '"'+output 
        else:
            output += '"'
    return output

