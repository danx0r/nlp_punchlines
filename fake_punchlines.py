import os.path
import pandas as pd
from datetime import datetime
import argparse
import transformers

import data_tools as dtools
import model_tools as mtools

BATCHSIZE = 100

def add_fake_punchlines(infile, start_i=0, downsample=1):
    '''
    Use a pre-trained NLP text-generator to create "fake" punchlines for each 
    joke in our dataset, with only the joke setup as a prompt.
    Save fake punchlines in a separate CSV file of "fake" jokes.
    '''
        
    # Create output CSV filename to store fake joke output 
    outfile = infile.replace('.csv','_fake.csv')
    if os.path.exists(outfile): os.remove(outfile)
    
    df = pd.read_csv(infile,dtype={'setup':str,'punchline':str,'score':int},keep_default_na=False)
    
    # Downsample if requested and write out downsampled "real" jokes as well
    if downsample != 1:
        df = df.sample(frac=1./downsample)
        df.to_csv(infile.replace('.csv','_ds{}.csv'.format(downsample)),index=False)
        outfile = outfile.replace('.csv','_ds{}.csv'.format(downsample))        
    print('{} jokes in the dataset'.format(df.shape[0]))    
    df['full_qa'] = df.apply(lambda x: dtools.joke_as_qa(x['setup'], x['punchline']), axis=1)
    df['prompt'] = df['full_qa'].apply(lambda x: x[:x.find('Answer: ')+len('Answer:')])    
    

    checkpoint = mtools.load_checkpoint('gpt2')
    tokenizer = mtools.load_tokenizer(checkpoint)
    model = mtools.load_model(checkpoint)
    
    # Iterate through the input "real" jokes in batches, get a "fake" punchline for each joke
    nlines = df.shape[0]
    t0 = datetime.now()
    for i in range(start_i, nlines, BATCHSIZE):          
        
        print('Working on jokes {}:{} out of {} -- {}'.format(i,i+BATCHSIZE,
                                                              nlines,datetime.now()-t0))
        
        # Get the next batch of jokes
        df_i = df.iloc[i:i+BATCHSIZE]
        
        output = mtools.generate(model, tokenizer, list(df_i['prompt']))
        
        # Clean up the responses 
        # Remove the prompt so we're just storing the newly generated text
        fakelines = [x[x.find(df_i.iloc[j]['prompt'])+len(df_i.iloc[j]['prompt']):] for j,x in enumerate(output)]
        # Get rid of newlines and leading/trailing spaces, just like we did in the initial dataset
        fakelines = [x.replace('\n',' ').replace('\r',' ').strip() for x in fakelines]
        
        # Store setup, cleaned fake punchlines, and score=0 (no upvotes) 
        output_df = pd.DataFrame()
        output_df['setup'] = df_i['setup']
        output_df['punchline'] = fakelines
        output_df['score'] = 0
        
        # Append this batch to the output file
        write_header = False if os.path.exists(outfile) else True  # only write the header the first time
        output_df.to_csv(outfile, mode='a', index=False, header=write_header)
        print('Done writing {}:{} to CSV'.format(i,i+BATCHSIZE))
        
    return


if __name__ == "__main__":
    
    # Get command-line arguments
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('file', type=str, default=None, 
                        help='input data file path')
    parser.add_argument('--downsample', type=int, default=1, 
                        help='downsample data by this factor (default = 1, no sampling)')
    parser.add_argument('--start', type=int, default=0, 
                        help='Which joke index to start on (default = 0, other values to continue crashed process.')
    args = parser.parse_args()

    print('Use input file {}'.format(args.file))
    csvfile = args.file
    add_fake_punchlines(csvfile, start_i=args.start, downsample=args.downsample)
    
    