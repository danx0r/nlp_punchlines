import os
import pandas as pd
import argparse

default_path = '/opt/tljh/user/share/nlp_punchlines_data/one-million-reddit-jokes.csv'

def clean_data(file=default_path):

    # Check that a data/ dir exists for output and make one if needed
    if os.path.exists('data') is False:
        os.mkdir('data')
    
    #-----------------------------------
    # Read in the Reddit jokes data
    #-----------------------------------
    print('Reading in raw jokes data...')
    raw_jokes = pd.read_csv(file,keep_default_na=False)
    print('{:>10} jokes in the raw dataset'.format(raw_jokes.shape[0]))

    #--------------------------------------------------------------------------------
    # Remove jokes whose punchlines have been deleted, only store relevant columns
    #--------------------------------------------------------------------------------
    remove = ['[removed]','[deleted]','\[removed\]']   # We'll remove jokes with these punchlines
    use_columns = ['title','selftext','score']  # We only care about these columns
    jokes = raw_jokes[raw_jokes['selftext'].apply(lambda x: x not in remove)][use_columns]

    #--------------------------------------
    # Rename columns and remove newlines
    #--------------------------------------
    jokes = jokes.rename(columns={'title':'setup','selftext':'punchline'})
    jokes['setup'] = jokes['setup'].apply(lambda x: x.replace('\n',' ').replace('\r',' '))
    jokes['punchline'] = jokes['punchline'].apply(lambda x: x.replace('\n',' ').replace('\r',' '))

    #--------------------------------------------------------------------
    # Filter for a specific type of joke:
    #    - Joke setup is a question
    #    - Joke Punchline is short (20 words max)
    #    - 1+ people thought the joke was funny (at least one upvote)
    #--------------------------------------------------------------------
    # Only keep jokes that are questions
    jokes = jokes[jokes['setup'].apply(lambda x: True if x[-1]=='?' else False)]
    # Only keep jokes with punchlines 1-20 words long
    jokes = jokes[jokes['punchline'].apply(lambda x: 1 <= len(x.split()) <= 20)]
    # Only keep jokes that got at least one upvote
    jokes = jokes[jokes['score'] >= 1]
    
    #--------------------------------
    # Get rid of duplicate entries
    #--------------------------------
    # Sum the scores for all jokes with the same setup and punchline
    jokes['score'] = jokes.groupby(['setup', 'punchline'])['score'].transform('sum')
    # Then drop the duplicate entries
    jokes = jokes.drop_duplicates(subset=['setup','punchline'])    

    #---------------------------
    # Save the clean dataset 
    #---------------------------
    print('{:>10} jokes in the final dataset (Q|A format, short punchlines, 1+ upvotes)'.format(jokes.shape[0]))
    outfile = 'data/short_jokes.csv'
    output_columns = ['setup','punchline','score']

    #------------------------------------------------------
    # Split into training and test datasets, save splits
    #------------------------------------------------------
    train_frac = 0.7  # Use 70% of jokes for training, 30% for testing
    seed = 40         # Use seed so that the split is always the same
    mini_count = 300  # Let's also store a small subset of the test data as a "mini" test to use during development.

    jokes_train = jokes.sample(frac=train_frac, axis=0, random_state=seed)
    jokes_test = jokes[~jokes.index.isin(jokes_train.index)]

    print('Joke splits written to:')
    for dset,name in [(jokes,'_all'),
                      (jokes_train, '_train'),
                      (jokes_test,'_test'),
                      (jokes_train.iloc[:int(mini_count*train_frac)],'_minitrain'),
                      (jokes_test.iloc[:int(mini_count*(1-train_frac))],'_minitest')]:
        dset[output_columns].to_csv(outfile.replace('.csv',name+'.csv'), header=True, index=False)
        print('{:>10} in {}'.format(dset.shape[0],outfile.replace('.csv',name+'.csv')))
        
    return


if __name__ == "__main__":
    
    # Get command-line arguments
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--file', type=str, default=default_path, 
                        help='input data file path')
    args = parser.parse_args()

    print('Using input file {}'.format(args.file))
    csvfile = args.file
    clean_data(file=csvfile)
    
    