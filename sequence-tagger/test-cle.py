
from flair.data_fetcher import NLPTaskDataFetcher
from flair.data import Dictionary
import random
import numpy as np

fh = "/home/liefe/data/pt/pos/macmorpho1"
cols = {0:"text", 1:"pos"}


# Fetch corpus
print("Getting corpus")
corpus = NLPTaskDataFetcher.fetch_column_corpus(fh, 
                                                cols, 
                                                train_file="train.txt",
                                                dev_file="dev.txt", 
                                                test_file="test.txt")


char_dict = Dictionary.load('common-chars')
use_cle = True
batch_size = 2

# Train epochs
train_data = corpus.train  
for epoch in range(1):
 
    
    # Shuffle data and form batches
    random.shuffle(train_data)
    batches = [train_data[i:i + batch_size] for i in range(0, len(train_data), batch_size)]        
           
    
    for batch_n, batch in enumerate(batches[:1]):
        
        # Sort batch and get lengths
        batch.sort(key=lambda i: len(i), reverse=True)
        
        # Remove super long sentences                
        max_sent_len = len(batch[0])                                    
        while len(batch) > 1 and max_sent_len > 100:
            batch = batch[1:]
            max_sent_len = len(batch[0])                    
            
        sent_lens = [len(s.tokens) for s in batch]
        n_sents = len(sent_lens)        
        

        if use_cle:
            char_seq_map = {'<pad>': 0}
            char_seqs = [[0]]
            char_seq_ids = []
            
        for i in range(n_sents):
            ids = np.zeros([max_sent_len])
            for j in range(sent_lens[i]):                    
                token = batch[i][j]
                
                print(token.text, len(token.text))

                if use_cle:
                    if token.text not in char_seq_map:
                        char_seq = [char_dict.get_idx_for_item(c) for c in token.text]                        
                        char_seqs.append(char_seq)
                        char_seq_map[token.text] = len(char_seqs)
                    ids[j] = char_seq_map[token.text]
                
                
                #gold_tags[i, j] = self.tag_dict.get_idx_for_item(token.get_tag(self.tag_type).value)      # tag index         
             
            # Append sentence char_seq_ids
            char_seq_ids.append(ids)
            
            # Pad char sequences            
            if use_cle:
                #char_seqs.sort(key=lambda i: len(i), reverse=True)
                char_seq_lens = [len(char_seq) for char_seq in char_seqs]
                #max_char_seq = len(char_seqs[0])
                max_char_seq = max(char_seq_lens)
                n = len(char_seqs)       
                batch_char_seqs = np.zeros([n, max_char_seq])
                
                for i in range(n):
                    batch_char_seqs[i, 0:len(char_seqs[i])] = char_seqs[i]            
            
            
            
            
            print(batch)
            
            print(char_seqs)
            
            print(char_seq_ids)
            
            print(char_seq_map)

            print(char_dict.item2idx)

            print(batch_char_seqs[0])

            n_chars = len(char_dict)
            cle = np.random.random([n_chars, 32])
            emb = np.zeros([len(batch_char_seqs), max_char_seq, 32])
            for i in range(len(batch_char_seqs)):
                for j in range(len(batch_char_seqs[i])):
                    emb[i, j] = cle[i]
            
            #print(cle[0])       
            print(emb[0])
            
#for sentence in corpus.train[:10]:

    ## Get character sequences
    #for token in sentence.tokens:
        #token: Token = token
        #char_indices = [self.char_dictionary.get_idx_for_item for char in token.text]
        #tokens_char_indices.append(char_indices)