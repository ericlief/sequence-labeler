from flair.data_fetcher import NLPTaskDataFetcher
from flair.embeddings import CharLMEmbeddings
import numpy as np
from gensim.models.keyedvectors import FastTextKeyedVectors
from gensim.models import KeyedVectors

#vec = FastTextKeyedVectors.load("/home/liefe/.flair/embeddings/pt-fasttext-300d-1M")
#vec = FastTextKeyedVectors.load("/home/liefe/.flair/embeddings/cc.pt.300.bin.gz")

#print(vec)

## 1. get the corpus
#train_fh = "/home/liefe/data/pt/UD_Portuguese-Bosque"
#cols = {1:"text", 2:"lemma", 3:"pos"}
#corpus = NLPTaskDataFetcher.fetch_column_corpus(train_fh, cols, 
                                                #train_file="train.txt",
                                                #dev_file="dev.txt", 
                                                #test_file="test.txt").downsample(.10)

#lm_fw_embed = CharLMEmbeddings("/home/liefe/code/sequence-tagger/sequence-tagger/resources/taggers/language_models/fwd/best-lm.pt")
#embeded = lm_fw_embed.embed(corpus.train)
#for t in corpus.train:
    #print(t)
    #print(t.embedding)
    
    
    
import io
# This will crash
def load_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        #print(tokens)
        data[tokens[0]] = map(float, tokens[1:])
    return data


#data = load_vectors("/home/liefe/.flair/embeddings/cc.fa.300.vec")
#data = load_vectors("/home/liefe/.flair/embeddings/cc.pt.300.vec")


# This works to load in gensim format
#data = KeyedVectors.load_word2vec_format('/home/liefe/.flair/embeddings/cc.pt.300.vec', limit=100000)
data = KeyedVectors.load_word2vec_format('/home/liefe/.flair/embeddings/cc.pt.300.vec')

#print(len(data))
print(data['de'])
print(data['.'])
print(data['pudesse'])	

# Save in gensim format for flair
data.save('/home/liefe/.flair/embeddings/cc.pt.300.kv')


#with open("/home/liefe/.flair/embeddings/pt-fasttext-300d-1M", 'rt', errors='ignore') as f:
                                                
                                                #print(f.read()
          

#with open("/home/liefe/.flair/embeddings/pt-fasttext-300d-1M", 'rt', errors='ignore') as f:
                                                #print(f.read())
#emb = np.load("/home/liefe/.flair/embeddings/pt-fasttext-300d-1M.vectors.npy")
#print(emb.shape)

