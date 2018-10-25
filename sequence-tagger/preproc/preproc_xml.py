import bz2, sys, io
import xml.etree.ElementTree as et
from gensim.test.utils import datapath, get_tmpfile
from gensim.corpora import WikiCorpus, MmCorpus


file_in = sys.argv[1]
file_out = sys.argv[2]

#with bz2.open(file_in, 'r') as f:
    #data = f.read() 
    #fh = io.StringIO(data.decode('utf8'))
    #print(fh)
    #tree = et.ElementTree()
    #root = tree.parse(fh)
    #pages = []
    #for e in tree.getiterator('page'):wiki = WikiCorpus(path_to_wiki_dump) # create word->word_id mapping, ~8h on full wiki
        #pages.append(e)
        
        
#path_to_wiki_dump = datapath(file_in)
#corpus_path = get_tmpfile("wiki-corpus.mm")

#wiki = WikiCorpus(path_to_wiki_dump) # create word->word_id mapping, ~8h on full wiki
wiki = WikiCorpus(file_in, lemmatize=False, lower=False) # create word->word_id mapping, ~8h on full wiki
#MmCorpus.serialize(corpus_path, wiki) # another 8h, creates a file in MatrixMarket format and mapping    
for text in wiki.get_texts():
    print(text)
    