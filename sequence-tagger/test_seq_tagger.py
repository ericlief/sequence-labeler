from flair.data_fetcher import NLPTaskDataFetcher
from flair.embeddings import CharLMEmbeddings, WordEmbeddings, StackedEmbeddings

#train_fh = "/home/liefe/data/pt/UD_Portuguese-Bosque"
#cols = {1:"text", 2:"lemma", 3:"pos"}

#corpus = NLPTaskDataFetcher.fetch_column_corpus(train_fh, cols, train_file="train.txt",
                        #dev_file="dev.txt", test_file="test.txt")
#print(len(corpus.train))
#print(corpus)
#print(corpus.train[0].to_tagged_string('pos'))
##for s in sentences:
    ##for e in s.get_spans('udep'):
    ##print(e)


#tag_type = "ne"
#tag_dict = corpus.make_tag_dictionary(tag_type=tag_type)

#print(tag_dict.idx2item)

#lm_fw_embed = CharLMEmbeddings("/home/liefe/code/sequence-tagger/sequence-tagger/resources/taggers/language_models/fwd/best-lm.pt")

#from flair.models import SequenceTagger

#tagger = SequenceTagger(hidden_size=64,
            #embeddings=lm_fw_embed,
            #tag_dictionary=tag_dict,
            #tag_type=tag_type,
            #use_crf=True)

#from flair.trainers import SequenceTaggerTrainer

#trainer = SequenceTaggerTrainer(tagger, corpus)

#trainer.train("resources/taggers/example-pos",
        #learning_rate=.1,
        #mini_batch_size=32,
        #max_epochs=150)

fh = "/home/liefe/data/pt/harem/data/HAREM"
#cols = {1:"text", 2:"lemma", 3:"pos"}
cols = {0:"text", 1:"ne"}    
corpus = NLPTaskDataFetcher.fetch_column_corpus(fh, 
                                                cols, 
                                                train_file="train.txt",
                                                dev_file="dev.txt", 
                                                test_file="test.txt").downsample(.1)



# Load Character Language Models (clms)
clm_fw = CharLMEmbeddings("/home/liefe/lm/fwd/best-lm.pt")  
clm_bw = CharLMEmbeddings("/home/liefe/lm/bw_p25/best-lm.pt")    
#clm_fw = CharLMEmbeddings("/home/liefe/code/sequence-tagger/sequence-tagger/resources/taggers/language_models/fw2/best-lm.pt")  
#clm_bw = CharLMEmbeddings("/home/liefe/code/sequence-tagger/sequence-tagger/resources/taggers/language_models/bw2/best-lm.pt")    

# Load festText word embeddings
#word_embedding = WordEmbeddings("en")
word_embedding = WordEmbeddings("/home/liefe/.flair/embeddings/cc.pt.300.kv")

# Instantiate StackedEmbeddings
#stacked_embedding = StackedEmbeddings(embeddings=[clm_fw, clm_bw])
stacked_embedding = StackedEmbeddings(embeddings=[clm_fw, clm_bw, word_embedding])
#dummy_sent = Sentence("O gato é negro.")
#stacked_embedding.embed(dummy_sent)
#embedding_dim = len(dummy_sent[0].get_embedding())
#print("Embedding dim: ", embedding_dim)

# What tag do we want to predict?
#tag_type = "pos"
tag_type = "ne"

# Make the tag dictionary from the corpus
tag_dict = corpus.make_tag_dictionary(tag_type=tag_type)


## Construct the tagger
#from flair.models import SequenceTagger
#tagger = SequenceTagger(hidden_size=64,
            #embeddings=stacked_embedding,
            #tag_dictionary=tag_dict,
            #tag_type=tag_type,
            #use_crf=True)


#from flair.trainers import SequenceTaggerTrainer

#trainer = SequenceTaggerTrainer(tagger, corpus)

train_data = corpus.train
stacked_embedding.embed(train_data)
for s in corpus.train:
    for t in s:
        print(t, t.get_tag(tag_type), t.embedding)
    
