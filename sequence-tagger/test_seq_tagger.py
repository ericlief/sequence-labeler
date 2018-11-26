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

#fh = "/home/liefe/data/pt/harem/data/HAREM"
##cols = {1:"text", 2:"lemma", 3:"pos"}
#cols = {0:"text", 1:"ne"}    

tag_type = "mwe"
fh = "/home/liefe/data/pt/mwe"
cols = {1:"text", 2:"lemma", 3:"upos", 4:"xpos", 5:"features", 6:"parent", 7:"deprel", 10:"mwe"}


corpus = NLPTaskDataFetcher.fetch_column_corpus(fh, 
                                                cols, 
                                                train_file="train.txt",
                                                dev_file="dev.txt", 
                                                test_file="test.txt") 


# Load festText word embeddings 
word_embedding = WordEmbeddings("/home/liefe/.flair/embeddings/cc.pt.300.kv")
#word_embedding = WordEmbeddings("/home/lief/files/embeddings/cc.pt.300.kv")

# Load Character Language Models (clms)
clm_fw = CharLMEmbeddings("/home/liefe/lm/fw_p25/best-lm.pt")  
clm_bw = CharLMEmbeddings("/home/liefe/lm/bw_p25/best-lm.pt")    
#clm_fw = CharLMEmbeddings("/home/lief/lm/fw_p25/best-lm.pt")
#clm_bw = CharLMEmbeddings("/home/lief/lm/bw_p25/best-lm.pt")

print("getting embeddings")
# Instantiate StackedEmbeddings
stacked_embedding = StackedEmbeddings(embeddings=[word_embedding, clm_fw, clm_bw])




# What tag do we want to predict?
#tag_type = "pos"
tag_type = "ne"

# Make the tag dictionary from the corpus
tag_dict = corpus.make_tag_dictionary(tag_type=tag_type)


# Construct the tagger
from flair.models import SequenceTagger
tagger = SequenceTagger(hidden_size=256,
            embeddings=stacked_embedding,
            tag_dictionary=tag_dict,
            tag_type=tag_type,
            use_crf=True)


from flair.trainers import SequenceTaggerTrainer

trainer = SequenceTaggerTrainer(tagger, corpus)

trainer.train("resources/taggers/example-mwe",
        learning_rate=.1,
        mini_batch_size=32,
        max_epochs=150,
        embeddings_in_memory=False)
        
        
#train_data = corpus.train
#stacked_embedding.embed(train_data)
#for s in corpus.train:
    #for t in s:
        #print(t, t.get_tag(tag_type), t.embedding)
    
