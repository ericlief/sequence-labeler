from flair.data_fetcher import NLPTaskDataFetcher
from flair.embeddings import CharLMEmbeddings

# 1. get the corpus
train_fh = "/home/liefe/data/pt/UD_Portuguese-Bosque"
cols = {1:"text", 2:"lemma", 3:"pos"}
corpus = NLPTaskDataFetcher.fetch_column_corpus(train_fh, cols, 
                                                train_file="train.txt",
                                                dev_file="dev.txt", 
                                                test_file="test.txt").downsample(.10)

lm_fw_embed = CharLMEmbeddings("/home/liefe/code/sequence-tagger/sequence-tagger/resources/taggers/language_models/fwd/best-lm.pt")
embeded = lm_fw_embed.embed(corpus.train)
for t in corpus.train:
    print(t)
    print(t.embedding)
    
    