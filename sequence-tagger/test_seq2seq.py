from flair.data_fetcher import NLPTaskDataFetcher
from flair.embeddings import CharLMEmbeddings

train_fh = "/home/liefe/data/pt/UD_Portuguese-Bosque"
cols = {1:"text", 2:"lemma", 3:"pos"}

corpus = NLPTaskDataFetcher.fetch_column_corpus(train_fh, cols, train_file="train.txt",
                                             dev_file="dev.txt", test_file="test.txt")
print(len(corpus.train))
print(corpus)
print(corpus.train[0].to_tagged_string('pos'))
#for s in sentences:
   #for e in s.get_spans('udep'):
      #print(e)
      

tag_type = "pos"
tag_dict = corpus.make_tag_dictionary(tag_type=tag_type)

print(tag_dict.idx2item)

lm_fw_embed = CharLMEmbeddings("/home/liefe/code/sequence-tagger/sequence-tagger/resources/taggers/language_models/fwd/best-lm.pt")

from flair.models import SequenceTagger

tagger = SequenceTagger(hidden_size=64,
                        embeddings=lm_fw_embed,
                        tag_dictionary=tag_dict,
                        tag_type=tag_type,
                        use_crf=True)

from flair.trainers import SequenceTaggerTrainer

trainer = SequenceTaggerTrainer(tagger, corpus)

trainer.train("resources/taggers/example-pos",
              learning_rate=.1,
              mini_batch_size=32,
              max_epochs=150)

