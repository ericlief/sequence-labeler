from flair.data import Dictionary, Sentence
from flair.models import LanguageModel
from flair.trainers.language_model_trainer import LanguageModelTrainer, TextCorpus
from flair.embeddings import CharacterEmbeddings, CharLMEmbeddings
import numpy as np

# are you training a forward or backward LM?
is_forward_lm = True

# load the default character dictionary
dictionary: Dictionary = Dictionary.load('chars')

print(len(dictionary.get_items()))
print(dictionary.get_items())


# get your corpus, process forward and at the character level
corpus = TextCorpus('/path/to/your/corpus',
    dictionary,
    is_forward_lm,
    character_level=True)

## instantiate your language model, set hidden size and number of layers
#language_model = LanguageModel(dictionary,
        #is_forward_lm,
        #hidden_size=128,
        #nlayers=1)

## train your language model
#trainer = LanguageModelTrainer(language_model, corpus)

#trainer.train('resources/taggers/language_model',
    #sequence_length=10,
    #mini_batch_size=10,
    #max_epochs=10)

 


file = "/home/liefe/py/pt/harem/train_harem_iob2.conll"
with open(file, 'r') as filein:
    in_sent = False 
    sents = []
    tags = []
    sent = ""
    for line in filein:
        if line == "\n":
            sents.append(sent.rstrip())
            #tag 
            sent = ""
            
        else:
            token, tag = line.split()
            sent += token + " "

n = len(sents)
#embedded_sents = np.zeros(shape=[n, 2048])

sentences = []
#print(sents[:20])
for s in sents[:10]:
    sent = Sentence(s)
    sentences.append(sent)
    #clme = CharLMEmbeddings("/home/liefe/tag/best-lm.pt")
    #clme.embed(sent)
    #for t in sent:
        #print(t, t.embedding, t.embedding.size())
    #print(sent.embedding)
    #print(sent.tokens)

#sents = Sentence(sents)
clme = CharLMEmbeddings("/home/liefe/tag/best-lm.pt")
clmes = clme.embed(sentences)

for s in sentences[:10]:
    print(s)
    for t in s:
        print(t, t.embedding, t.embedding.size())
    