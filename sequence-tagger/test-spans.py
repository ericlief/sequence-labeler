from flair.data_fetcher import NLPTaskDataFetcher
from flair.embeddings import CharLMEmbeddings, WordEmbeddings, StackedEmbeddings
from typing import List, Dict
from flair.data import Span, Token, Label
from collections import defaultdict


def get_spans(sent, tag_type: str, min_score=-1) -> List[Span]:

    spans: List[Span] = []

    current_span = []

    tags = defaultdict(lambda: 0.0)

    previous_tag_value: str = 'O'
    discontin_span = False
    for token in sent:

        tag: Label = token.get_tag(tag_type)
        tag_value = tag.value
        
        print(token, tag)
        
        # non-set tags are OUT tags
        if tag_value == '' or tag_value == 'O':
            tag_value = 'O-'

        # anything that is not a BIOES tag is a SINGLE tag
        if tag_value[0:2] not in ['B-', 'I-', 'O-', 'E-', 'S-', '*']:
            tag_value = 'S-' + tag_value

        # anything that is not OUT is IN
        in_span = False
        if tag_value[0:2] not in ['O-', '*']:
            in_span = True
        print("in span ", in_span)

            
        if previous_tag_value[0:2] in ['B-', 'I-'] and tag_value == '*':
            discontin_span = True
        print("disc span", discontin_span)
        
        # single and begin tags start a new span
        # OK
        starts_new_span = False
        if tag_value[0:2] in ['B-', 'S-']:
            starts_new_span = True
            #start = tag_value[2:] # ?????
        
        # single tag
        if previous_tag_value[0:2] in ['S-'] and previous_tag_value[2:] != tag_value[2:] and in_span:
            starts_new_span = True            
        
        print("new span", starts_new_span)

        if not discontin_span and (starts_new_span or not in_span) and len(current_span) > 0:
            
            print("calc score")
            
            scores = [t.get_tag(tag_type).score for t in current_span]
            span_score = sum(scores) / len(scores)
            if span_score > min_score:
                spans.append(Span(
                        current_span,
                        tag=sorted(tags.items(), key=lambda k_v: k_v[1], reverse=True)[0][0],
                        score=span_score)
                                 )
            current_span = []
            tags = defaultdict(lambda: 0.0)

        if in_span:
            current_span.append(token)
            weight = 1.1 if starts_new_span else 1.0
            tags[tag_value[2:]] += weight
            
            print("added to span")
            
        # for discontinuous MWEs
        if discontin_span and tag_value[0:2] == 'E-':
            discontin_span = False 
        
            print("changed disc span to false")
            
        # remember previous tag
        previous_tag_value = tag_value

    if len(current_span) > 0:
        scores = [t.get_tag(tag_type).score for t in current_span]
        span_score = sum(scores) / len(scores)
        if span_score > min_score:
            spans.append(Span(
                    current_span,
                    tag=sorted(tags.items(), key=lambda k_v: k_v[1], reverse=True)[0][0],
                    score=span_score)
                         )

    return spans


#tag_type = "ne"
#fh = "/home/liefe/data/pt/ner/harem"
#cols = {0:"text", 1:"ne"}     

#fh = "/home/liefe/data/es/conll2002"

#tag_type = "pos"
#fh = "/home/liefe/data/pt/pos/UD_Portuguese-Bosque"
#cols = {1:"text", 2:"lemma", 3:"pos"}

tag_type = "mwe"
fh = "/home/liefe/data/pt/mwe"
cols = {1:"text", 2:"lemma", 3:"upos", 4:"xpos", 5:"features", 6:"parent", 7:"deprel", 10:"mwe"}

corpus = NLPTaskDataFetcher.fetch_column_corpus(fh, 
                                                cols, 
                                                train_file="short2-out.txt",
                                                dev_file="dev.txt", 
                                                test_file="test.txt")


# test spans        
tag_dict = corpus.make_tag_dictionary(tag_type)
print(tag_dict.idx2item)
for s in corpus.train:
    print(s)
    spans = get_spans(s, tag_type)
    print(spans)
    
    
    

## Load Character Language Models (clms)
#clm_fw = CharLMEmbeddings("/home/liefe/lm/fw_p25/best-lm.pt")  
##clm_bw = CharLMEmbeddings("/home/liefe/lm/bw_p25/best-lm.pt")    
##clm_fw = CharLMEmbeddings("/home/liefe/code/sequence-tagger/sequence-tagger/resources/taggers/language_models/fw2/best-lm.pt")  
##clm_bw = CharLMEmbeddings("/home/liefe/code/sequence-tagger/sequence-tagger/resources/taggers/language_models/bw2/best-lm.pt")    

## Load festText word embeddings
##word_embedding = WordEmbeddings("pt")
#word_embedding = WordEmbeddings("/home/liefe/.flair/embeddings/cc.pt.300.kv")

## Instantiate StackedEmbeddings
##stacked_embedding = StackedEmbeddings(embeddings=[clm_fw, clm_bw])
#stacked_embedding = StackedEmbeddings(embeddings=[clm_fw, clm_bw, word_embedding])
##dummy_sent = Sentence("O gato é negro.")
##stacked_embedding.embed(dummy_sent)
##embedding_dim = len(dummy_sent[0].get_embedding())
##print("Embedding dim: ", embedding_dim)

 
## Make the tag dictionary from the corpus
#tag_dict = corpus.make_tag_dictionary(tag_type=tag_type)


## Construct the tagger
#from flair.models import SequenceTagger
#tagger = SequenceTagger(hidden_size=256,
            #embeddings=stacked_embedding,
            #tag_dictionary=tag_dict,
            #tag_type=tag_type,
            #use_crf=True)


#from flair.trainers import SequenceTaggerTrainer

#trainer = SequenceTaggerTrainer(tagger, corpus)

## 7. start training
#trainer.train('resources/taggers/example-ner',
              #learning_rate=0.1,
              #mini_batch_size=32,
              #max_epochs=150,
              #patience=25,
              #embeddings_in_memory=False)

# test embed
#train_data = corpus.train
#stacked_embedding.embed(train_data)
#for s in corpus.train:
    #for t in s:
        #print(t, t.get_tag(tag_type), t.embedding)
 
