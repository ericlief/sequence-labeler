from flair.data_fetcher import NLPTaskDataFetcher
from flair.embeddings import CharLMEmbeddings, WordEmbeddings, StackedEmbeddings
from typing import List, Dict
from flair.data import Span, Token, Label
from collections import defaultdict

#def get_spans(sent, tag_type: str, min_score=-1) -> List[Span]:

    #spans: List[Span] = []

    #current_span = []

    #tags = defaultdict(lambda: 0.0)

    #previous_tag_value: str = 'O'
    #for token in sent:

        #tag: Label = token.get_tag(tag_type)
        #tag_value = tag.value

        ## non-set tags are OUT tags
        #if tag_value == '' or tag_value == 'O':
            #tag_value = 'O-'

        ## anything that is not a BIOES tag is a SINGLE tag
        #if tag_value[0:2] not in ['B-', 'I-', 'O-', 'E-', 'S-']:
            #tag_value = 'S-' + tag_value

        ## anything that is not OUT is IN
        #in_span = False
        #if tag_value[0:2] not in ['O-']:
            #in_span = True

        ## single and begin tags start a new span
        #starts_new_span = False
        #if tag_value[0:2] in ['B-', 'S-']:
            #starts_new_span = True

        #if previous_tag_value[0:2] in ['S-'] and previous_tag_value[2:] != tag_value[2:] and in_span:
            #starts_new_span = True

        #if (starts_new_span or not in_span) and len(current_span) > 0:
            #scores = [t.get_tag(tag_type).score for t in current_span]
            #span_score = sum(scores) / len(scores)
            #if span_score > min_score:
                #spans.append(Span(
                    #current_span,
                    #tag=sorted(tags.items(), key=lambda k_v: k_v[1], reverse=True)[0][0],
                    #score=span_score)
                #)
            #current_span = []
            #tags = defaultdict(lambda: 0.0)

        #if in_span:
            #current_span.append(token)
            #weight = 1.1 if starts_new_span else 1.0
            #tags[tag_value[2:]] += weight

        ## remember previous tag
        #previous_tag_value = tag_value

    #if len(current_span) > 0:
        #scores = [t.get_tag(tag_type).score for t in current_span]
        #span_score = sum(scores) / len(scores)
        #if span_score > min_score:
            #spans.append(Span(
                #current_span,
                #tag=sorted(tags.items(), key=lambda k_v: k_v[1], reverse=True)[0][0],
                #score=span_score)
            #)

    #return spans


def get_spans(sent, tag_type: str, min_score=-1) -> List[Span]:

    spans: List[Span] = []

    current_span = []

    tags = defaultdict(lambda: 0.0)

    previous_tag_value: str = 'O'
    for token in sent:

        tag: Label = token.get_tag(tag_type)
        tag_value = tag.value

        # non-set tags are OUT tags
        if tag_value == '' or tag_value == 'O':
            tag_value = 'O-'

        # anything that is not a BIOES tag is a SINGLE tag
        if tag_value[0:2] not in ['B-', 'I-', 'O-', 'E-', 'S-']:
            tag_value = 'S-' + tag_value

        # anything that is not OUT is IN
        in_span = False
        if tag_value[0:2] not in ['O-']:
            in_span = True

        # single and begin tags start a new span
        starts_new_span = False
        if tag_value[0:2] in ['B-', 'S-']:
            starts_new_span = True

        if previous_tag_value[0:2] in ['S-'] and previous_tag_value[2:] != tag_value[2:] and in_span:
            starts_new_span = True

        if (starts_new_span or not in_span) and len(current_span) > 0:
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


#fh = "/home/liefe/data/pt/harem/data/HAREM"
fh = "/home/liefe/data/es/conll2002"
#cols = {1:"text", 2:"lemma", 3:"pos"}
cols = {0:"text", 1:"ne"}     
#cols = {0:"text", 1:"pos", 2:"ne"}     
corpus = NLPTaskDataFetcher.fetch_column_corpus(fh, 
                                                cols, 
                                                train_file="train.txt",
                                                dev_file="dev.txt", 
                                                test_file="test.txt")



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


# Construct the tagger
from flair.models import SequenceTagger
tagger = SequenceTagger(hidden_size=256,
            embeddings=stacked_embedding,
            tag_dictionary=tag_dict,
            tag_type=tag_type,
            use_crf=True)


from flair.trainers import SequenceTaggerTrainer

trainer = SequenceTaggerTrainer(tagger, corpus)

# test embed
#train_data = corpus.train
#stacked_embedding.embed(train_data)
#for s in corpus.train:
    #for t in s:
        #print(t, t.get_tag(tag_type), t.embedding)
 
# test spans        
#for s in corpus.train[:15]:
    #print(s)
    #spans = get_spans(s, tag_type)
    #print(spans)