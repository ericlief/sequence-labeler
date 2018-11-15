from flair.data import Sentence
from collections import defaultdict

def print_metrics(totals, totals_per_tag):
    accuracy = (float(totals['tp']) + totals['tn']) / (totals['tp'] + totals['fp'] + totals['tn'] + totals['fn'])
    precision = float(totals['tp']) / (totals['tp'] + totals['fp'])
    recall = float(totals['tp']) / (totals['tp'] + totals['fn'])
    f1 = 2 * precision * recall / (precision + recall)
    print("accuracy = {:.2f}".format(100 * accuracy))
    print("precision = {:.2f}".format(100 * precision))
    print("recall = {:.2f}".format(100 * recall))
    print("f1 = {:.2f}".format(100 * f1))
    print(totals)
    print(totals_per_tag)
    for tag in totals_per_tag:
        #print(tag, end=" ")
        #for cnt in totals_per_tag[tag]:
        try: 
            accuracy = (float(totals_per_tag[tag]['tp']) + totals_per_tag[tag]['tn']) / (totals_per_tag[tag]['tp'] + totals_per_tag[tag]['fp'] + totals_per_tag[tag]['tn'] + totals_per_tag[tag]['fn'])
        except ZeroDivisionError: 
            accuracy = 0
        try: 
            precision = float(totals_per_tag[tag]['tp']) / (totals_per_tag[tag]['tp'] + totals_per_tag[tag]['fp'])
        except ZeroDivisionError: 
            precision = 0
        try: 
            recall = float(totals_per_tag[tag]['tp']) / (totals_per_tag[tag]['tp'] + totals_per_tag[tag]['fn'])
        except ZeroDivisionError: 
            recall = 0
        try:    
            f1 = 2 * precision * recall / (precision + recall)            
        except ZeroDivisionError: 
            f1 = 0
        
        print("{}\tacc {:.3f}\t prec {:.3f}\trec\t{:.3f}\tf1 {:.3f}\n".format(tag, accuracy, precision, recall, f1))    
    
sentence = Sentence("Eu vou te pegar")
sentence[0].add_tag("pos", "pro", 0)
sentence[1].add_tag("pos", "vb", 0)
sentence[2].add_tag("pos", "pro", 0)
sentence[3].add_tag("pos", "vb", 0)
sentence[0].add_tag("predicted", "det", 0)
sentence[1].add_tag("predicted", "vb", 0)
sentence[2].add_tag("predicted", "nn", 0)
sentence[3].add_tag("predicted", "adp", 0)

for t in sentence:
    print(t.get_tag('pos'))

totals_per_tag = defaultdict(lambda: defaultdict(int))
totals = defaultdict(int)
gold_tags = [(tag.tag, str(tag)) for tag in sentence.get_spans('pos')]
predicted_tags = [(tag.tag, str(tag)) for tag in sentence.get_spans('predicted')]
for tag, pred in predicted_tags:
    if (tag, pred) in gold_tags:
        totals['tp'] += 1
        totals_per_tag[tag]['tp'] += 1
    else:
        totals['fp'] +=1
        totals_per_tag[tag]['fp'] += 1

for tag, gold in gold_tags:
    if (tag, gold) not in predicted_tags:
        totals['fn'] += 1
        totals_per_tag[tag]['fn'] += 1  
    else:
        totals['tn'] +=1
        totals_per_tag[tag]['tn'] += 1 # tn?
        
print_metrics(totals, totals_per_tag)