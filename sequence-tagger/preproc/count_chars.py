

if __name__ == "__main__":
    
    #import io
    import sys
    import pickle
    
    #from mypkgs.tokenizer.tokenizer import tokenize
    #import argparse    
    from collections import defaultdict, Counter
    import codecs
    
    file = sys.argv[1]
    #f = """this is a test
    #lets continue
    #this is the last."""
    #char_map = defaultdict(int)
    
    #with open(file, 'r', encoding='utf-8', errors='ignore') as f:
    with codecs.open(file, 'r', encoding='latin1') as f:
        #char_cnt = 0
        #for line in f:
            #line = line.rstrip()
            #words = line.split()
            #for word in words:
                #for ch in word:
                    #char_map[c] += 1
                    #char_cnt += 1
                    
            char_map = Counter(ch for line in f for ch in line.rstrip())
            #print(char_map)    
            total = sum(char_map.values())
            for k,v in char_map.items():
                char_map[k] = v/total
            print(char_map)
            #for k,v in char_map.items():
                #if v <= .000001:
                    #char_map[k] = "<unk>"
            #print(char_map)
            
    with open('char_data.pickle', 'wb') as f:
        pickle.dump(char_map, f, pickle.HIGHEST_PROTOCOL)
        