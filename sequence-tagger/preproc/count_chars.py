

if __name__ == "__main__":
    
    #import io
    import sys
    import pickle
    
    #from mypkgs.tokenizer.tokenizer import tokenize
    #import argparse    
    from collections import defaultdict, Counter
    import codecs
    
    #file = sys.argv[1]
    file = "/home/liefe/code/sequence-tagger/sequence-tagger/xaa"
    
    #f = """this is a test
    #lets continue
    #this is the last."""
    #char_map = defaultdict(int)
    
    with open(file, 'r', encoding='utf-8', errors='ignore') as f:
    #with codecs.open(file, 'r', encoding='latin1') as f:
        char_map = Counter(ch for line in f for ch in line.rstrip())
        #print(char_map)    
        total = sum(char_map.values())
        for k,v in char_map.items():
            char_map[k] = v/total
        print(char_map)
            
            
        #char_cnt = 0
        #for line in f: 
            #words = line.split()
            #for word in words:
                #for ch in word:
                    #char_map[c] += 1
                    #char_cnt += 1
                
        
            #for k,v in char_map.items():
                #if v <= .000001:
                    #char_map[k] = "<unk>"
            #print(char_map)
            
    with open('char_data.pickle', 'wb') as f:
        pickle.dump(char_map, f, pickle.HIGHEST_PROTOCOL)
        
        
    #with open(file, 'r', encoding='utf8', errors='ignore') as f:
        #for line in f:
            #print(line)