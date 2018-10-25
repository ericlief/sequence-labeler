import pickle
import sys

file = sys.argv[1]

with open("char_data.pickle", "rb") as f:
    char_freqs = pickle.load(f)
    print(char_freqs)
    
unk_symbol = 'ů'
with open(file, 'rt', encoding='utf-8') as f:
    with open(file+".unked", 'wt', encoding='utf-8') as out:
        for line in f:
            new_line = ""
            for ch in line.rstrip():
                if char_freqs[ch] <= .00001:
                    new_line += unk_symbol
                else:
                    new_line += ch
            new_line += "\n"
            out.write(new_line)
            
                    
        
