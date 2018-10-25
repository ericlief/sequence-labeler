# -*- coding: utf-8 -*-


if __name__ == "__main__":
    
    import io
    import sys
    from mypkgs.tokenizer.tokenizer import tokenize
    import argparse    
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=None, type=str, help="file out.")
    parser.add_argument("--min_words", default=5, type=int, help="minimum words per sentence.")
    parser.add_argument("--min_chars", default=20, type=int, help="minimum characters per sentence.")
    parser.add_argument("--max_words", default=100, type=int, help="maximum words per sentence.")
    parser.add_argument("--lc", default=None, type=str, help="Lowercasing.")    
    parser.add_argument("--norm_nums", default=None, type=str, help="Normalize all numbers to 0.")
    parser.add_argument("--min_word_count", default=None, type=int, help="minimum word count for vocabulary.")
    
    args = parser.parse_args()
    #file_in = sys.argv[1]
    #file_out = sys.argv[2]
        
    #print(sys.path)
        
    #sents = []
    stream = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8')
    with open(args.out, 'wt') as f:
        for line in stream:
            if args.lc:
                line = line.lower()
            #print(line)
            tokens = tokenize(line)
            if len(tokens) < args.min_words or len(line) < args.min_chars or len(tokens) > args.max_words:
                continue
                #sents.append(line)
            #print(" ".join(tokens))
            print(" ".join(tokens), file=f)
            
   # print(sents)
