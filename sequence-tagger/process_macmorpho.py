def process(fname):
    
    words_tags = []
    with open(fname, "rt") as f:
        for line in f:
            line = line.split()
            words_tags.append(line)
            
    with open(fname[:-3]+"cols.txt", "wt") as f:
        for line in words_tags:
            for word_tag in line:
                word, tag = word_tag.split("_")
                print(word, tag, file=f) 
            print("", file=f)
        
if __name__ == "__main__":
    process("/home/liefe/data/pt/pos/macmorpho/macmorpho-train.txt")
    process("/home/liefe/data/pt/pos/macmorpho/macmorpho-dev.txt")
    process("/home/liefe/data/pt/pos/macmorpho/macmorpho-test.txt")

        


