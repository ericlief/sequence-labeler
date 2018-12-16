#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
from collections import deque
import random
import torch
from torch._six import inf
from collections import defaultdict


class SequenceTagger:
    def __init__(self, 
                 args,
                 corpus,
                 embedding,
                 tag_type,
                 seed=42,
                 restore_model=False,
                 model_path=None,
                 char_dict_path=None):  
      
        
        self.corpus = corpus # flair corpus type: List of Sentences  
        self.embedding = embedding # flair LM embedding or stacked type
        self.embedding_dim = embedding.embedding_length # the total/concatenated length
        self.tag_type = tag_type # what we're predicting
        self.metrics = Metrics() # for logging metrics
        #self.use_pos_tags = args.use_pos_tags # train with pos tags
        #self.use_lemmas = args.use_lemmas # train with lemmas
        
        # Instantiate scheduler for learning rate annealing
        self.scheduler = ReduceLROnPlateau(args.lr, args.annealing_factor, args.patience) 
        
        # Make the tag dictionary from the corpus
        self.tag_dict = corpus.make_tag_dictionary(tag_type=tag_type)  # id to tag
        n_tags = len(self.tag_dict)
        
        #print(self.tag_dict.idx2item, n_tags)
        #print(corpus.train[0][0].text)
        
        # Make dictionary for pos tags and lemmas if desired
        if args.use_pos_tags:
            self.pos_tag_dict = corpus.make_tag_dictionary("upos")  # id to tag
        if args.use_lemmas:
            self.lemma_dict = corpus.make_tag_dictionary("lemma")  # id to tag
                     
        #print(tag_type, self.tag_dict.idx2item)
        #print(corpus.train)
        
        # Create graph and session
        graph = tf.Graph()
        graph.seed = seed
        self.session = tf.Session(graph = graph, config=tf.ConfigProto(log_device_placement=False))
        
        # Construct graph
        self.construct(args, n_tags, restore_model, model_path, char_dict_path)
        
    def construct(self, args, n_tags, restore_model, model_path, char_dict_path):
        
        with self.session.graph.as_default():

            # Shape = (batch_size, max_sent_len, embedding_dim)
            self.embedded_sents = tf.placeholder(tf.float32, [None, None, self.embedding_dim], name="embedded_sents")
            # Shape = (batch_size, max_sent_len)
            self.char_seq_ids = tf.placeholder(tf.int32, [None, None], name="char_seq_ids")
            # Shape = (num_char_seqs, indef]
            self.char_seqs = tf.placeholder(tf.int32, [None, None], name="char_seqs")
            # Shape = (batch_size)
            self.char_seq_lens = tf.placeholder(tf.int32, [None], name="char_seq_lens")
            # Shape = (batch_size, max_sent_len)
            self.gold_tags = tf.placeholder(tf.int32, [None, None], name="tags")            
            # Trainable params or not
            self.is_training = tf.placeholder(tf.bool, [], name="is_training")
            # Shape = (batch_size)
            self.sentence_lens = tf.placeholder(tf.int32, [None], name="sentence_lens")
  
  
  
            inputs = self.embedded_sents
            if args.use_pos_tags:
                # Shape = (batch_size, max_sent_len)                
                self.pos_tags = tf.placeholder(tf.int32, [None, None], name="tags")
                n_pos_tags = len(self.pos_tag_dict)
                pos_tag_embedding = tf.get_variable("tag_embedding", [n_pos_tags, rnn_dim], tf.float32)
                embedded_pos_tags = tf.nn.embedding_lookup(pos_tag_embedding, self.pos_tags)
                inputs = tf.concat([inputs, embedded_pos_tags], axis=2)
            
            if args.use_lemmas:
                # Shape = (batch_size, max_sent_len)                
                self.lemmas = tf.placeholder(tf.int32, [None, None], name="lemmas")
                n_lemmas = len(self.lemma_dict)
                lemma_embedding = tf.get_variable("lemma_embedding", [n_lemmas, rnn_dim], tf.float32)
                embedded_lemmas = tf.nn.embedding_lookup(pos_tag_embedding, self.pos_tags)
                inputs = tf.concat([inputs, embedded_lemmas], axis=2)
            
            
            # Character embeddings
            if args.use_char_emb:
                with tf.variable_scope('cle'):
                   
                    # Build char dictionary
                    if char_dict_path is None:
                        self.char_dict = Dictionary.load('common-chars')
                    else:
                        self.char_dict = Dictionary.load_from_file(cle_path)    
                    num_chars = len(self.char_dict.idx2item)
                    
         
                    # Generate character embeddings for num_chars of dimensionality cle_dim.
                    char_embeddings = tf.get_variable('char_embeddings', [num_chars, args.cle_dim])
                    
                    # Embed self.chaseqs (list of unique words in the batch) using the character embeddings.
                    embedded_chars = tf.nn.embedding_lookup(char_embeddings, self.char_seqs)
                    print('emb chars', embedded_chars)
                    
                    # TODO: Use `tf.nn.bidirectional_dynamic_rnn` to process embedded self.charseqs using
                    # a GRU cell of dimensionality `args.cle_dim`.
                    # Add locked/variational dropout wrapper
                    # Choose RNN cell according to args.rnn_cell (LSTM and GRU)
                    # if rnn_cell == 'GRU':
                    #     cell_fw = tf.contrib.cudnn_rnn.CudnnCompatibleGRUCell(rnn_dim)
                    #     cell_bw = tf.contrib.cudnn_rnn.CudnnCompatibleGRUCell(rnn_dim)
        
                    # elif rnn_cell == 'LSTM':
                    #     cell_fw = tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(rnn_dim)
                    #     cell_bw = tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(rnn_dim)
        
                    # else: 
                    #     raise Exception("Must select an rnn cell type")     
                    
                    
                    cell_fw = tf.contrib.cudnn_rnn.CudnnCompatibleGRUCell(args.rnn_dim)
                    cell_bw = tf.contrib.cudnn_rnn.CudnnCompatibleGRUCell(args.rnn_dim)
                    
                    cell_fw = tf.nn.rnn_cell.DropoutWrapper(cell_fw, input_keep_prob=1-args.locked_dropout, output_keep_prob=1-args.locked_dropout)
                    cell_bw = tf.nn.rnn_cell.DropoutWrapper(cell_bw, input_keep_prob=1-args.locked_dropout, output_keep_prob=1-args.locked_dropout)
                    
                    # Run cell in limited scope fw and bw
                    # in order to encode char information (subword factors)
                    # The cell is size of char dim, e.g. 32 -> output (?,32)
                    outputs, states = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_fw, 
                                                                      cell_bw=cell_bw, 
                                                                      inputs=embedded_chars, 
                                                                      sequence_length=self.char_seq_lens, 
                                                                      dtype=tf.float32)
                    
                    # Sum the resulting fwd and bwd state to generate character-level word embedding (CLE)
                    # of unique words in the batch 
                    #print(states)
                    #print(states[0], states[1])
                    #cle = states[0] + states[1] 
                    #print('cle', cle)
                    #cle = tf.add(states[0], states[1])
                    #cle = tf.reduce_sum([self.states[0][1], self.states[1][1]], axis=0)
                    #cle = tf.concat([self.states[0][1], self.states[1][1]], axis=-1)
                    cle = tf.reduce_sum(states, axis=0) 
                    #print(cle)
                    emb = tf.nn.embedding_lookup(cle, self.char_seq_ids)
                    #print(emb)
                    # Concatenate the stacked embeddings and the cle (in this order).
                    inputs = tf.concat([inputs, emb], axis=-1)                    
                    
                    ## For kernel sizes of {2..args.cnne_max}, do the following:
                    ## - use `tf.layers.conv1d` on input embedded characters, with given kernel size
                    ##   and `args.cnne_filters`; use `VALID` padding, stride 1 and no activation.
                    ## - perform channel-wise max-pooling over the whole word, generating output
                    ##   of size `args.cnne_filters` for every word.
                    #features = []
                    #for kernel_size in range(2, args.cnne_max + 1):
                        #conv = tf.layers.conv1d(embedded_chars, args.cnne_filters, kernel_size, strides=1, padding='VALID', activation=None, use_bias=False, name='cnne_layer_'+str(kernel_size))
                        ## Apply batch norm
                        #if args.bn:
                            #conv = tf.layers.batch_normalization(conv, training=self.is_training, name='cnn_layer_BN_'+str(kernel_size))
                        #pooling = tf.reduce_max(conv, axis=1)
                        #features.append(pooling)
               
                    ## Concatenate the computed features (in the order of kernel sizes 2..args.cnne_max).
                    ## Consequently, each word from `self.charseqs` is represented using convolutional embedding
                    ## (CNNE) of size `(args.cnne_max-1)*args.cnne_filters`.                
                    #features_concat = tf.concat(features, axis=1)
                    ## Generate CNNEs of all words in the batch by indexing the just computed embeddings
                    ## by self.charseq_ids (using tf.nn.embedding_lookup).
                    #cnne = tf.nn.embedding_lookup(features_concat, self.char_seq_ids)
                    ## Concatenate the word embeddings (computed above) and the CNNE (in this order).
                    #inputs = tf.concat([inputs, cnne], axis=-1)            
           
            # Apply batch normalization
            if args.bn:
                inputs = tf.layers.batch_normalization(inputs, training=self.is_training, name='bn')
                    
            # Normal dropout
            if args.dropout:
                inputs = tf.nn.dropout(inputs, 1-dropout)
            
            # Apply word dropout
            if args.word_dropout:
                inputs = self.word_dropout(inputs, args.word_dropout)
                    
            # Default learning rate
            #self.lr = args.lr
            
            # Choose RNN cell according to args.rnn_cell (LSTM and GRU)
            if args.rnn_cell == 'GRU':
                cell_fw = tf.contrib.cudnn_rnn.CudnnCompatibleGRUCell(args.rnn_dim)
                cell_bw = tf.contrib.cudnn_rnn.CudnnCompatibleGRUCell(args.rnn_dim)

            elif args.rnn_cell == 'LSTM':
                cell_fw = tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(args.rnn_dim)
                cell_bw = tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(args.rnn_dim)

            else: 
                raise Exception("Must select an rnn cell type")     

            # Add locked/variational dropout wrapper
            cell_fw = tf.nn.rnn_cell.DropoutWrapper(cell_fw, input_keep_prob=1-args.locked_dropout, output_keep_prob=1-args.locked_dropout)
            cell_bw = tf.nn.rnn_cell.DropoutWrapper(cell_bw, input_keep_prob=1-args.locked_dropout, output_keep_prob=1-args.locked_dropout)

            # Process embedded inputs with rnn cell
            outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, 
                                                         cell_bw, 
                                                         inputs, 
                                                         sequence_length=self.sentence_lens, 
                                                         dtype=tf.float32,
                                                         time_major=False)

            if args.dropout:
                outputs = tf.nn.dropout(outputs, 1-dropout)
                
            # Concatenate the outputs for fwd and bwd directions (in the third dimension).
            out_concat = tf.concat(outputs, axis=-1)

            # Add a dense layer (without activation) into num_tags classes 
            logits = tf.layers.dense(out_concat, n_tags) 


            # Decoding
            # Generate `weights` as a 1./0. mask of valid/invalid words (using `tf.sequence_mask`).
            weights = tf.sequence_mask(self.sentence_lens, dtype=tf.float32)

            # Use crf for decoding
            if args.use_crf:

                # Compute log likelihood and transition parameters using tf.contrib.crf.crf_log_likelihood
                # and store the mean of sentence losses into `loss`.
                self.transition_params = tf.get_variable("transition_params", [n_tags, n_tags], initializer=tf.glorot_uniform_initializer())
                #self.transition_params = tf.get_variable("transition_params", [n_tags, n_tags], initializer=tf.random_normal_initializer)            
                log_likelihood, self.transition_params = tf.contrib.crf.crf_log_likelihood(inputs=logits, 
                                                                                           tag_indices=self.gold_tags, 
                                                                                           sequence_lengths=self.sentence_lens,
                                                                                           transition_params=self.transition_params)

                self.loss = tf.reduce_mean(-log_likelihood)
                self.reduc_loss = self.loss
                # Compute the CRF predictions into `self.predictions` with `crf_decode`.
                self.predictions, self.scores = tf.contrib.crf.crf_decode(logits, self.transition_params, self.sentence_lens)

            # Use local softmax decoding
            else:             
                self.loss = tf.losses.sparse_softmax_cross_entropy(labels=self.gold_tags, logits=logits, weights=weights)
                self.reduc_loss = tf.reduce_mean(self.loss)  

                # Generate `self.predictions`.
                self.predictions = tf.argmax(logits, axis=-1) # 3rd dim!                

            global_step = tf.train.create_global_step()            
  
            # Choose optimizer                                              
            if args.optimizer == "SGD" and args.momentum:
                optimizer = tf.train.MomentumOptimizer(learning_rate=self.scheduler.lr, momentum=args.momentum) 
                #self.training = tf.train.GradientDescentOptimizer(learning_rate) 
            elif args.optimizer == "SGD":
                optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.scheduler.lr) 
            else:                
                optimizer = tf.train.AdamOptimizer(learning_rate=self.scheduler.lr) 

            # Note how instead of `optimizer.minimize` we first get the # gradients using
            # `optimizer.compute_gradients`, then optionally clip them and
            # finally apply then using `optimizer.apply_gradients`.
            gradients, variables = zip(*optimizer.compute_gradients(self.loss))
            # Compute norm of gradients using `tf.global_norm` into `gradient_norm`.
            gradient_norm = tf.global_norm(gradients) 
            
            # If args.clip_gradient, clip gradients (back into `gradients`) using `tf.clip_by_global_norm`.            
            if args.clip_gradient is not None:
                gradients, _ = tf.clip_by_global_norm(gradients, clip_norm=args.clip_gradient, use_norm=gradient_norm)
            
            # Update ops for bn
            if args.bn:
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                with tf.control_dependencies(update_ops):     
                    self.training = optimizer.apply_gradients(zip(gradients, variables), global_step=global_step)
            else:
                self.training = optimizer.apply_gradients(zip(gradients, variables), global_step=global_step)
                
            # Summaries
            self.current_accuracy, self.update_accuracy = tf.metrics.accuracy(self.gold_tags, self.predictions, weights=weights)
            self.current_loss, self.update_loss = tf.metrics.mean(self.loss, weights=tf.reduce_sum(weights))
            self.reset_metrics = tf.variables_initializer(tf.get_collection(tf.GraphKeys.METRIC_VARIABLES))

            summary_writer = tf.contrib.summary.create_file_writer(logdir, flush_millis=10 * 1000)
            self.summaries = {}
            with summary_writer.as_default(), tf.contrib.summary.record_summaries_every_n_global_steps(10):
                self.summaries["train"] = [tf.contrib.summary.scalar("train/loss", self.update_loss),
                                           tf.contrib.summary.scalar("train/gradient_norm", gradient_norm),
                                           tf.contrib.summary.scalar("train/accuracy", self.update_accuracy)]

            with summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
                for dataset in ["dev", "test"]:
                    self.summaries[dataset] = [tf.contrib.summary.scalar(dataset + "/loss", self.current_loss),
                                               tf.contrib.summary.scalar(dataset + "/accuracy", self.current_accuracy)]
                                               
            # To save model
            self.saver = tf.train.Saver()
            
            if restore_model:
                self.saver.restore(self.session, model_path)  # restore model
                print("Restoring model from ", model_path)
            else:
                self.session.run(tf.global_variables_initializer())                 # initialize variables
            
            with summary_writer.as_default():
                tf.contrib.summary.initialize(session=self.session, graph=self.session.graph)


    def train(self, 
              args,
              embeddings_in_memory=False,
              checkpoint=False): 
                    
        # Reset batch metrics
        self.session.run(self.reset_metrics)
        
        # Train epochs
        train_data = corpus.train  
        dev_score = 0
        for epoch in range(args.epochs):
            
            # Stop if lr gets to small
            if self.scheduler.lr < args.final_lr:
                print("Learning rate has become to small. Exiting training: lr=", self.scheduler.lr)
                break    
            
            # Shuffle data and form batches
            random.shuffle(train_data)
            batches = [train_data[i:i + args.batch_size] for i in range(0, len(train_data), args.batch_size)]        
            
            # To store metrics
            totals_per_tag = defaultdict(lambda: defaultdict(int))
            totals = defaultdict(int)            
            
            for batch_n, batch in enumerate(batches):
                
                # Sort batch and get lengths
                batch.sort(key=lambda i: len(i), reverse=True)
                
                # Remove super long sentences                
                
                #print("max sent len", max_sent_len)
                max_sent_len = len(batch[0])                                    
                while len(batch) > 1 and max_sent_len > 100:
                    #print("removing long sentence")
                    batch = batch[1:]
                    max_sent_len = len(batch[0])                    
                    
                sent_lens = [len(s.tokens) for s in batch]
                n_sents = len(sent_lens)     
                
                #print("embedding sents")
                # Embed sentences using flair embeddings
                self.embedding.embed(batch)            
                
                # Pad sentences and tags
                embedded_sents = np.zeros([n_sents, max_sent_len, self.embedding_dim])
                gold_tags = np.zeros([n_sents, max_sent_len]) 
                
                if args.use_pos_tags:
                    pos_tags = np.zeros([n_sents, max_sent_len]) 
                if args.use_lemmas:
                    lemmas = np.zeros([n_sents, max_sent_len]) 
                if args.use_char_emb:
                    char_seq_map = {'<pad>': 0}
                    char_seqs = [[char_seq_map['<pad>']]]
                    char_seq_ids = []                       
                
                for i in range(n_sents):
                    ids = np.zeros([max_sent_len])

                    # Next sentence                    
                    for j in range(sent_lens[i]):                    
                        token = batch[i][j]
                        
                        #print(token.text, len(token.text))
                        
                        embedded_sents[i, j] = token.embedding.numpy() # convert torch tensor to numpy array
                        if args.use_pos_tags:
                            pos_tags[i, j] = self.pos_tag_dict.get_idx_for_item(token.get_tag("upos").value)
                        if args.use_lemmas:
                            lemmas[i, j] = self.lemma_dict.get_idx_for_item(token.get_tag("lemma").value)                            
                        if args.use_char_emb:
                            if token.text not in char_seq_map:
                                char_seq = [self.char_dict.get_idx_for_item(c) for c in token.text]
                                char_seq_map[token.text] = len(char_seqs)
                                char_seqs.append(char_seq)                        

                            ids[j] = char_seq_map[token.text]     
                        gold_tags[i, j] = self.tag_dict.get_idx_for_item(token.get_tag(self.tag_type).value)      # tag index         
                        
                    
                    # Append sentence char_seq_ids
                    if args.use_char_emb:
                        char_seq_ids.append(ids)
                        
                # Run graph for batch
                
                feed_dict = {self.sentence_lens: sent_lens,
                             self.embedded_sents: embedded_sents,
                             self.gold_tags: gold_tags, 
                             self.is_training: True}                              
                
                
                # Pad char sequences                
                if args.use_char_emb:
                    #char_seqs.sort(key=lambda i: len(i), reverse=True)
                    char_seq_lens = [len(char_seq) for char_seq in char_seqs]
                    #max_char_seq = len(char_seqs[0])
                    max_char_seq = max(char_seq_lens)
                    n = len(char_seqs)
                    batch_char_seqs = np.zeros([n, max_char_seq])
                    for i in range(n):
                        batch_char_seqs[i, 0:len(char_seqs[i])] = char_seqs[i]
                    
                    feed_dict[self.char_seqs] = batch_char_seqs
                    feed_dict[self.char_seq_ids] = char_seq_ids
                    feed_dict[self.char_seq_lens] = char_seq_lens                
                
                
                if args.use_pos_tags:
                    feed_dict[self.pos_tags] = pos_tags
                    
                    #feed_dict = {self.sentence_lens: sent_lens,
                                 #self.embedded_sents: embedded_sents,
                                 #self.gold_tags: gold_tags, 
                                 #self.is_training: True})                    
                    
                if args.use_lemmas:
                    feed_dict[self.lemmas] = lemmas
       
                    
                #_, _, loss, predicted_tag_ids = self.session.run([self.training, self.summaries["train"], self.loss, self.predictions],
                                 #{self.sentence_lens: sent_lens,
                                  #self.embedded_sents: embedded_sents,
                                  #self.pos_tags: pos_tags,
                                  #self.gold_tags: gold_tags, 
                                  #self.is_training: True})
           
                _, _, loss, predicted_tag_ids = self.session.run([self.training, self.summaries["train"], self.loss, self.predictions],
                                 feed_dict)
                    
                #print("annotating tags")
                # DEBUG: Add predicted tag to each token (annotate)    
                for i in range(n_sents):
                    for j in range(sent_lens[i]):
                        token = batch[i][j]
                        predicted_tag = self.tag_dict.get_item_for_index(predicted_tag_ids[i][j])
                        token.add_tag('predicted', predicted_tag)
                        #print(token, predicted_tag)
    
                # Tally metrics        
                for sentence in batch:
                    gold_tags = [(tag.tag, str(tag)) for tag in sentence.get_spans(self.tag_type)]
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
                 
                #self.metrics.log_metrics("train", totals, totals_per_tag, epoch, batch_n)                
                self.metrics.log_metrics("train", totals, totals_per_tag, epoch, batch_n, self.scheduler.lr, self.scheduler.bad_epochs, dev_score)                
                
                if not embeddings_in_memory:
                    self.clear_embeddings_in_batch(batch)                
                
            # Save model if checkpoints enabled
            if checkpoint:
                save_path = self.saver.save(self.session, "{}/checkpoint.ckpt".format(logdir))
                print("Checkpoint saved at ", save_path)
           
            # Evaluate with dev data
            dev_data = corpus.dev                
            #dev_score = self.evaluate("dev", dev_data, dev_batch_size, epoch, embeddings_in_memory=embeddings_in_memory)
            dev_score = self.evaluate(args, "dev", dev_data, epoch, embeddings_in_memory=embeddings_in_memory)
             
            # Perform one step on lr scheduler
            is_reduced = self.scheduler.step(dev_score)
            #if is_reduced:
                #self.lr = self.scheduler.lr
                
            #print("Epoch {} batch {}: train loss \t{}\t lr \t{}\t dev score \t{}\t bad epochs \t{}".format(epoch, batch_n, loss, self.lr, dev_score, self.scheduler.bad_epochs))        
            print("Epoch {} batch {}: train loss \t{}\t lr \t{}\t dev score \t{}\t bad epochs \t{}".format(epoch, batch_n, loss, self.scheduler.lr, dev_score, self.scheduler.bad_epochs))        
            
            # Save best model
            #if dev_score == self.scheduler.best:
                #save_path = self.saver.save(self.session, "{}/best-model.ckpt".format(logdir))
                #print("Best model saved at ", save_path)
            if dev_score == self.scheduler.best:
                save_path = self.saver.save(self.session, "{}/best-model.ckpt".format(logdir))
                print("Best model saved at ", save_path) 

    def evaluate(self, args, dataset_name, dataset, epoch=None, test_mode=False, embeddings_in_memory=False, metric="accuracy"):                
    #def evaluate(self, dataset_name, dataset, eval_batch_size=32, epoch=None, test_mode=False, embeddings_in_memory=False, metric="accuracy"):
        
        #eval_batch_size = args.eval_batch_size
        
        print("evaluating")
        
        self.session.run(self.reset_metrics)  # for batch statistics
        # Get batches
        batches = [dataset[x:x + args.eval_batch_size] for x in range(0, len(dataset), args.eval_batch_size)]
        
        # To store metrics
        totals_per_tag = defaultdict(lambda: defaultdict(int))
        totals = defaultdict(int)
        dev_loss = 0
        
        # Forward pass for each batch
        for batch_n, batch in enumerate(batches):
            
            # Sort batch and get lengths
            batch.sort(key=lambda i: len(i), reverse=True)
            
            # Remove super long sentences                
            max_sent_len = len(batch[0])
            #print("max sent len", max_sent_len)
            while len(batch) > 1 and max_sent_len > 200:
                #print("removing long sentence")
                batch = batch[1:]
                max_sent_len = len(batch[0])                    
                
            sent_lens = [len(s.tokens) for s in batch]
            n_sents = len(sent_lens)             
            
            # Embed sentences using flair embeddings
            self.embedding.embed(batch)            
            
            # Pad sentences and tags
            embedded_sents = np.zeros([n_sents, max_sent_len, self.embedding_dim])
            gold_tags = np.zeros([n_sents, max_sent_len]) 
            
            if args.use_pos_tags:
                pos_tags = np.zeros([n_sents, max_sent_len]) 
            if args.use_lemmas:
                lemmas = np.zeros([n_sents, max_sent_len]) 
            if args.use_char_emb:
                char_seq_map = {'<pad>': 0}
                char_seqs = [[char_seq_map['<pad>']]]
                char_seq_ids = []    
                
            for i in range(n_sents):
                ids = np.zeros([max_sent_len])

                # Sentence
                for j in range(sent_lens[i]):                    
                    token = batch[i][j] 
                    embedded_sents[i, j] = token.embedding.numpy() # convert torch tensor to numpy array
                    if args.use_pos_tags:
                        pos_tags[i, j] = self.pos_tag_dict.get_idx_for_item(token.get_tag("upos").value)
                    if args.use_lemmas:
                        lemmas[i, j] = self.lemma_dict.get_idx_for_item(token.get_tag("lemma").value)                          
                    if args.use_char_emb:
                        if token.text not in char_seq_map:
                            char_seq = [self.char_dict.get_idx_for_item(c) for c in token.text]
                            char_seq_map[token.text] = len(char_seqs)
                            char_seqs.append(char_seq)                    
                        ids[j] = char_seq_map[token.text]                            

                    gold_tags[i, j] = self.tag_dict.get_idx_for_item(token.get_tag(self.tag_type).value)      # tag index         
             
                # Append sentence char_seq_ids
                if args.use_char_emb:
                    char_seq_ids.append(ids)        
                    
            feed_dict = {self.sentence_lens: sent_lens,
                         self.embedded_sents: embedded_sents,
                         self.is_training: False}                              
            
            # Pad char sequences            
            if args.use_char_emb:
                #char_seqs.sort(key=lambda i: len(i), reverse=True)
                char_seq_lens = [len(char_seq) for char_seq in char_seqs]
                #max_char_seq = len(char_seqs[0])
                max_char_seq = max(char_seq_lens)
                n = len(char_seqs)
                batch_char_seqs = np.zeros([n, max_char_seq])
                for i in range(n):
                    batch_char_seqs[i, 0:len(char_seqs[i])] = char_seqs[i]
                
                feed_dict[self.char_seqs] = batch_char_seqs
                feed_dict[self.char_seq_ids] = char_seq_ids
                feed_dict[self.char_seq_lens] = char_seq_lens           
           
            if args.use_pos_tags:
                feed_dict[self.pos_tags] = pos_tags                              
            if args.use_lemmas:
                feed_dict[self.lemmas] = lemmas
   
            
            # For dev data
            if not test_mode:
                feed_dict[self.gold_tags] = gold_tags
                _, _, dev_loss, predicted_tag_ids =  self.session.run([self.update_accuracy, 
                                                                       self.update_loss, 
                                                                       self.current_loss, 
                                                                       self.predictions],
                                                                      feed_dict)   
             
                print("dev loss ", dev_loss)
            
            # For test data
            else:
                predicted_tag_ids = self.session.run(self.predictions,
                                                     feed_dict)                
                                       
            # Add predicted tag to each token (annotate)    
            for i in range(n_sents):
                for j in range(sent_lens[i]):
                    token = batch[i][j]
                    predicted_tag = self.tag_dict.get_item_for_index(predicted_tag_ids[i][j])
                    token.add_tag('predicted', predicted_tag)
           
            # Tally metrics        
            for sentence in batch:
                gold_tags = [(tag.tag, str(tag)) for tag in sentence.get_spans(self.tag_type)]
                predicted_tags = [(tag.tag, str(tag)) for tag in sentence.get_spans('predicted')]
                
                
                #print("pred id len", len(predicted_tags))
                #print("gold id len", len(gold_tags))
                
                
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
             
            self.metrics.log_metrics(dataset_name, totals, totals_per_tag, epoch, batch_n, self.scheduler.lr, self.scheduler.bad_epochs)

            
            if not embeddings_in_memory:
                self.clear_embeddings_in_batch(batch)             
                        
        # Write test results
        if test_mode:                
            with open("{}/tagger_tag_test.txt".format(logdir), "w") as test_file:
                for i in range(len(batches)):
                    for j in range(len(batches[i])): 
                        for k in range(len(batches[i][j])):
                            token = batches[i][j][k]
                            gold_tag = token.get_tag(self.tag_type).value
                            predicted_tag = token.get_tag('predicted').value
                            print("{} {} {}".format(token.text, gold_tag, predicted_tag), file=test_file)
                        print("", file=test_file)
            return
        
        # Save and print metrics                  
        #self.metrics.log_metrics(dataset_name, totals, totals_per_tag, batch_n)
        self.metrics.print_metrics()
        
        if metric == "accuracy":
            return self.metrics.accuracy
        
        else:
            return self.metrics.f1
    
    def clear_embeddings_in_batch(self, batch):
        for sentence in batch:
            for token in sentence.tokens:
                token.clear_embeddings()      


    def word_dropout(self, x, dropout_rate):
        mask = tf.distributions.Bernoulli(1 - dropout_rate, 
                                          dtype=tf.float32).sample([tf.shape(x)[0], tf.shape(x)[1], 1])                                        
                
        return mask * x        


                
               
class Metrics:
    """Helper class to calculate and store metrics"""
    
    def log_metrics(self, dataset_name, totals, totals_per_tag, epoch, batch_n, lr=0, bad_epochs=0, dev_score=0):
        
        self.totals = totals
        self.totals_per_tag = totals_per_tag
        
        with open("{}/metrics-{}.txt".format(logdir, dataset_name), "a") as f:
           
            # Total metrics
            try: 
                self.accuracy = (float(totals['tp']) + totals['tn']) / (totals['tp'] + totals['fp'] + totals['tn'] + totals['fn'])
            except ZeroDivisionError: 
                self.accuracy = 0
            try: 
                self.precision = float(totals['tp']) / (totals['tp'] + totals['fp'])
            except ZeroDivisionError: 
                self.precision = 0             
            try: 
                self.recall = float(totals['tp']) / (totals['tp'] + totals['fn'])
            except ZeroDivisionError: 
                self.recall = 0            
            try:    
                self.f1 = 2 * self.precision * self.recall / (self.precision + self.recall)           
            except ZeroDivisionError: 
                self.f1 = 0            
            # Save
            f.write("\nEpoch {}\t Batch {}\t lr {}\tBad epochs {}\t Dev score {:.3f} \n".format(epoch, batch_n, lr, bad_epochs, dev_score ))                        
            f.write("tp {}\t fp {}\t tn {}\t fn {}\t acc {:.3f}\t prec {:.3f}\trec\t{:.3f}\tf1 {:.3f}\n".format(totals['tp'], totals['fp'], totals['tn'], totals['fn'], self.accuracy, self.precision, self.recall, self.f1))            
           
            # Metrics per tag
            for tag in totals_per_tag:
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
                # Save
                f.write("{}\tacc {:.3f}\t prec {:.3f}\trec\t{:.3f}\tf1 {:.3f}\n".format(tag, accuracy, precision, recall, f1))            
     
    def print_metrics(self):
        print("acc {:.3f}\t prec {:.3f}\trec\t{:.3f}\tf1 {:.3f}\n".format(self.accuracy, self.precision, self.recall, self.f1))            


class ReduceLROnPlateau:
    """Reduce learning rate when a the loss has stopped improving.
        Models often benefit from reducing the learning rate by a factor
        of 2-10 once learning stagnates. This scheduler determines   
        whether to reduce the current learning rate if no change has
        been seen for a given number of epochs (the patience)
        Args:
            factor (float): Factor by which the learning rate will be
                reduced. new_lr = lr * factor. Default: 0.5.
            patience (int): Number of epochs with no improvement after
                which learning rate will be reduced. For example, if
                `patience = 5`, the first 5 epochs
                with no improvement will be ignored, and the lr will be
                reduced in the sixth epoch if the loss still hasn't improved then.
                Default: 10.
            verbose (bool): If ``True``, prints a message to stdout for
                each update. Default: ``False``.
            threshold (float): Threshold for measuring the new optimum,
                to only focus on significant changes. Default: 1e-4.
            threshold_mode (str): One of `rel`, `abs`. In `rel` mode,
                dynamic_threshold = best * ( 1 + threshold ) in 'max'
                mode or best * ( 1 - threshold ) in `min` mode.
                In `abs` mode, dynamic_threshold = best + threshold in
                `max` mode or best - threshold in `min` mode. Default: 'rel'.
            eps (float): Minimal decay applied to lr. If the difference
                between new and old lr is smaller than eps, the update is
                ignored. Default: 1e-8.

            """
    def __init__(self, lr, factor=.5, patience=5, threshold=1e-4, threshold_mode="rel", eps=1e-8):
        self.lr = lr
        self.factor = factor
        self.patience = patience
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.eps = eps
        self.bad_epochs = 0
        self.last_epoch = -1
        self.best = None
        self.reset()

    def reset(self):
        self.bad_epochs = 0
        self.best = -inf

    def step(self, metric, epoch=None):
        epoch = self.last_epoch = self.last_epoch + 1
        cur = metric
        if self.is_better(cur, self.best):
            self.best = cur
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1

        if self.bad_epochs > self.patience:
            self.bad_epochs = 0
            return self.reduce_lr(epoch)
        else:
            return False

    def is_better(self, cur, best):
        if self.threshold_mode == "rel":
            eps = 1.0 - self.threshold
            return cur > best * eps
        elif self.threshold_mode == "abs":
            return cur > best - eps

    def reduce_lr(self, epoch):
        old_lr = self.lr
        new_lr = self.factor * old_lr
        if old_lr - new_lr > self.eps:
            self.lr = new_lr
            print("Epoch {:5d}: reducing learning rate from {} to {:.4e}.".format(epoch, old_lr, new_lr))                
            return True
        else:
            return False             
        
       
if __name__ == "__main__":
    import argparse
    import datetime
    import os
    import re
    from flair.data_fetcher import NLPTaskDataFetcher
    from flair.embeddings import CharLMEmbeddings, WordEmbeddings, StackedEmbeddings
    from flair.data import Sentence, Dictionary
    import numpy as np
    
    # Fix random seed
    np.random.seed(42)


    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size.")
    parser.add_argument("--eval_batch_size", default=32, type=int, help="Batch size for dev.")
    parser.add_argument("--rnn_cell", default="LSTM", type=str, help="RNN cell type.")
    parser.add_argument("--rnn_dim", default=256, type=int, help="RNN cell dimension.")    
    parser.add_argument("--optimizer", default="SGD", type=str, help="Optimizer.")    
    parser.add_argument("--momentum", default=None, type=float, help="Momentum.")    
    parser.add_argument("--cle_dim", default=16, type=int, help="Character-level embedding dimension.")
    parser.add_argument("--cnne_filters", default=64, type=int, help="CNN embedding filters per length.")
    parser.add_argument("--cnne_max", default=8, type=int, help="Maximum CNN filter length.")
    parser.add_argument("--epochs", default=150, type=int, help="Number of epochs.")
    parser.add_argument("--lr", default=0.1, type=float, help="Initial learning rate.") 
    parser.add_argument("--final_lr", default=.001, type=float, help="Final learning rate.")    
    parser.add_argument("--dropout", default=0, type=float, help="Dropout rate.")
    parser.add_argument("--locked_dropout", default=.5, type=float, help="Locked/Variational dropout rate.")
    parser.add_argument("--word_dropout", default=.05, type=float, help="Word dropout rate.")
    parser.add_argument("--use_crf", default=False, type=bool, help="Conditional random field decoder.")
    parser.add_argument("--use_pos_tags", default=False, type=bool, help="PoS tag embeddings.")
    parser.add_argument("--use_lemmas", default=False, type=bool, help="Lemma embeddings.")
    parser.add_argument("--use_word_emb", default=False, type=bool, help="Pretrained word embeddings.")        
    parser.add_argument("--use_char_emb", default=False, type=bool, help="Character level embeddings.")
    parser.add_argument("--use_lm", default=False, type=bool, help="CharLM embeddings.")
    parser.add_argument("--clip_gradient", default=.25, type=float, help="Norm for gradient clipping.")
    parser.add_argument("--layers", default=1, type=int, help="Number of rnn layers.")
    parser.add_argument("--annealing_factor", default=.5, type=float, help="Patience for lr schedule.")    
    parser.add_argument("--patience", default=20, type=int, help="Patience for lr schedule.")    
    parser.add_argument("--bn", default=False, type=bool, help="Batch normalization.")
    parser.add_argument("--tag_type", default=None, type=str, help="Task.")
    
    args = parser.parse_args()

    # Create logdir name  
    logdir = "logs/{}-{}-{}".format(
    #logdir = "/home/lief/files/tagger/logs/{}-{}-{}".format(
        os.path.basename(__file__),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items())))
    )
    ## Create logdir name
    ##logdir = "logs/{}-{}".format(
    #logdir = "/home/lief/files/tagger/logs/{}-{}".format(
    #os.path.basename(__file__),
        #datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"))
    
    # Get the corpus
    
    #tag_type = "pos"                                                                                                                       
    #fh = "/home/liefe/data/pt/UD_Portuguese-Bosque"  # pos                                                             
    #cols = {1:"text", 2:"lemma", 3:"pos"
 
    #tag_type = "pos"
    #fh = "/home/liefe/data/pt/pos/macmorpho1"
    #fh = "/home/lief/files/data/pt/pos/macmorpho1"
    #cols = {0:"text", 1:"pos"}
    
                                                                                                   
    tag_type = "ne"
    fh = "/home/liefe/data/pt/ner/harem" # ner
    #fh = "/home/lief/files/data/pt/ner/harem" # ner                                                                                       0
    cols = {0:"text", 1:"ne"}    
    


    #tag_type = "mwe"
    #fh = "/home/liefe/data/pt/mwe"
    #fh = "/home/lief/files/data/pt/mwe" 
    #cols = {1:"text", 2:"lemma", 3:"upos", 4:"xpos", 5:"features", 6:"parent", 7:"deprel", 10:"mwe"}
    #cols = {1:"text", 2:"lemma", 3:"upos", 10:"mwe"}
    
    # Fetch corpus
    print("Getting corpus")
    corpus = NLPTaskDataFetcher.fetch_column_corpus(fh, 
                                                    cols, 
                                                    train_file="train.txt",
                                                    dev_file="dev.txt", 
                                                    test_file="test.txt")
    

    # if args.use_word_emb:
    #     # Load festText word embeddings 
    #     #word_embedding = WordEmbeddings("/home/liefe/.flair/embeddings/cc.pt.300.kv")
    #     word_embedding = WordEmbeddings("/home/lief/files/embeddings/cc.pt.300.kv")
    
    # # Load Character Language Models (clms)
    # #clm_fw = CharLMEmbeddings("/home/liefe/lm/fw/best-lm.pt", use_cache=True, cache_directory="/home/liefe/tag/cache/pos")  
    # #clm_bw = CharLMEmbeddings("/home/liefe/lm/bw/best-lm.pt", use_cache=True, cache_directory="/home/liefe/tag/cache/pos")    
    # clm_fw = CharLMEmbeddings("/home/lief/lm/fw/best-lm.pt")
    # clm_bw = CharLMEmbeddings("/home/lief/lm/bw/best-lm.pt")
    
    # # Instantiate StackedEmbeddings
    # print("Getting embeddings")    
    # #stacked_embeddings = StackedEmbeddings(embeddings=[clm_fw, clm_bw])
    # stacked_embeddings = StackedEmbeddings(embeddings=[word_embedding, clm_fw, clm_bw])
      # Load festText word embeddings     

    embeddings = []
    if args.use_word_emb:
        embeddings.append(WordEmbeddings("/home/liefe/.flair/embeddings/cc.pt.300.kv"))
        #embeddings.append(WordEmbeddings("/home/lief/files/embeddings/cc.pt.300.kv"))
        
    # Load Character Language Models (clms)
    
    #embeddings.append(CharLMEmbeddings("/home/lief/lm/fw/best-lm.pt", use_cache=True, cache_directory="/home/lief/files/embeddings/cache/pos"))
    #embeddings.append(CharLMEmbeddings("/home/lief/lm/bw/best-lm.pt", use_cache=True, cache_directory="/home/lief/files/embeddings/cache/pos"))
    if args.use_lm:
        #embeddings.append(CharLMEmbeddings("/home/lief/files/language_models-backup/fw_p25/best-lm.pt", use_cache=False))
        #embeddings.append(CharLMEmbeddings("/home/lief/files/language_models-backup/bw_p25/best-lm.pt", use_cache=False))
        #embeddings.append(CharLMEmbeddings("/home/lief/lm/fw/best-lm.pt", use_cache=False))
        #embeddings.append(CharLMEmbeddings("/home/lief/lm/bw/best-lm.pt", use_cache=False))
        embeddings.append(CharLMEmbeddings("/home/liefe/lm/fw/best-lm.pt", use_cache=True, cache_directory="/home/liefe/tag/cache/pos"))
        embeddings.append(CharLMEmbeddings("/home/liefe/lm/bw/best-lm.pt", use_cache=True, cache_directory="/home/liefe/tag/cache/pos"))        
    
    # Instantiate StackedEmbeddings
    print("Stacking embeddings")    
    stacked_embeddings = StackedEmbeddings(embeddings)
    
    # Construct the tagger
    print("Constructing tagger")
    #path = "/home/liefe/tag/logs/mwe-150-16-16-20-pos/best-model.ckpt"
    #tagger = SequenceTagger(corpus, stacked_embeddings, tag_type, restore_model=True, model_path=path)
    #tagger = SequenceTagger(corpus, stacked_embeddings, tag_type, dropout=0, locked_dropout=.5, word_dropout=.05, use_lemmas=False, use_pos_tags=False)
    tagger = SequenceTagger(args, corpus, stacked_embeddings, tag_type)
    
     
    # Train
    print("Beginning training") 
    tagger.train(args, checkpoint=True, embeddings_in_memory=True)   
     
    # Test 
    test_data = corpus.test
    tagger.evaluate(args, "test",  test_data, test_mode=True, embeddings_in_memory=True)

    ## Train
    #print("Beginning training")    
    #tagger.train(epochs=150, batch_size=32, dev_batch_size=32, patience=10, checkpoint=True, embeddings_in_memory=True)   
     
    ## Test 
    #test_data = corpus.test
    #tagger.evaluate("test", test_data, test_mode=True, embeddings_in_memory=True)
