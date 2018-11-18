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
                 corpus,
                 embedding,
                 tag_type,
                 rnn_cell="LSTM",    
                 rnn_dim=256,
                 optimizer="SGD", 
                 momentum=False,
                 dropout=.5,
                 clip_gradient=.25,
                 use_crf=True,
                 threads=1, 
                 seed=42):                  
        
        self.corpus = corpus
        self.embedding = embedding
        self.embedding_dim = embedding.embedding_length
        self.tag_type = tag_type # what we're predicting
        self.metrics = Metrics() # for logging metrics
        
        # Make the tag dictionary from the corpus
        self.tag_dict = corpus.make_tag_dictionary(tag_type=tag_type)
        n_tags = len(self.tag_dict)
        #print(tag_type, self.tag_dict.idx2item)
        #print(corpus.train)
        
        # Create graph and session
        graph = tf.Graph()
        graph.seed = seed
        self.session = tf.Session(graph = graph, config=tf.ConfigProto(log_device_placement=False))
        
        # Construct graph
        self.construct(rnn_cell, rnn_dim, optimizer, momentum, dropout, clip_gradient, n_tags, use_crf)
        
    def construct(self, rnn_cell, rnn_dim, optimizer, momentum, dropout, clip_gradient, n_tags, use_crf):
        
        with self.session.graph.as_default():

            # Shape = (batch_size, max_sent_len, embedding_dim)
            self.embedded_sents = tf.placeholder(tf.float32, [None, None, self.embedding_dim], name="embedded_sents")
            # Shape = (batch_size, max_sent_len)
            self.tags = tf.placeholder(tf.int32, [None, None], name="tags")
            # Trainable params or not
            self.is_training = tf.placeholder(tf.bool, [], name="is_training")
            # Shape = (batch_size)
            self.sentence_lens = tf.placeholder(tf.int32, [None], name="sentence_lens")
            # Default learning rate
            self.lr = .1
            
            # Choose RNN cell according to args.rnn_cell (LSTM and GRU)
            if rnn_cell == 'GRU':
                cell_fw = tf.contrib.cudnn_rnn.CudnnCompatibleGRUCell(rnn_dim)
                cell_bw = tf.contrib.cudnn_rnn.CudnnCompatibleGRUCell(rnn_dim)

            elif rnn_cell == 'LSTM':
                cell_fw = tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(rnn_dim)
                cell_bw = tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(rnn_dim)

            else: 
                raise Exception("Must select an rnn cell type")     

            # Add dropout
            cell_fw = tf.nn.rnn_cell.DropoutWrapper(cell_fw, input_keep_prob=1-dropout, output_keep_prob=1-dropout)
            cell_bw = tf.nn.rnn_cell.DropoutWrapper(cell_bw, input_keep_prob=1-dropout, output_keep_prob=1-dropout)

            # Process embedded inputs with rnn cell
            outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_fw, 
                                                         cell_bw=cell_bw, 
                                                         inputs=self.embedded_sents, 
                                                         sequence_length=self.sentence_lens, 
                                                         dtype=tf.float32,
                                                         time_major=False)

            #output1 = tf.nn.batch_normalization(outputs[0], training=self.is_training, name='birnn_bn1'+str(kernel_size))
            #output1 = tf.nn.relu(output1, name='birnn_relu1'+str(kernel_size))
            #output2 = tf.nn.batch_normalization(outputs[0], training=self.is_training, name='birnn_bn2'+str(kernel_size))
            #output2 = tf.nn.relu(output2, name='birnn_relu2'+str(kernel_size))

            # Concatenate the outputs for fwd and bwd directions (in the third dimension).
            #print('out', outputs, outputs[0])
            out_concat = tf.concat(outputs, axis=-1)
            #print('out concat', outputs)

            # Add a dense layer (without activation) into num_tags classes 
            logits = tf.layers.dense(out_concat, n_tags) 


            # Decoding

            # Generate `weights` as a 1./0. mask of valid/invalid words (using `tf.sequence_mask`).
            weights = tf.sequence_mask(self.sentence_lens, dtype=tf.float32)

            # Use crf for decoding
            if use_crf:

                # Compute log likelihood and transition parameters using tf.contrib.crf.crf_log_likelihood
                # and store the mean of sentence losses into `loss`.
                self.transition_params = tf.get_variable("transition_params", [n_tags, n_tags], initializer=tf.glorot_uniform_initializer())
                #self.transition_params = tf.get_variable("transition_params", [n_tags, n_tags], initializer=tf.random_normal_initializer)            
                log_likelihood, self.transition_params = tf.contrib.crf.crf_log_likelihood(inputs=logits, 
                                                                                           tag_indices=self.tags, 
                                                                                           sequence_lengths=self.sentence_lens,
                                                                                           transition_params=self.transition_params)

                self.loss = tf.reduce_mean(-log_likelihood)
                self.reduc_loss = self.loss
                # Compute the CRF predictions into `self.predictions` with `crf_decode`.
                self.predictions, self.scores = tf.contrib.crf.crf_decode(logits, self.transition_params, self.sentence_lens)
                #print('predicitons', self.predictions)

            # Use local softmax decoding
            else:             
                self.loss = tf.losses.sparse_softmax_cross_entropy(labels=self.tags, logits=logits, weights=weights)
                self.reduc_loss = tf.reduce_mean(self.loss)  
                #print("loss", self.reduc_loss)

                # Generate `self.predictions`.
                self.predictions = tf.argmax(logits, axis=-1) # 3rd dim!                


            ## To store previous losses for LRReductionOnPlateau
            #self.losses = deque([], maxlen=args.patience+1)

            global_step = tf.train.create_global_step()            

            # Choose optimizer                                              
            if optimizer == "SGD" and momentum:
                optimizer = tf.train.MomentumOptimizer(learning_rate=self.lr, momentum=momentum) 
                #self.training = tf.train.GradientDescentOptimizer(learning_rate) 
            elif optimizer == "SGD":
                optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.lr) 
            else:                
                optimizer = tf.train.AdamOptimizer(learning_rate=self.lr) 


            #optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)

            # Note how instead of `optimizer.minimize` we first get the # gradients using
            # `optimizer.compute_gradients`, then optionally clip them and
            # finally apply then using `optimizer.apply_gradients`.
            gradients, variables = zip(*optimizer.compute_gradients(self.loss))
            # TODO: Compute norm of gradients using `tf.global_norm` into `gradient_norm`.
            gradient_norm = tf.global_norm(gradients) 
            # TODO: If args.clip_gradient, clip gradients (back into `gradients`) using `tf.clip_by_global_norm`.            
            if clip_gradient is not None:
                gradients, _ = tf.clip_by_global_norm(gradients, clip_norm=clip_gradient, use_norm=gradient_norm)
            self.training = optimizer.apply_gradients(zip(gradients, variables), global_step=global_step)


            # Summaries
            self.current_accuracy, self.update_accuracy = tf.metrics.accuracy(self.tags, self.predictions, weights=weights)
            #self.current_precision, self.update_precision = tf.metrics.precision(self.tags, self.predictions, weights=weights)
            #self.current_recall, self.update_recall = tf.metrics.recall(self.tags, self.predictions, weights=weights)
            self.current_loss, self.update_loss = tf.metrics.mean(self.loss, weights=tf.reduce_sum(weights))
            self.reset_metrics = tf.variables_initializer(tf.get_collection(tf.GraphKeys.METRIC_VARIABLES))

            summary_writer = tf.contrib.summary.create_file_writer(logdir, flush_millis=10 * 1000)
            self.summaries = {}
            with summary_writer.as_default(), tf.contrib.summary.record_summaries_every_n_global_steps(10):
                self.summaries["train"] = [tf.contrib.summary.scalar("train/loss", self.update_loss),
                                           tf.contrib.summary.scalar("train/gradient_norm", gradient_norm),
                                           tf.contrib.summary.scalar("train/accuracy", self.update_accuracy)]
                                           #tf.contrib.summary.scalar("train/precision", self.update_precision),
                                           #tf.contrib.summary.scalar("train/recall", self.update_recall)]
            with summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
                for dataset in ["dev", "test"]:
                    self.summaries[dataset] = [tf.contrib.summary.scalar(dataset + "/loss", self.current_loss),
                                               tf.contrib.summary.scalar(dataset + "/accuracy", self.current_accuracy)]
                                               #tf.contrib.summary.scalar(dataset + "/precision", self.current_precision),
                                               #tf.contrib.summary.scalar(dataset + "/recall", self.current_recall)
                                               
            # Save model
            self.saver = tf.train.Saver()
            
            # Initialize variables
            self.session.run(tf.global_variables_initializer())
            with summary_writer.as_default():
                tf.contrib.summary.initialize(session=self.session, graph=self.session.graph)


    #def reduce_lr_on_plateau(self, cur_loss):
        ##global lr
        ##print(cur_loss, self.losses, self.lr)
        #self.losses.append(cur_loss)    
        #if len(self.losses) < args.patience + 1:
            #return
        #init_past_loss = self.losses.popleft()
        ##print(cur_loss, self.losses)
        ##print('past loss', init_past_loss)
        ##self.losses.append(cur_loss)
        #reduce_lr = True
        #for l in self.losses:
            #if l < init_past_loss:
                #reduce_lr = False

        #if reduce_lr:
            #print("Reducing lr from ", self.lr, end=" ")
            #self.lr = args.annealing_factor * self.lr
            #print("to ", self.lr)



    def train(self, 
              lr=.1,
              batch_size=32,
              dev_batch_size=32,
              epochs=150,
              annealing_factor=.5,
              patience=5,
              checkpoint=False):
        
        #self.lr = lr
        #self.batch_size = batch_size
        self.scheduler = ReduceLROnPlateau(lr, annealing_factor, patience)
            
        # Reset batch metrics
        self.session.run(self.reset_metrics)
        
       
        
        
        #train_data = corpus.train # downsample training data to 10
        #dev_data = corpus.dev
        #for e in range(args.epochs):
            #random.shuffle(train_data)
            #batches = [train_data[i:i + args.batch_size] for i in range(0, len(train_data), args.batch_size)]
            #for i, batch in enumerate(batches):
                #print("epoch\t", e, "\tbatch\t", i)
                #loss = tagger.train_epoch(batch, args.batch_size)
                #print("train loss\t", loss, "\tlr\t", tagger.lr)        
                
                ##accuracy, precision, recall, _, _ = tagger.evaluate("dev", dev_data, args.batch_size)
                ##acc, scores, loss = tagger.evaluate("dev", dev_data, args.batch_size)     #return totals_per_tag, totals, scores, loss                
                #tagger.evaluate("dev", dev_data, args.batch_size)     #return totals_per_tag, totals, scores, loss                
          
                ##[self.update_accuracy, self.update_precision, self.update_recall, self.update_loss, self.predictions
                ##print("dev loss\t", loss, "\tlr\t", tagger.lr)        
                ##print("tf dev accuracy = {:.2f}".format(100 * acc))
                
                ##print("precision = {:.2f}".format(100 * precision))
                ##print("recall = {:.2f}".format(100 * recall))
                
                ## Stop if lr becomes to small
                #if tagger.lr < .001:
                    #print("Exiting, lr has become to small: ", tagger.lr)
                    #break                
        
        
        train_data = corpus.train  
        for epoch in range(epochs):
            
            # Stop if lr becomes to small
            if self.lr < .001:
                print("Learning rate has become to small. Exiting training: lr=", tagger.lr)
                break    
            
            # Shuffle data and form batches
            random.shuffle(train_data)
            batches = [train_data[i:i + batch_size] for i in range(0, len(train_data), batch_size)]        
            #predicted_tags = np.zeros([n_sents, max_sent_len]) 
            for batch in batches:
                
                # Embed sentences
                self.embedding.embed(batch)
                # Sort batch and get lengths
                batch.sort(key=lambda i: len(i), reverse=True)
                max_sent_len = len(batch[0])
                sent_lens = [len(s.tokens) for s in batch]
                n_sents = len(sent_lens)    
            
                # Pad sentences and tags
                embedded_sents = np.zeros([n_sents, max_sent_len, self.embedding_dim])
                gold_tags = np.zeros([n_sents, max_sent_len]) 
                #gold_tags = np.zeros([max_sent_len, n_sents])        
                for i in range(n_sents):
                    for j in range(sent_lens[i]):                    
                        token = batch[i][j] 
                        #embedded_sents[j, i] = token.embedding.numpy() # convert torch tensor to numpy array
                        #gold_tags[j, i] = tag_dict.get_idx_for_item(token.get_tag(tag_type).value)  
                        embedded_sents[i, j] = token.embedding.numpy() # convert torch tensor to numpy array
                        gold_tags[i, j] = self.tag_dict.get_idx_for_item(token.get_tag(tag_type).value)      # tag index         

                # Run graph for batch
                _, _, loss = self.session.run([self.training, self.summaries["train"], self.loss],
                                 {self.sentence_lens: sent_lens,
                                  self.embedded_sents: embedded_sents,
                                  self.tags: gold_tags, self.is_training: True})
                
               
                # Save model if checkpoints enabled
                if checkpoint:
                    save_path = self.saver.save(self.session, "{}/checkpoint.ckpt".format(logdir))
                    print("Checkpoint saved at ", save_path)
               
                # Evaluate with dev data
                dev_data = corpus.dev                
                dev_score = self.evaluate("dev", dev_data, dev_batch_size)
                 
                # Perform one step on lr scheduler
                is_reduced = self.scheduler.step(dev_score)
                if is_reduced:
                    self.lr = self.scheduler.lr
                print("Epoch {}: train loss \t{}\t lr \t{}\t dev score \t{}\t bad epochs \t{}".format(epoch, loss, self.lr, dev_score, self.scheduler.bad_epochs))        
                
                # Save best model
                if dev_score == self.scheduler.best:
                    save_path = self.saver.save(self.session, "{}/best-model.ckpt".format(logdir))
                    print("Best model saved at ", save_path)
                                  
            #print("predictions\n", predictions)
            #print([tag_dict.get_item_for_index(x) for s in predictions for x in s])
            
            #BUG: REDUCEONPLATAEU
            #self.reduce_lr_on_plateau(loss)
            #have_reduced_lr = self.scheduler.step(loss)
            #if have_reduced_lr:
                #self.lr = self.scheduler.lr
                
                
            
    def evaluate(self, dataset_name, dataset, eval_batch_size=32, test_mode=False, metric="accuracy"):
        
        self.session.run(self.reset_metrics)  # for batch statistics
        # Get batches
        batches = [dataset[x:x+eval_batch_size] for x in range(0, len(dataset), eval_batch_size)]
        #print('dev baches', len(batches))
        
        # To store metrics
        totals_per_tag = defaultdict(lambda: defaultdict(int))
        totals = defaultdict(int)
        dev_loss = 0
        for batch in batches:
            
            # Embed sentences
            self.embedding.embed(batch)
            # Sort batch and get lengths
            batch.sort(key=lambda i: len(i), reverse=True)
            max_sent_len = len(batch[0])
            sent_lens = [len(s.tokens) for s in batch]
            n_sents = len(sent_lens)    
        
            # Pad sentences and tags
            embedded_sents = np.zeros([n_sents, max_sent_len, self.embedding_dim])
            gold_tags = np.zeros([n_sents, max_sent_len]) 
            #gold_tags = np.zeros([max_sent_len, n_sents])        

            for i in range(n_sents):
                for j in range(sent_lens[i]):                    
                    token = batch[i][j] 
                    #embedded_sents[j, i] = token.embedding.numpy() # convert torch tensor to numpy array
                    #gold_tags[j, i] = tag_dict.get_idx_for_item(token.get_tag(tag_type).value)  
                    embedded_sents[i, j] = token.embedding.numpy() # convert torch tensor to numpy array
                    gold_tags[i, j] = self.tag_dict.get_idx_for_item(token.get_tag(tag_type).value)      # tag index         
            
            if not test_mode:    
                _, _, dev_loss, predicted_tag_ids =  self.session.run([self.update_accuracy, 
                                                             self.update_loss, 
                                                             self.current_loss, 
                                                             self.predictions],
                                                            {self.sentence_lens: sent_lens,
                                                             self.embedded_sents: embedded_sents,
                                                             self.tags: gold_tags, 
                                                             self.is_training: False})   
                print("dev loss ", dev_loss)
            else:
                predicted_tag_ids = self.session.run(self.predictions,
                                                     {self.sentence_lens: sent_lens,
                                                      self.embedded_sents: embedded_sents,
                                                      self.is_training: False})                
                
            for i in range(n_sents):
                for j in range(sent_lens[i]):
                    token = batch[i][j]
                    predicted_tag = self.tag_dict.get_item_for_index(predicted_tag_ids[i][j])
                    token.add_tag('predicted', predicted_tag)
                    
                    
                    #print(token, predicted_tag)

                    
            for sentence in batch:
                #for token in sentence.tokens:
                    #predicted_tag = token.get_tag('predicted')
                    
                    ## append both to file for evaluation
                    #eval_line = '{} {} {}\n'.format(token.text,
                                                    #token.get_tag(self.model.tag_type).value,
                                                    #predicted_tag.value)                    
                    #lines.append(eval_line)
                #lines.append('\n')                            
        
                gold_tags = [(tag.tag, str(tag)) for tag in sentence.get_spans(tag_type)]
                predicted_tags = [(tag.tag, str(tag)) for tag in sentence.get_spans('predicted')]
                #print(len(gold_tags), gold_tags)
                #print(len(predicted_tags), predicted_tags)
                
                
                for tag, pred in predicted_tags:
                    if (tag, pred) in gold_tags:
                        
                        print(tag,pred)
                        print('tp')
                        
                        totals['tp'] += 1
                        totals_per_tag[tag]['tp'] += 1
                    else:
                        #print('fp')
                        totals['fp'] +=1
                        totals_per_tag[tag]['fp'] += 1
            
                for tag, gold in gold_tags:
                    #print(tag,gold)
                    if (tag, gold) not in predicted_tags:
                        #print('fn')
                        totals['fn'] += 1
                        totals_per_tag[tag]['fn'] += 1  
                    else:
                        #print('tn')
                        totals['tn'] +=1
                        totals_per_tag[tag]['tn'] += 1 # tn?
        
        if test_mode:                
            with open("{}/tagger_tag_test.txt".format(logdir), "w") as test_file:
                for i in range(len(batches)):
                    for j in range(len(batches[i])): 
                        for k in range(len(batches[i][j])):
                            token = batches[i][j][k]
                            gold_tag = token.get_tag(tag_type).value
                            predicted_tag = token.get_tag('predicted').value
                            print("{} {} {}".format(token.text, gold_tag, predicted_tag), file=test_file)
            return
        
                #forms = test.factors[test.FORMS].strings
                #gold_tags = test.factors[test.TAGS].strings
                #tags = tagger.predict(test, args.batch_size)
                #print(tags[:25])
                
                #test_data = corpus.test
                #forms, tags, gold = tagger.predict(test_data)
                #for s in range(len(forms)):
                    #for i in range(len(forms[s])):
                        #print(forms[s][i])
                        #print(gold_tags[s][i])
                        #print(tags[s][i])
                        #print("{} {} {}".format(forms[s][i], gold_tags[s][i], tags[s][i]), file=test_file)
                        ###print("{} {} {}".format(forms[s][i], gold_tags[s][i], test.factors[test.TAGS].words[tags[s][i]]), file=test_file)                        
                     
        self.metrics.log_metrics(dataset_name, totals, totals_per_tag)
        self.metrics.print_metrics()
        
        if metric == "accuracy":
            return self.metrics.accuracy
        
        else:
            return self.metrics.f1
        
        #return acc, scores, loss                
            
            
            
            
            #for i, s in enumerate(batch):
                #sent_len = len(s)     
                ##forms.append([t.text for t in s])
                
                #embedded_sents[i, 0:sent_len] = torch.cat([t.get_embedding().unsqueeze(0)
                                                         #for t in s], 0).numpy()
                #gold_tags[i, 0:sent_len] = [tag_dict.get_idx_for_item(t.get_tag(tag_type).value)
                                                  #for t in s]            
                #gold_tags_strings.append([t.get_tag(tag_type).value for t in s])                 
            
        
            ## Pad sentences
            #forms = []
            #gold_tags = []
            #embedded_sents = np.zeros([n_sents, max_sent_len, embedding_dim])
            #for i, s in enumerate(dataset):
                #sent_len = len(s)
                #forms.append([t.text for t in s])
                #embedded_sents[i, 0:sent_len] = torch.cat([t.get_embedding().unsqueeze(0)
                                                               #for t in s], 0).numpy()
                #gold.append([t.get_tag(tag_type).value for t in s])                 
        
            
            
            
        # Evaluate predictions

        #predicted_tags = [[tag_dict.get_item_for_index(t) for t in s] for s in predictions]
        #for s in range(len(predicted_tags)):
            #for t in range(len(s)):
                #if predicted_tags[s][t] == gold_tags_strings[s][t]:
                    
            
            
        #return self.session.run([self.update_accuracy, self.update_precision, self.update_recall, self.update_loss, self.predictions],
                                                   #{self.sentence_lens: sent_lens,
                                                    #self.embedded_sents: embedded_sents,
                                                    #self.tags: tags, self.is_training: False})
           
        #print("predictions\n")
        #print([tag_dict.get_item_for_index(t) for s in predictions for t in s])
        
        
        #while not dataset.epoch_finished():
            #sentence_lens, word_ids, charseq_ids, charseqs, charseq_lens = dataset.next_batch(batch_size, including_charseqs=True)
            #self.session.run([self.update_accuracy, self.update_precision, self.update_recall, self.update_loss],
                             #{self.sentence_lens: sentence_lens,
                              #self.charseqs: charseqs[train.FORMS], self.charseq_lens: charseq_lens[train.FORMS],
                              #self.word_ids: word_ids[train.FORMS], self.charseq_ids: charseq_ids[train.FORMS],
                              #self.tags: word_ids[train.TAGS], self.is_training: False})
        #return self.session.run([self.current_accuracy, self.current_precision, self.current_recall, self.summaries[dataset_name]]) 

    def predict(self, dataset):
        
        ## Embed sentences
        #stacked_embedding.embed(dataset)
        #dataset.sort(key=lambda i: len(i), reverse=True)
        #max_sent_len = len(dataset[0])
        #sent_lens = [len(s.tokens) for s in dataset]
        #n_sents = len(sent_lens)    
        batches = [dataset[x:x+batch_size] for x in range(0, len(dataset), batch_size)]
        
        for batch in batches:
            
            # Embed sentences
            stacked_embedding.embed(batch)
            batch.sort(key=lambda i: len(i), reverse=True)
            max_sent_len = len(batch[0])
            sent_lens = [len(s.tokens) for s in batch]
            n_sents = len(sent_lens)    
            
            embedded_sents = np.zeros([n_sents, max_sent_len, self.embedding_dim])
            #embedded_sents = np.zeros([max_sent_len, n_sents, embedding_dim])
            #forms = []
            gold_tags_strings = []
            gold_tags = np.zeros([n_sents, max_sent_len]) 
            #gold_tags = np.zeros([max_sent_len, n_sents])        
            
            predicted_tags = np.zeros([n_sents, max_sent_len]) 
            for i, sentence in enumerate(batch):
                for j in range(len(sentence)):
                    
                    #sent_len = len(s)     
                    #forms.append([t.text for t in s])
                    token = batch[i][j] 
                    #embedded_sents[j, i] = token.embedding.numpy() # convert torch tensor to numpy array
                    #gold_tags[j, i] = tag_dict.get_idx_for_item(token.get_tag(tag_type).value)             
                    embedded_sents[i, j] = token.embedding.numpy() # convert torch tensor to numpy array
                    gold_tags[i, j] = self.tag_dict.get_idx_for_item(token.get_tag(tag_type).value)      # tag index         
                    
        ## Pad sentences
        #forms = []
        #gold_tags = []
        #embedded_sents = np.zeros([n_sents, max_sent_len, embedding_dim])
        #for i, s in enumerate(dataset):
            #sent_len = len(s)
            #forms.append([t.text for t in s])
            #embedded_sents[i, 0:sent_len] = torch.cat([t.get_embedding().unsqueeze(0)
                                                           #for t in s], 0).numpy()
            #gold_tags.append([t.get_tag(tag_type).value for t in s])                    
    
        # Predict tags
        #predicted_tags = []   
        predictions = self.session.run(self.predictions,
                                     {self.sentence_lens: sent_lens,
                                      self.embedded_sents: embedded_sents,
                                      self.is_training: False})
        
        print(predictions)
        predicted_tags = [[tag_dict.get_item_for_index(t) for t in s] for s in predictions]
        
        
        #tags.extend([[tag_dict.get_item_for_index(t) for s in self.session.run(self.predictions,
                                     #{self.sentence_lens: sent_lens,
                                      #self.embedded_sents: embedded_sents,
                                      #self.is_training: False})] for t in s])
    
        #print("predictions\n")
        #print([tag_dict.get_item_for_index(x) for x in predicted_tags])
        print(forms)
        print(predicted_tags)
        print(gold_tags)
        return forms, tags, gold
        
        
        #tags = []
        #while not dataset.epoch_finished():
            #sentence_lens, word_ids, charseq_ids, charseqs, charseq_lens = dataset.next_batch(batch_size, including_charseqs=True)
            #tags.extend(self.session.run(self.predictions,
                                         #{self.sentence_lens: sentence_lens,
                                          #self.charseqs: charseqs[train.FORMS], self.charseq_lens: charseq_lens[train.FORMS],
                                          #self.word_ids: word_ids[train.FORMS], self.charseq_ids: charseq_ids[train.FORMS], 
                                          #self.is_training: False}))
        #return tags

class Metrics:
    
    def log_metrics(self, dataset_name, totals, totals_per_tag):
        
        self.totals = totals
        self.totals_per_tag = totals_per_tag
        
        with open("{}/metrics-{}.txt".format(logdir, dataset_name), "w") as f:
           
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
            f.write("acc {:.3f}\t prec {:.3f}\trec\t{:.3f}\tf1 {:.3f}\n".format(self.accuracy, self.precision, self.recall, self.f1))            
           
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
    from flair.data import Sentence
    import numpy as np
    
    # Fix random seed
    np.random.seed(42)

    ## Parse arguments
    #parser = argparse.ArgumentParser()
    #parser.add_argument("--batch_size", default=32, type=int, help="Batch size.")
    #parser.add_argument("--cle_dim", default=32, type=int, help="Character-level embedding dimension.")
    #parser.add_argument("--cnne_filters", default=16, type=int, help="CNN embedding filters per length.")
    #parser.add_argument("--optimizer", default="SGD", type=str, help="Optimizer.")    
    #parser.add_argument("--cnne_max", default=4, type=int, help="Maximum CNN filter length.")
    #parser.add_argument("--epochs", default=100, type=int, help="Number of epochs.")
    #parser.add_argument("--rnn_cell", default="LSTM", type=str, help="RNN cell type.")
    #parser.add_argument("--rnn_dim", default=64, type=int, help="RNN cell dimension.")
    #parser.add_argument("--num_layers", default=1, type=int, help="number of rnn layers.")
    #parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
    #parser.add_argument("--we_dim", default=100, type=int, help="Word embedding dimension.")
    #parser.add_argument("--lr", default=0.1, type=float, help="Initial learning rate.")
    #parser.add_argument("--annealing_factor", default=.5, type=float, help="Annealing factor for learning rate reduction on plateau.")
    #parser.add_argument("--patience", default=4, type=int, help="Patience for lr scheduler.")
    #parser.add_argument("--momentum", default=None, type=float, help="Momentum.")
    #parser.add_argument("--dropout", default=.5, type=float, help="Dropout rate.")
    #parser.add_argument("--bn", default=False, type=bool, help="Batch normalization.")
    #parser.add_argument("--clip_gradient", default=.25, type=float, help="Norm for gradient clipping.")
    #parser.add_argument("--use_crf", default=False, type=bool, help="Use conditional random field.")
    
    
    #args = parser.parse_args()

    # Create logdir name
    logdir = "logs/{}-{}".format(
        os.path.basename(__file__),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"))
    
    #logdir = "logs/{}-{}-{}".format(
        #os.path.basename(__file__),
        #datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        #",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items())))    
    #if not os.path.exists("logs"): os.mkdir("logs") # TF 1.6 will do this by itself

    #Load the data
    #train = morpho_dataset.MorphoDataset("/home/liefe/data/es/esp.train")
    #dev = morpho_dataset.MorphoDataset("/home/liefe/data/es/esp.testa", train=train, shuffle_batches=False)
    #test = morpho_dataset.MorphoDataset("/home/liefe/data/es/esp.testb", train=train, shuffle_batches=False)
    #train = morpho_dataset.MorphoDataset("/home/liefe/data/pt/harem/train_harem_iob2.conll")
    #dev = morpho_dataset.MorphoDataset("/home/liefe/data/pt/harem/dev_harem_iob2.conll", train=train, shuffle_batches=False)
    #test = morpho_dataset.MorphoDataset("/home/liefe/data/pt/harem/test_harem_iob2.conll", train=train, shuffle_batches=False)
    #train = morpho_dataset.MorphoDataset("/home/liefe/data/pt/UD_Portuguese-Bosque/pt_bosque-ud-train.conllu")
    #dev = morpho_dataset.MorphoDataset("/home/liefe/data/pt/UD_Portuguese-Bosque/pt_bosque-ud-dev.conllu", train=train, shuffle_batches=False)
    #test = morpho_dataset.MorphoDataset("/home/liefe/data/pt/UD_Portuguese-Bosque/pt_bosque-ud-test.conllu", train=train, shuffle_batches=False)
    
    
    #print("train, stats", len(train.factors[train.FORMS].words), len(train.factors[train.FORMS].alphabet),
                      #len(train.factors[train.TAGS].words))
    
    #print("tags", train.factors[train.TAGS].words)
    
    #print("use crf ", args.use_crf)

    # Get the corpus
    #fh = "/home/liefe/data/pt/UD_Portuguese-Bosque"
    fh = "/home/liefe/data/pt/harem/data/HAREM"
    #cols = {1:"text", 2:"lemma", 3:"pos"}
    cols = {0:"text", 1:"ne"}    
    corpus = NLPTaskDataFetcher.fetch_column_corpus(fh, 
                                                    cols, 
                                                    train_file="train.txt",
                                                    dev_file="dev.txt", 
                                                    test_file="test.txt").downsample(.2) 
    

    
    # Load Character Language Models (clms)
    clm_fw = CharLMEmbeddings("/home/liefe/lm/fw_p25/best-lm.pt")  
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
    
    ## Make the tag dictionary from the corpus
    #tag_dict = corpus.make_tag_dictionary(tag_type=tag_type)
    #n_tags = len(tag_dict)
    
    # Construct the tagger
    tagger = SequenceTagger(corpus, stacked_embedding, tag_type)
    #tagger.construct(args, len(train.factors[train.FORMS].words), len(train.factors[train.FORMS].alphabet),
                      #len(train.factors[train.TAGS].words))
    #tagger.construct(args, n_tags)
        
    # Train
    tagger.train(patience=20, checkpoint=True)   
    #train_data = corpus.train
    #train_data = corpus.train # downsample training data to 10%
    
    
    #dev_data = corpus.dev
    #for e in range(args.epochs):
        #random.shuffle(train_data)
        #batches = [train_data[i:i + args.batch_size] for i in range(0, len(train_data), args.batch_size)]
        #for i, batch in enumerate(batches):
            #print("epoch\t", e, "\tbatch\t", i)
            #loss = tagger.train_epoch(batch, args.batch_size)
            #print("train loss\t", loss, "\tlr\t", tagger.lr)        
            
            ##accuracy, precision, recall, _, _ = tagger.evaluate("dev", dev_data, args.batch_size)
            ##acc, scores, loss = tagger.evaluate("dev", dev_data, args.batch_size)     #return totals_per_tag, totals, scores, loss                
            #tagger.evaluate("dev", dev_data, args.batch_size)     #return totals_per_tag, totals, scores, loss                
      
            ##[self.update_accuracy, self.update_precision, self.update_recall, self.update_loss, self.predictions
            ##print("dev loss\t", loss, "\tlr\t", tagger.lr)        
            ##print("tf dev accuracy = {:.2f}".format(100 * acc))
            
            ##print("precision = {:.2f}".format(100 * precision))
            ##print("recall = {:.2f}".format(100 * recall))
            
            ## Stop if lr becomes to small
            #if tagger.lr < .001:
                #print("Exiting, lr has become to small: ", tagger.lr)
                #break        
    
    # FIX NOT BREAKING FROM OUTER TRAIN
    
    test_data = corpus.test
    tagger.evaluate("test", test_data, test_mode=True)
    
    
    #with open("{}/tagger_tag_test.txt".format(args.logdir), "w") as test_file:
        ##forms = test.factors[test.FORMS].strings
        ##gold_tags = test.factors[test.TAGS].strings
        ##tags = tagger.predict(test, args.batch_size)
        ##print(tags[:25])
        
        #test_data = corpus.test
        #forms, tags, gold = tagger.predict(test_data)
        #for s in range(len(forms)):
            #for i in range(len(forms[s])):
                #print(forms[s][i])
                #print(gold_tags[s][i])
                #print(tags[s][i])
                #print("{} {} {}".format(forms[s][i], gold_tags[s][i], tags[s][i]), file=test_file)
                ##print("{} {} {}".format(forms[s][i], gold_tags[s][i], test.factors[test.TAGS].words[tags[s][i]]), file=test_file)
                
            #print("", file=test_file)            