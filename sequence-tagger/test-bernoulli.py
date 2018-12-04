import torch                                                           
from torch import nn                                                
from torch.distributions.bernoulli import Bernoulli
#from tensorflow.distributions import Bernoulli
import tensorflow as tf
import numpy as np

nwords, dim = 4, 2
emb = nn.Embedding(nwords, dim)
print(emb)
#input = torch.LongTensor([[0, 1, 2, 2, 1],
                          #[0, 3, 2, 1, 2]])
#input = torch.LongTensor([[0, 2, 1],
                          #[0, 1, 2]])
#out = emb(input)
#print(out)
#rw = Bernoulli(0.3).sample((out.shape[1], ))
#print(rw)
#out_ = out[:, rw==1].mean(dim=1)
#print(out_)

batch_size = 2
dropout_rate = .25
x = torch.LongTensor([[0, 2, 1],
                      [0, 1, 2]])
x = emb(x)
print(x)
print(x.size(0))
m = x.data.new(x.size(0), 1, 1).bernoulli_(1 - dropout_rate)
print('m', m)
#b = Bernoulli(1 - dropout_rate).sample((x.shape[0], 1, 1))
#print('b', b)
mask = torch.autograd.Variable(m, requires_grad=False)
mask = mask.expand_as(x)
print('mask', mask)
res = mask * x
print(res)

print(x.data)
x_ = x.data.new(x.size())
print(x_)



sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
x = tf.constant([[0, 2, 1],
                [0, 1, 2]], tf.int32)
rate = tf.constant(1 - dropout_rate)
print(x)

emb = tf.get_variable("emb", [nwords, dim], tf.float32)
embedded = tf.nn.embedding_lookup(emb, x)
print(embedded)
sess.run(tf.global_variables_initializer())
res = sess.run(embedded)
print(res)

#probs = tf.broadcast_to(rate, x)
probs = np.array([1 - dropout_rate] * batch_size )
print(probs)
m = tf.contrib.distributions.Bernoulli(probs=1-dropout_rate)
sample = m.sample([embedded.get_shape()[0], 1, 1])
#mask = tf.broadcast_to(sample, x)
mask = tf.tile([1, embedded.get_shape()[1], embedded.get_shape()[2]])

#print(sess.run(m.sample([embedded.get_shape()[0], 1])))
print(sess.run(mask))