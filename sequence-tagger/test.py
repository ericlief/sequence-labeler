import torch.nn
import numpy as np
data = torch.load("/home/liefe/tag/best-lm.pt", map_location=lambda storage, loc: storage)
#torch.load('my_file.pt', map_location=lambda storage, loc: storage) 
print(data)
a = data.keys()
print(a)
d = data['state_dict'].keys()
#print(d)
encoder = data['state_dict']['encoder.weight']
print('encoder\n', encoder.size())
rnn = data['state_dict']['rnn.weight_ih_l0']
print('rnn_ih\n', rnn.size())
rnn = data['state_dict']['rnn.weight_hh_l0']
print('rnn_hh\n', rnn.size())
rnn = data['state_dict']['decoder.weight']
print('decoder', rnn.size())
x=torch.ones(10,2,2)
m = x.data.new(x.size(0), 1, 1).bernoulli_(1 - .23)
m = x.data.new(1, x.size(1), x.size(2)).bernoulli_(.60)
#print(m)
mask = m.expand_as(x)
#print(mask)