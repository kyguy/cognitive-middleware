"""
Minimal character-level Vanilla RNN model. Written self.by Andrej Karpathy (@karpathy)
BSD License
"""
import numpy as np
import pickle
import os

DATA       = "input.txt"
CHECKPOINT = "./data/checkpoint.txt"
ITERATION  = 100000

class RNN(object):

  def __init__(self, hidden_size, seq_length, learning_rate):

    #data I/O
    self.data = open('input.txt', 'r').read() # should be simple plain text file
    self.chars = list(set(self.data))
    self.data_size, self.vocab_size = len(self.data), len(self.chars)

    self.seq_length = seq_length
    self.hidden_size = hidden_size
    self.learning_rate = learning_rate

    #################################################
    #  INITIALIZE STATE
    #################################################
    if(os.path.isfile(CHECKPOINT)):
      checkpoint = open(CHECKPOINT, 'rb')
      state = pickle.load(checkpoint)
      self.Wxh, self.Whh, self.Why, self.bh, self.by, self.mWxh, self.mWhh, self.mWhy, self.mbh, self.mby,\
      self.inputs, self.targets, self.hprev, self.loss, self.smooth_loss, self.p, self.n,\
      self.min_loss, self.chars, self.char_to_ix, self.ix_to_char = state
      checkpoint.close()
      print("==================================================: ")
      print("Load from checkpoint: min_loss: ", self.min_loss)
      print("==================================================: ")
    else:
      print('data has %d characters, %d unique.' % (self.data_size, self.vocab_size))
      self.char_to_ix = { ch:i for i,ch in enumerate(self.chars) }
      self.ix_to_char = { i:ch for i,ch in enumerate(self.chars) }

      # model parameters
      self.Wxh = np.random.randn(hidden_size, vocab_size)*0.01 # input to hidden
      self.Whh = np.random.randn(hidden_size, hidden_size)*0.01 # hidden to hidden
      self.Why = np.random.randn(vocab_size, hidden_size)*0.01 # hidden to output
      self.bh = np.zeros((hidden_size, 1)) # hidden bias
      self.by = np.zeros((vocab_size, 1))  # output bias

      self.n, self.p = 0, 0
      self.mWxh, self.mWhh, self.mWhy = np.zeros_like(self.Wxh), np.zeros_like(self.Whh), np.zeros_like(self.Why)
      self.mbh, self.mby = np.zeros_like(self.bh), np.zeros_like(self.by)  # memory variables for Adagrad
      self.smooth_loss = -np.log(1.0/self.vocab_size)*self.seq_length      # self.loss at iteration 0

      self.inputs, self.targets, hprev = [], [], []
      self.loss, self.smooth_loss = 0
      self.min_loss = 500


  def lossFun(self, inputs, targets, hprev):
    """
    self.inputs,self.targets are both list of integers.
    self.hprev is Hx1 array of initial hidden state
    returns the self.loss, gradients on model parameters, and last hidden state
    """
    xs, hs, ys, ps = {}, {}, {}, {}
    hs[-1] = np.copy(self.hprev)
    self.loss = 0
    # forward pass
    for t in range(len(self.inputs)):
      xs[t] = np.zeros((self.vocab_size,1)) # encode in 1-of-k representation
      xs[t][self.inputs[t]] = 1
      hs[t] = np.tanh(np.dot(self.Wxh, xs[t]) + np.dot(self.Whh, hs[t-1]) + self.bh) # hidden state
      ys[t] = np.dot(self.Why, hs[t]) + self.by # unnormalized log probabilities for next self.chars
      ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t])) # probabilities for next self.chars
      self.loss += -np.log(ps[t][self.targets[t],0]) # softmax (cross-entropy self.loss)
    # backward pass: compute gradients going backwards
    dWxh, dWhh, dWhy = np.zeros_like(self.Wxh), np.zeros_like(self.Whh), np.zeros_like(self.Why)
    dbh, dby = np.zeros_like(self.bh), np.zeros_like(self.by)
    dhnext = np.zeros_like(hs[0])
    for t in reversed(range(len(self.inputs))):
      dy = np.copy(ps[t])
      dy[self.targets[t]] -= 1 # backprop into y. see http://cs231n.github.io/neural-networks-case-study/#grad if confused here
      dWhy += np.dot(dy, hs[t].T)
      dby += dy
      dh = np.dot(self.Why.T, dy) + dhnext # backprop into h
      dhraw = (1 - hs[t] * hs[t]) * dh # backprop through tanh nonlinearity
      dbh += dhraw
      dWxh += np.dot(dhraw, xs[t].T)
      dWhh += np.dot(dhraw, hs[t-1].T)
      dhnext = np.dot(self.Whh.T, dhraw)
    for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
      np.clip(dparam, -5, 5, out=dparam) # clip to mitigate exploding gradients
    return self.loss, dWxh, dWhh, dWhy, dbh, dby, hs[len(self.inputs)-1]

  def sample(self, h, seed_ix, n):
    """
    sample a sequence of integers from the model
    h is memory state, seed_ix is seed letter for first time step
    """
    x = np.zeros((self.vocab_size, 1))
    x[seed_ix] = 1
    ixes = []
    for t in range(n):
      h = np.tanh(np.dot(self.Wxh, x) + np.dot(self.Whh, h) + self.bh)
      y = np.dot(self.Why, h) + self.by
      p = np.exp(y) / np.sum(np.exp(y))
      ix = np.random.choice(range(self.vocab_size), p=p.ravel())
      x = np.zeros((self.vocab_size, 1))
      x[ix] = 1
      ixes.append(ix)
    return ixes

  def train(self):
    while True:
      # prepare self.inputs (we're sweeping from left to right in steps seq_length long)
      if self.p+self.seq_length+1 >= len(self.data) or self.n == 0:
        self.hprev = np.zeros((self.hidden_size,1)) # reset RNN memory
        self.p = 0 # go from start of data
      self.inputs  = [self.char_to_ix[ch] for ch in self.data[self.p:self.p+self.seq_length]]
      self.targets = [self.char_to_ix[ch] for ch in self.data[self.p+1:self.p+self.seq_length+1]]
      # sample from the model now and then
      if self.n % ITERATION == 0:
        sample_ix = self.sample(self.hprev, self.inputs[0], 200)
        txt = ''.join(self.ix_to_char[ix] for ix in sample_ix)
        print('----\n %s \n----' % (txt, ))

      # forward seq_length characters through the net and fetch gradient
      self.loss, dWxh, dWhh, self.dWhy, dbh, dby, self.hprev = self.lossFun(self.inputs, self.targets, self.hprev)
      self.smooth_loss = self.smooth_loss * 0.999 + self.loss * 0.001
      #################################################
      #  SAMPLE STATE
      #################################################
      if self.n % ITERATION == 0: print('iter %d, self.loss: %f' % (self.n, self.smooth_loss)) # print progress
      # perform parameter update with Adagrad
      for param, dparam, mem in zip([self.Wxh, self.Whh, self.Why, self.bh, self.by],
                               [dWxh, dWhh, self.dWhy, dbh, dby],
                               [self.mWxh, self.mWhh, self.mWhy, self.mbh, self.mby]):
        mem += dparam * dparam
        param += -self.learning_rate * dparam / np.sqrt(mem + 1e-8) # adagrad update

      self.p += self.seq_length # move data pointer
      self.n += 1 # iteration counter

      #################################################
      #  CHECKPOINT STATE
      #################################################
      if(self.n % ITERATION == 0 or self.smooth_loss < self.min_loss):
        self.min_loss = self.smooth_loss
        state = (self.Wxh, self.Whh, self.Why, self.bh, self.by, self.mWxh, self.mWhh, self.mWhy, self.mbh, self.mby,
                 self.inputs, self.targets, self.hprev, self.loss, self.smooth_loss, self.p, self.n,
                 self.min_loss, self.chars, self.char_to_ix, self.ix_to_char)
        checkpoint = open(CHECKPOINT, 'wb')
        pickle.dump(state, checkpoint)
        checkpoint.close()
        print("UPDATED min_loss: ", self.min_loss)

def test():

  # hyperparameters
  hidden_size = 100 # size of hidden layer of neurons
  seq_length = 25 # number of steps to unroll the RNN for
  learning_rate = 1e-1

  rnn = RNN(hidden_size, seq_length, learning_rate)
  rnn.train()

test()
