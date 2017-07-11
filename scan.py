#!/usr/bin/python
'''
FILE: check.py
DESC: Scans given text for irregularities
EXEC: python3 scan.py <file_name>.json
VERS: Python 3
'''
import sys
import numpy as np
import pickle
import os

CHECKPOINT = "./data/checkpoint.txt"
INPUT = "./data/input.txt"
NUM_PRIMER_CHARS = 40
RANK_LIMIT = 5    # Number of charaters
CONFIDENCE = 0.99 # Minimal confidence level to make changes

class Neural_State(object):

  def __init__(self):
    self.data = open(INPUT, 'r').read()
    self.chars = list(set(self.data))
    self.data_size, self.vocab_size = len(self.data), len(self.chars)

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
      print("==================================================")
      print("Load from checkpoint: min_loss: ", self.min_loss)
      print("==================================================")

    self.h = self.hprev
    self.text = open(sys.argv[1], 'r').read()

  def update_state(self, x):
    ''' Updates Neural State '''
    self.h = np.tanh(np.dot(self.Wxh, x) + np.dot(self.Whh, self.h) + self.bh)
    y = np.dot(self.Why, self.h) + self.by
    p = np.exp(y) / np.sum(np.exp(y)) # Probabilities
    return p

  def predict_state(self, state):
    ''' Predicts next state given the current state '''
    Wxh, Whh,  Why, h, bh, by, p_char= state

    x = np.zeros((self.vocab_size, 1))
    x[self.char_to_ix[p_char]] = 1

    h = np.tanh(np.dot(Wxh, x) + np.dot(Whh, h) + bh)
    y = np.dot(Why, h) + by
    p = np.exp(y) / np.sum(np.exp(y)) # Probabilities

    rc = self.rank_chars(p)           # Ranked Characters
    x = np.zeros((self.vocab_size, 1))

    p_char = rc[0][0]   # Predicted Character

    state =  [Wxh, Whh, Why, h, bh, by, p_char]
    return state

  def print_ranked_chars(self, actual, rc):
    """
    Prints the predicted charaters and their probabilities
    """
    s_char = {'\n': "\\n", ' ': ' '} # special charaters

    print("\n=======================================")
    actual = s_char.get(actual) if actual in s_char else actual
    print("Next Char:", "\'" +  actual + "\'  ", str([val for key, val in rc if key == actual])[1:-1])
    print("Predicted:")
    for i in rc[0:RANK_LIMIT]:
      char = i[0]
      if(i[0] in s_char.keys()):
        char = s_char.get(char)
      print("           ", char, "  " ,i[1])
    print("====================================")

  def rank_chars(self, p):
    """
    Returns list of chars in descending order of probability
    """
    rc = []
    for j in range(len(p)):
      rc.append((self.ix_to_char[j], list(p[j])[0]))
    rc.sort(key=lambda x:x[1], reverse=True)
    return rc

  def prime_net(self, primer):
    """
    Prime neural network with a sequence of characters
    """
    x = np.zeros((self.vocab_size, 1))
    x[self.char_to_ix[primer[0]]] = 1

    for i in range(len(primer)):
      self.h = np.tanh(np.dot(self.Wxh, x) + np.dot(self.Whh, self.h) + self.bh)
      y = np.dot(self.Why, self.h) + self.by
      p = np.exp(y) / np.sum(np.exp(y)) # Probabilities
      ix = self.char_to_ix[primer[i]]
      x = np.zeros((self.vocab_size, 1))
      x[ix] = 1
    return x

  def speculate(self, text, p_char):
    ''' Given the text up to point i, speculates as to
        what the current word is '''
    state = [self.Wxh, self.Whh, self.Why, self.h, self.bh, self.by, p_char]

    while(p_char != " " and p_char != '\n'):
      text = list(text) + [p_char]
      text = ''.join(text)
      state = self.predict_state(state)
      Wxh, Whh,  Why, h, bh, by, p_char = state
    return text

  def scan(self, text):
    """
    Scans text by char, stopping at first error found by neural net
    and offers suggestions for fix
    """
    digits = len(str(count_lines(text)))
    c_char = '' # Current Character
    p_char = '' # Predicted Character
    line_number = 1

    char = ''
    for i in range(len(text)):
      if(char == '\n' or i == 0):
        spaces = " " * (digits - len(str(line_number)))
        print(str(line_number) + spaces, end='', flush=True)
        line_number += 1

      x = np.zeros((self.vocab_size, 1))
      if(i < NUM_PRIMER_CHARS):
        # Prime Neural Net
        x[self.char_to_ix[text[i]]] = 1
        p = self.update_state(x)
        char = text[i]
      else:
        # Scan Characters
        rc = self.rank_chars(p)                   # Ranked Characters
        pc = pick_char(text[i], rc[0:RANK_LIMIT]) # Picked Character
        x = np.zeros((self.vocab_size, 1))

        c_char = text[i] # Current Character
        p_char = pc[0]   # Predicted Character

        if(p_char != c_char):
          #self.print_ranked_chars(c_char, rc)
          start = get_previous_space(text, i)
          end   = get_next_space(text, i)
          spec  = self.speculate(text[0:i], p_char)
          print()
          print("\nUNEXPECTED SEQUENCE:", "\'" + text[start:end] + "\'"  ,"in line 5")
          print("\nReplacement Suggestions:")
          print("------------------------")
          print("\n", text[start:end], "--->", end ='', flush = True)
          print('\033[92m' + spec[start:len(spec)] + '\033[0m')
          break
        else:
          x[self.char_to_ix[p_char]] = 1
          p = self.update_state(x)
          char = p_char
          #self.print_ranked_chars(c_char, rc)
      print(char, end='', flush=True)
    print()

#################################################
# HELPER FUNCTIONS
#################################################
def pick_char(a, rank):
  """
  Return highest ranked character or the current char
  a if it is within the top RANK_LIMIT of the rank list
  """
  c = 0 # Char index
  p = 1 # Prob index
  for i in range(len(rank)):
    if(a == rank[i][c]): #and rank[i][p] > 0.10):
      return rank[i]
  if(rank[0][p] > CONFIDENCE):
    return rank[0][c]
  else:
    return a

def get_previous_space(text, i):
  """ Returns most recent space of text sequence """
  for j in range(i-1, 0, -1):
    if(text[j] == ' ' or text[j] == '\n'):
      return j
  return NULL

def get_next_space(text, i):
  """ Returns next space of text sequence """
  for j in range(i, len(text)):
    if(text[j] == ' ' or text[j] == '\n'):
      return j
  return NULL

def count_lines(text):
  count = 0
  for i in text:
    if(i == '\n'):
      count += 1
  return count

#################################################
# RUN
#################################################
def run():
  ns = Neural_State()
  ns.scan(ns.text)
  #print("==================================================: ")
  #print("Load from checkpoint: min_loss: ", ns.min_loss)
  #print("==================================================: ")

run()






