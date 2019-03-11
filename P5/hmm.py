from __future__ import print_function
import json
import numpy as np
import sys

def forward(pi, A, B, O):
  """
  Forward algorithm

  Inputs:
  - pi: A numpy array of initial probabilities. pi[i] = P(Z_1 = s_i)
  - A: A numpy array of transition probabilities. A[i, j] = P(Z_t = s_j|Z_t-1 = s_i)
  - B: A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
  - O: A list of observation sequence (in terms of index, not the actual symbol)

  Returns:
  - alpha: A numpy array alpha[j, t-1] = P(Z_t = s_j, X_{1:t}=x_{1:t})
  """
  S = len(pi)
  N = len(O)
  alpha = np.zeros([S, N])
  ###################################################
  # Q3.1 Edit here
  alpha[:,0]=pi*B[:,0]
  for t in range(1,N):
    for j in range(S):
      alpha[j][t]=B[j][O[t]]*np.sum(alpha[:,t-1]*A[:,j]) 
  ###################################################
  
  return alpha


def backward(pi, A, B, O):
  """
  Backward algorithm

  Inputs:
  - pi: A numpy array of initial probabilities. pi[i] = P(Z_1 = s_i)
  - A: A numpy array of transition probabilities. A[i, j] = P(Z_t = s_j|Z_t-1 = s_i)
  - B: A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
  - O: A list of observation sequence (in terms of index, not the actual symbol)

  Returns:
  - beta: A numpy array beta[j, t-1] = P(X_{t+1:N}=x_{t+1:N} | Z_t = s_j)
  """
  S = len(pi)
  N = len(O)
  beta = np.zeros([S, N])
  ###################################################
  # Q3.1 Edit here
  beta[:,N-1]=1
  for t in range(N-2,-1,-1):
    for i in range(S):
      for j in range(S):
        beta[i][t]+=beta[j][t+1]*A[i][j]*B[j][O[t+1]]
  ###################################################
  
  return beta

def seqprob_forward(alpha):
  """
  Total probability of observing the whole sequence using the forward messages

  Inputs:
  - alpha: A numpy array alpha[j, t-1] = P(Z_t = s_j, X_{1:t}=x_{1:t})

  Returns:
  - prob: A float number of P(X_{1:N}=O)
  """
  prob = 0
  ###################################################
  # Q3.2 Edit here
  ###################################################
  prob=np.sum(alpha[:,-1])
  
  return prob


def seqprob_backward(beta, pi, B, O):
  """
  Total probability of observing the whole sequence using the backward messages

  Inputs:
  - beta: A numpy array beta: A numpy array beta[j, t-1] = P(X_{t+1:N}=x_{t+1:N} | Z_t = s_j)
  - pi: A numpy array of initial probabilities. pi[i] = P(Z_1 = s_i)
  - B: A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
  - O: A list of observation sequence
      (in terms of the observation index, not the actual symbol)

  Returns:
  - prob: A float number of P(X_{1:N}=O)
  """
  prob = 0
  ###################################################
  # Q3.2 Edit here
  ###################################################
  prob=np.sum(beta[:,0]*pi*B[:,O[0]])
  return prob

def viterbi(pi, A, B, O):
  """
  Viterbi algorithm

  Inputs:
  - pi: A numpy array of initial probabilities. pi[i] = P(Z_1 = s_i)
  - A: A numpy array of transition probabilities. A[i, j] = P(Z_t = s_j|Z_t-1 = s_i)
  - B: A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
  - O: A list of observation sequence (in terms of index, not the actual symbol)

  Returns:
  - path: A list of the most likely hidden state path (in terms of the state index)
  """
  path = []
  ###################################################
  # Q3.3 Edit here
  ###################################################
  T=len(O)
  S=len(pi)

  prob=np.zeros((S,T))
  delta=np.zeros((S,T))

  prob[:,0]=pi*B[:,0]

  for t in range(1,T):
    for s in range(S):
      prob[s][t]=max(prob[:,t-1]*A[:,s])*B[s][O[t]]
      delta[s][t]=np.argmax(prob[:,t-1]*A[:,s])

  path.append(np.argmax(prob[:,T-1]))
 
  for t in range(T-2,-1,-1):
    path.append(int(delta[path[-1]][t]))
  return path


##### DO NOT MODIFY ANYTHING BELOW THIS ###################
def main():
  model_file = sys.argv[1]
  Osymbols = sys.argv[2]

  #### load data ####
  with open(model_file, 'r') as f:
    data = json.load(f)
  A = np.array(data['A'])
  B = np.array(data['B'])
  pi = np.array(data['pi'])
  #### observation symbols #####
  obs_symbols = data['observations']
  #### state symbols #####
  states_symbols = data['states']

  N = len(Osymbols)
  O = [obs_symbols[j] for j in Osymbols]

  alpha = forward(pi, A, B, O)
  beta = backward(pi, A, B, O)

  prob1 = seqprob_forward(alpha)
  prob2 = seqprob_backward(beta, pi, B, O)
  print('Total log probability of observing the sequence %s is %g, %g.' % (Osymbols, np.log(prob1), np.log(prob2)))

  viterbi_path = viterbi(pi, A, B, O)

  print('Viterbi best path is ')
  for j in viterbi_path:
    print(states_symbols[j], end=' ')

if __name__ == "__main__":
  main()
