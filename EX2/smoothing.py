import numpy as np

# Transition probabilities
tr = np.array([[0.8, 0.2],
                [0.3, 0.7]])


# Emission probabilities
em = np.array([[0.75, 0.25],
                [0.2, 0.8]])
            

# Equal Probabilities for the initial distribution
f_initial_distribution = np.array([0.5, 0.5])
b_initial_distribution = np.array([1.0, 1.0])

# Evidence
evidence = np.array([0, 0, 1, 0, 1, 0])

# Forward algorithm
alpha = []
for i in range(len(evidence)):
    if i == 0:
        pi = f_initial_distribution
    else:
        pi = alpha[i-1]
    
    obs = np.diag(em[:,evidence[i]])
    f = np.matmul(np.matmul(obs, tr.transpose()), pi)
    f = f/f.sum(axis=0, keepdims=1)
    alpha.append(f)


# Backward algorithm
beta = [0]*len(evidence)
for i in range(len(evidence)-1, -1, -1):
    if i == len(evidence)-1:
        pi = b_initial_distribution
    else:
        pi = beta[i+1]
    
    obs = np.diag(em[:,evidence[i]])
    b = np.matmul(np.matmul(tr.transpose(), obs), pi)
    b = b/b.sum(axis=0, keepdims=1)
    beta[i] = b


# Compute smoothed probabilites
gamma = []

for i in range(len(evidence)+1):
    if i == 0:
        a = f_initial_distribution
        b = beta[i]
    elif i == len(evidence):
        b = b_initial_distribution
        a = alpha[i-1]
    else:
        a = alpha[i-1]
        b = beta[i]
    
    g = a*b
    g = g/g.sum(axis=0, keepdims=1)
    gamma.append(g)


print('\nDay\tTrue\tFalse')
print('-----------------------')

for i in range(len(gamma)-1):
    print(f'{i}\t{round(gamma[i][0],4)}\t{round(gamma[i][1],4)}')
print('\n') 

