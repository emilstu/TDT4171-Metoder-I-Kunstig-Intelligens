import numpy as np

# Transition probabilities
tr = np.array([[0.8, 0.2],
                [0.3, 0.7]])


# Emission probabilities
em = np.array([[0.75, 0.25],
                [0.2, 0.8]])


# Equal Probabilities for the initial distribution
initial_distribution = np.array([0.5, 0.5])

# Evidence
evidence = np.array([0, 0, 1, 0, 1, 0])

alpha = []
for i in range(30):
    
    # Calculate forward probabilities 
    if i < len(evidence):    
        if i == 0:
            pi = initial_distribution
        else:
            pi = alpha[i-1]
        
        # Get observation matrix
        obs = np.diag(em[:,evidence[i]])

        # Multiply matrices
        f = np.matmul(np.matmul(obs, tr.transpose()),pi)
        
        # Normalize result
        f = f/f.sum(axis=0, keepdims=1)
    
    # Predict future  
    else:
        pi = alpha[i-1]
        
        # Multiply matrices
        f = np.matmul(tr.transpose(),pi)

        # Normalize result
        f = f/f.sum(axis=0, keepdims=1)

    alpha.append(f)    


print('\nDay\tTrue\tFalse')
print('-----------------------')

for i in range(len(evidence), len(alpha)):
    print(f'{i+1}\t{round(alpha[i][0],4)}\t{round(alpha[i][1],4)}')
print('\n') 
