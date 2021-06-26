import numpy as np

# Initial probabilities
p_fish = 0.5
p_nfish = 0.5

# Transition probabilities
p_fish_fish = 0.8
p_fish_nfish = 0.2
p_nfish_fish = 0.3
p_nfish_nfish = 0.7

# Emission probabilities
p_fish_bird = 0.75
p_fish_nbird = 0.25
p_nfish_bird = 0.2
p_nfish_nbird = 0.8

evidence = ['bird', 'bird', 'n_bird', 'bird', 'n_bird', 'bird']

probabilities = []
mls_result = []

if evidence[0] == 'bird':
    probabilities.append((p_fish*p_fish_bird, p_nfish*p_nfish_bird))
else:
    probabilities.append((p_fish*p_fish_nbird, p_nfish*p_nfish_nbird))

for i in range(1,len(evidence)):
    yesterday_fish, yesterday_nfish = probabilities[-1]
    if evidence[i] == 'bird':
        today_fish = max(yesterday_fish*p_fish_fish*p_fish_bird, yesterday_nfish*p_nfish_fish*p_fish_bird)
        today_nfish = max(yesterday_fish*p_fish_nfish*p_nfish_bird, yesterday_nfish*p_nfish_nfish*p_nfish_bird)
        probabilities.append((today_fish, today_nfish))
    else:
        today_fish = max(yesterday_fish*p_fish_fish*p_fish_nbird, yesterday_nfish*p_nfish_fish*p_fish_nbird)
        today_nfish = max(yesterday_fish*p_fish_nfish*p_nfish_nbird, yesterday_nfish*p_nfish_nfish*p_nfish_nbird)
        probabilities.append((today_fish, today_nfish))

for prob in probabilities:
    if prob[0] > prob[1]:
        mls_result.append('True')
    else:
        mls_result.append('False')

print('\nDay\tTrue\tFalse')
print('-----------------------')

for i in range(len(probabilities)):
    print(f'{i+1}\t{round(probabilities[i][0],4)}\t{round(probabilities[i][1],4)}')

print('\nMost likely sequence')
print('------------------------')
for i in range(len(mls_result)):
    print(f'Day {i+1}: \t{mls_result[i]}')
print('\n')
