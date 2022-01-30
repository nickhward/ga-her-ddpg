import os

query = "python3 train.py"

for _ in range(20):
    print(query)
    #calling training to calculate number of epochs required to reach close to maximum success rate
    os.system(query)