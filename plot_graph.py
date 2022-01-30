import matplotlib.pyplot as plt
import csv
  
x = []
y = []
  
with open('reward_episode.csv','r') as csvfile:
    plots = csv.reader(csvfile, delimiter = ',')
      
    for row in plots:
        x.append(row[0])
        y.append(row[1])
  
#plt.bar(x, y, color = 'g', width = 0.72, label = "Age")
plt.plot(y, x, color = 'g', linestyle = 'dashed',
         label = "GA Params")
plt.xlabel('Episodes')
plt.ylabel('Rewards')
plt.title('GA vs Default Hyperparameters')
plt.legend()
plt.show()