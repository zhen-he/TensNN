import json
import matplotlib.pyplot as plt


drawTrain = 1

str = 'val'
if drawTrain == 1: 
  str = 'train'

# with open('t3_256_lastBN.json', 'r') as f1:
#   data1 = json.load(f1)
# hist1 = data1[str + '_loss_history']
 
with open('t3_256_lastBN.json', 'r') as f2:
  data2 = json.load(f2)
hist2 = data2[str + '_loss_history']

with open('t32_256_BN.json', 'r') as f3:
  data3 = json.load(f3)
hist3 = data3[str + '_loss_history']

# with open('t1_len100.json', 'r') as f4:
#   data4 = json.load(f4)
# hist4 = data4[str + '_loss_history']


plt.plot(hist2, 'g-', hist3, 'b-')
plt.ylabel('val_loss')
plt.show()
