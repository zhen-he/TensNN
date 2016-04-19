import json
import matplotlib.pyplot as plt


drawTrain = 0

str = 'val'
if drawTrain == 1: 
  str = 'train'

with open('t1_s128_noBN_10680.json', 'r') as f1:
  data1 = json.load(f1)
hist1 = data1[str + '_loss_history']
 
with open('t1_s128_inputBN_10680.json', 'r') as f2:
  data2 = json.load(f2)
hist2 = data2[str + '_loss_history']

with open('t1_s128_tensorBN_1000.json', 'r') as f3:
  data3 = json.load(f3)
hist3 = data3[str + '_loss_history']

# with open('t1_len100.json', 'r') as f4:
#   data4 = json.load(f4)
# hist4 = data4[str + '_loss_history']


plt.plot(hist1, 'r-', hist2, 'g-', hist3, 'b-')
plt.ylabel('val_loss')
plt.show()
