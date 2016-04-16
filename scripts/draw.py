import json
import matplotlib.pyplot as plt
   
with open('t3_256.json', 'r') as f1:
  data1 = json.load(f1)
hist1 = data1['val_loss_history']

with open('checkpoint_4000.json', 'r') as f2:
  data2 = json.load(f2)
hist2 = data2['val_loss_history']


plt.plot(hist1, 'r-', hist2, 'g-')
plt.ylabel('val_loss')
plt.show()
