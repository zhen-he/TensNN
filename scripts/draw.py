import json
import matplotlib.pyplot as plt

with open('cv1.json', 'r') as f_cv1:
  data_cv1 = json.load(f_cv1)
hist_cv1 = data_cv1['train_loss_history']
   
with open('cv3dru.json', 'r') as f_cv2:
  data_cv2 = json.load(f_cv2)
hist_cv2 = data_cv2['train_loss_history']
   
# with open('cv4.json', 'r') as f_cv4:
#   data_cv4 = json.load(f_cv4)
# hist_cv4 = data_cv4['train_loss_history']
#  
# with open('cv22.json', 'r') as f_cv22:
#   data_cv22 = json.load(f_cv22)
# hist_cv22 = data_cv22['train_loss_history']


# with open('cv_1.json', 'r') as f_cv_1:
#   data_cv_1 = json.load(f_cv_1)
# hist_cv_1 = data_cv_1['train_loss_history']
#    
# with open('cv_2.json', 'r') as f_cv_2:
#   data_cv_2 = json.load(f_cv_2)
# hist_cv_2 = data_cv_2['train_loss_history']
#    
# with open('cv_3.json', 'r') as f_cv_3:
#   data_cv_3 = json.load(f_cv_3)
# hist_cv_3 = data_cv_3['train_loss_history']


# with open('cv_1b.json', 'r') as f_cv_1b:
#   data_cv_1b = json.load(f_cv_1b)
# hist_cv_1b = data_cv_1b['train_loss_history']
#    
# with open('cv_2b.json', 'r') as f_cv_2b:
#   data_cv_2b = json.load(f_cv_2b)
# hist_cv_2b = data_cv_2b['train_loss_history']
#    
# with open('cv_3b.json', 'r') as f_cv_3b:
#   data_cv_3b = json.load(f_cv_3b)
# hist_cv_3b = data_cv_3b['train_loss_history']


# plt.plot(hist_cv_1b, 'r-', hist_cv_2b, 'g-', hist_cv_3b, 'b-')
# plt.plot(hist_cv_1, 'r-', hist_cv_2, 'g-', hist_cv_3, 'b-', hist_cv_1b, 'ro', hist_cv_2b, 'go', hist_cv_3b, 'bo')
plt.plot(hist_cv1, 'r-', hist_cv2, 'g-')
plt.ylabel('train_loss')
plt.show()
