import json
import matplotlib.pyplot as plt


drawTrain = 1

str = 'val'
if drawTrain == 1: 
  str = 'train'

with open('t3_s128_multi_80.json', 'r') as f1:
  data1 = json.load(f1)
hist1 = data1[str + '_loss_history']
time1 = data1['forward_backward_times']
memory1 = data1['memory_usage']

# with open('t33_s517_nodp_10000.json', 'r') as f2:
#   data2 = json.load(f2)
# hist2 = data2[str + '_loss_history']
# time2 = data2['forward_backward_times']
# memory2 = data2['memory_usage']
#  
# with open('t7_s700_nodp_14240.json', 'r') as f3:
#   data3 = json.load(f3)
# hist3 = data3[str + '_loss_history']
# time3 = data3['forward_backward_times']
# memory3 = data3['memory_usage']
#    
# with open('t44_s517_nodp_14240.json', 'r') as f4:
#   data4 = json.load(f4)
# hist4 = data4[str + '_loss_history']
# time4 = data4['forward_backward_times']
# memory4 = data4['memory_usage']

plt.figure(1)
plt.plot(hist1, 'r-')
# plt.plot(hist1, 'r-', hist2, 'g-', hist3, 'b-', hist4, 'y-')
plt.ylabel('loss')

# plt.figure(2)
# # plt.plot(time1, 'r-', time2, 'g-')
# plt.plot(time1, 'r-', time2, 'g-', time3, 'b-')
# plt.ylabel('time')
#    
# plt.figure(3)
# # plt.plot(memory1, 'r-', memory2, 'g-')
# plt.plot(memory1, 'r-', memory2, 'g-', memory3, 'b-')
# plt.ylabel('memory')



plt.show()


