require 'torch'
require 'hdf5'


local header_name = 'enwik8'

local h5_file_name = dir_name .. '.h5'
local train_file_name = dir_name .. '_train.t7'
local val_file_name = dir_name .. '_val.t7'
local test_file_name = dir_name .. '_test.t7'

local f = hdf5.open(h5_file_name, 'r')
local train = f:read('/train'):all():float()
local val = f:read('/val'):all():float()
local test = f:read('/test'):all():float()

torch.save(train_file_name, train)
torch.save(val_file_name, val)
torch.save(test_file_name, test)


