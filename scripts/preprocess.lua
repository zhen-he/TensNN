require 'torch'
require 'hdf5'


local cmd = torch.CmdLine()
cmd:option('-input', 'tiny-shakespeare')
local opt = cmd:parse(arg)

local h5_file_name = opt.input .. '.h5'
local train_file_name = opt.input .. '_train.t7'
local val_file_name = opt.input .. '_val.t7'
local test_file_name = opt.input .. '_test.t7'

local f = hdf5.open(h5_file_name, 'r')
local train = f:read('/train'):all()
local val = f:read('/val'):all()
local test = f:read('/test'):all()

torch.save(train_file_name, train)
torch.save(val_file_name, val)
torch.save(test_file_name, test)


