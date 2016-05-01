import 'torch'
require 'cutorch'

N = 100

--local a = torch.randn(N, N):fill(1):cuda()
--local b = torch.randn(N, N):fill(1):cuda()
--local at = torch.Tensor(1000 * N, N):fill(1):cuda()
--local bt = torch.Tensor(1000 * N, N):fill(1):cuda()


--timer = torch.Timer() -- start the Timer

--at:cmul(bt):cmul(bt):cmul(bt):cmul(bt):exp():sqrt()
--at:sqrt():pow(2):sqrt():pow(2):sqrt():pow(2):sqrt():pow(2):sqrt():pow(2):sqrt():pow(2)
--
--print('Time elapsed: ' .. timer:time().real * 1e3 .. ' msec.') -- 4 ms on my machine
--
--
--
--
--
--timer:reset()
--
--for i = 1, 1000 do
--  a:cmul(b):cmul(b):cmul(b):cmul(b):exp():sqrt()
--  a:sqrt():pow(2):sqrt():pow(2):sqrt():pow(2):sqrt():pow(2):sqrt():pow(2):sqrt():pow(2)
--end
--
--print('Time elapsed: ' .. timer:time().real * 1e3 .. ' msec.') -- 0.04 ms!! on my machine




local aa=torch.randn(450,128):fill(5):cuda()
local cc=torch.randn(450,128):fill(5):cuda()

timer = torch.Timer()
--local bb = cc:tanh(aa)
local bb = cc:copy(aa):tanh()
print('Time elapsed: ' .. timer:time().real * 1e3 .. ' msec.')

