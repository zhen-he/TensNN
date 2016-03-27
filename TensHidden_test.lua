require 'torch'
require 'nn'

require 'TensHidden'
local gradcheck = require 'util.gradcheck'


local tests = {}
local tester = torch.Tester()


local function check_size(x, dims)
  tester:assert(x:dim() == #dims)
  for i, d in ipairs(dims) do
    tester:assert(x:size(i) == d)
  end
end


function tests.testForward()

  local inputShape = {7}
  local tensShape = {5}
  local nodeSize = 3
  local batchSize = 2

  local hidden = nn.TensHidden(inputShape, tensShape, nodeSize, batchSize)

  local N, H = batchSize, nodeSize
  local sz = {N}
  for _, v in ipairs(hidden.inputShape) do
    table.insert(sz, v)
  end
  table.insert(sz, 2 * H)
  local x  = torch.randn(torch.LongStorage(sz))

  sz = {N}
  for i, v in ipairs(hidden.hiddenShape) do
    if i ~= hidden.inputDim and i ~= hidden.hiddenDim then
      table.insert(sz, v)
    end
  end
  table.insert(sz, H)
  local h0 = torch.randn(torch.LongStorage(sz))
  local c0 = torch.randn(torch.LongStorage(sz))

  local y = hidden:forward({x, h0, c0})
  y = y:clone()

  -- Do a naive forward pass
  local h, c = hidden.h, hidden.c
  
  local w1i  = hidden.weight[{{1, H}, {1, H}}]
  local w1f1 = hidden.weight[{{1, H}, {H + 1, 2 * H}}]
  local w1f2 = hidden.weight[{{1, H}, {2 * H + 1, 3 * H}}]
  local w1o  = hidden.weight[{{1, H}, {3 * H + 1, 4 * H}}]
  local w1g  = hidden.weight[{{1, H}, {4 * H + 1, 5 * H}}]
  
  local w2i  = hidden.weight[{{H + 1, 2 * H}, {1, H}}]
  local w2f1 = hidden.weight[{{H + 1, 2 * H}, {H + 1, 2 * H}}]
  local w2f2 = hidden.weight[{{H + 1, 2 * H}, {2 * H + 1, 3 * H}}]
  local w2o  = hidden.weight[{{H + 1, 2 * H}, {3 * H + 1, 4 * H}}]
  local w2g  = hidden.weight[{{H + 1, 2 * H}, {4 * H + 1, 5 * H}}]
  
  local bi  = hidden.bias[{{1, H}}]:view(1, H):expand(N, H)
  local bf1 = hidden.bias[{{H + 1, 2 * H}}]:view(1, H):expand(N, H)
  local bf2 = hidden.bias[{{2 * H + 1, 3 * H}}]:view(1, H):expand(N, H)
  local bo  = hidden.bias[{{3 * H + 1, 4 * H}}]:view(1, H):expand(N, H)
  local bg  = hidden.bias[{{4 * H + 1, 5 * H}}]:view(1, H):expand(N, H)

  local coor = {}
  for i = 1, hidden.hiddenDim do
    coor[i] = 1
  end

  for nodeId = 1, hidden.nodeNum do
    local decompNodeId = (nodeId - 1) % hidden.decompNum + 1
    -- get the predecessor states
    local h1, c1 = hidden:GetPredecessorState(x, coor, hidden.hiddenDim)
    local h2, c2 = hidden:GetPredecessorState(x, coor, hidden.hiddenDim - 1 - decompNodeId)
    -- update the current node
    local i  = torch.sigmoid(torch.mm(h1, w1i)  + torch.mm(h2, w2i)  + bi)
    local f1 = torch.sigmoid(torch.mm(h1, w1f1) + torch.mm(h2, w2f1) + bf1)
    local f2 = torch.sigmoid(torch.mm(h1, w1f2) + torch.mm(h2, w2f2) + bf2)
    local o  = torch.sigmoid(torch.mm(h1, w1o)  + torch.mm(h2, w2o)  + bo)
    local g  =    torch.tanh(torch.mm(h1, w1g)  + torch.mm(h2, w2g)  + bg)
    local cn = torch.cmul(f1, c1) + torch.cmul(f2, c2) + torch.cmul(i, g)
    local hn = torch.cmul(o, torch.tanh(cn))
    h[{{}, unpack(coor)}]:copy(hn)
    c[{{}, unpack(coor)}]:copy(cn)
    hidden:MoveCoor(coor, 1)
  end
  local naive_y = hidden.output:clone()

  tester:assertTensorEq(naive_y, y, 1e-10)
end

function tests.gradcheck()
  
  local inputShape = {1}
  local tensShape = {1}
  local nodeSize = 1
  local batchSize = 1

  local hidden = nn.TensHidden(inputShape, tensShape, nodeSize, batchSize)

  local N, H = batchSize, nodeSize
  local sz = {N}
  for _, v in ipairs(hidden.inputShape) do
    table.insert(sz, v)
  end
  table.insert(sz, 2 * H)
  local x  = torch.randn(torch.LongStorage(sz)):fill(1);print('\nx:',x)

  sz = {N}
  for i, v in ipairs(hidden.hiddenShape) do
    if i ~= hidden.inputDim and i ~= hidden.hiddenDim then
      table.insert(sz, v)
    end
  end
  table.insert(sz, H)
  local h0 = torch.randn(torch.LongStorage(sz)):fill(2);print('\nh0:',h0)
  local c0 = torch.randn(torch.LongStorage(sz)):fill(2);print('\nc0:',c0)
--hidden.weight:fill(1);hidden.bias:fill(1)
  print('\nweght:',hidden.weight);print('\nbias:',hidden.bias)
  local y = hidden:forward({x, h0, c0});print('\ny:',y)
  local dy = torch.randn(#y):fill(1);print('\ndy:',dy)

  hidden:zeroGradParameters()
  local dx, dh0, dc0 = unpack(hidden:backward({x, h0, c0}, dy))
  dx = dx:clone();print('\ndx:',dx)
  dh0 = dh0:clone();print('\ndh0:',dh0)
  dc0 = dc0:clone();print('\ndc0:',dc0)
  local dw = hidden.gradWeight:clone();print('\ndw:',dw)
  local db = hidden.gradBias:clone();print('\ndb:',db)

  local function fx(x)   return hidden:forward{x, h0, c0} end
  local function fh0(h0) return hidden:forward{x, h0, c0} end
  local function fc0(c0) return hidden:forward{x, h0, c0} end

  local function fw(w)
    local old_w = hidden.weight
    hidden.weight = w
    local out = hidden:forward{x, h0, c0}
    hidden.weight = old_w
    return out
  end

  local function fb(b)
    local old_b = hidden.bias
    hidden.bias = b
    local out = hidden:forward{x, h0, c0}
    hidden.bias = old_b
    return out
  end

  local dx_num = gradcheck.numeric_gradient(fx, x, dy)
  local dh0_num = gradcheck.numeric_gradient(fh0, h0, dy)
  local dc0_num = gradcheck.numeric_gradient(fc0, c0, dy)
  local dw_num = gradcheck.numeric_gradient(fw, hidden.weight, dy)
  local db_num = gradcheck.numeric_gradient(fb, hidden.bias, dy)

  local dx_error = gradcheck.relative_error(dx_num, dx);
  local dh0_error = gradcheck.relative_error(dh0_num, dh0)
  local dc0_error = gradcheck.relative_error(dc0_num, dc0)
  local dw_error = gradcheck.relative_error(dw_num, dw)
  local db_error = gradcheck.relative_error(db_num, db)

  tester:assertle(dh0_error, 1e-4)
  tester:assertle(dc0_error, 1e-5)
  tester:assertle(dx_error, 1e-5)
  tester:assertle(dw_error, 1e-4)
  tester:assertle(db_error, 1e-5)
end



tester:add(tests)
tester:run()
