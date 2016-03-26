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

  local inputShape = {10}
  local tensShape = {4,3}
  local nodeSize = 8
  local batchSize = 5

  local hidden = nn.TensHidden(inputShape, tensShape, nodeSize, batchSize)

  local N, H = batchSize, nodeSize
  local sz = {N}
  for _, v in ipairs(hidden.inputShape) do
    table.insert(sz, v)
  end
  table.insert(sz, 2 * H)
  local x  = torch.randn(unpack(sz))

  local output = hidden:forward(x)
  output = output:clone()

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
    coor = hidden:MoveCoor(coor, 1)
  end
  local naive_output = hidden.output:clone()

  tester:assertTensorEq(naive_output, output, 1e-10)
end


tester:add(tests)
tester:run()

