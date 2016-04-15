require 'torch'
require 'nn'

require 'TensHidden'
local gradcheck = require 'util.gradcheck'

isBN = 1
local tests = torch.TestSuite()
local tester = torch.Tester()


local function check_size(x, dims)
  tester:assert(x:dim() == #dims)
  for i, d in ipairs(dims) do
    tester:assert(x:size(i) == d)
  end
end


local function GetInputAndInitStateSizes(inputShape, tensShape, nodeSize, batchSize)

  local N, H = batchSize, nodeSize

  local sz_x = {N}
  for _, v in ipairs(inputShape) do
    table.insert(sz_x, v)
  end
  table.insert(sz_x, H)

  local sz_h = {N}
  for i, v in ipairs(inputShape) do
    if i ~= #inputShape then
      table.insert(sz_h, v)
    end
  end
  for _, v in ipairs(tensShape) do
    table.insert(sz_h, v)
  end
  table.insert(sz_h, H)

  return sz_x, sz_h
end


function tests.testForward()

  local inputShape = {3, 4}
  local tensShape = {2, 2, 2}
  local nodeSize = 3
  local batchSize = 2

  local hidden = nn.TensHidden(tensShape, nodeSize, isBN)

  local sz_x, sz_h = GetInputAndInitStateSizes(inputShape, tensShape, nodeSize, batchSize)
  local x  = torch.randn(torch.LongStorage(sz_x))
  local h0 = torch.randn(torch.LongStorage(sz_h))

  local y = hidden:forward({x, h0})
  y = y:clone()

  -- Do a naive forward pass
  local h = hidden.h
  local N, H = batchSize, nodeSize
  local w1  = hidden.weight[{{1, H}}]
  local w2  = hidden.weight[{{H + 1, 2 * H}}]

  local gamma1 = hidden.bias[{{1, 3 * H}}]:view(1, 3 * H):expand(N, 3 * H)
  local gamma2 = hidden.bias[{{3 * H + 1, 6 * H}}]:view(1, 3 * H):expand(N, 3 * H)
  local beta = hidden.bias[{{6 * H + 1, 9 * H}}]:view(1, 3 * H):expand(N, 3 * H)

  local coor = {}
  for i = 1, hidden.hiddenDim do
    coor[i] = 1
  end

  for nodeId = 1, hidden.nodeNum do
    local decompNodeId = (nodeId - 1) % hidden.decompNum + 1
    -- get the predecessor states
    local h1, mean1, var1, norm1 = hidden:GetPredecessorState(x, coor, hidden.hiddenDim, 1)
    local h2, mean2, var2, norm2 = hidden:GetPredecessorState(x, coor, hidden.hiddenDim - 1 - decompNodeId, 2)
    
    -- update the current node
    local gates, gates1, gates2
    if hidden.isBatchNorm then
      gates1 = torch.mm(h1, w1)
      gates2 = torch.mm(h2, w2)
      hidden:batchNormForward(gates1, mean1, var1, norm1)
      hidden:batchNormForward(gates2, mean2, var2, norm2)
      gates = torch.addcmul(beta, norm1, gamma1):addcmul(norm2, gamma2)
    else
      gates = torch.addmm(beta, h1, w1):addmm(h2, w2)
    end

    gates:narrow(2, 1, 2 * H):sigmoid() -- for gates
    gates:narrow(2, 2 * H + 1, H):tanh() -- for new content
    local f  = gates:narrow(2, 1, H)
    local s = gates:narrow(2, H + 1, H)
    local g = gates:narrow(2, 2 * H + 1, H)
    local hn = torch.cmul(s, h1):add(h2):addcmul(-1, s, h2):cmul(f):add(g):addcmul(-1, f, g)

    h[{{}, unpack(coor)}]:copy(hn)
    hidden:MoveCoor(coor, 1)
  end
  local naive_y = hidden._output:clone()

  tester:assertTensorEq(naive_y, y, 1e-10)
end

function tests.gradcheck()
  
  local inputShape = {4}
  local tensShape = {2}
  local nodeSize = 3
  local batchSize = 5

  local hidden = nn.TensHidden(tensShape, nodeSize, isBN)

  local sz_x, sz_h = GetInputAndInitStateSizes(inputShape, tensShape, nodeSize, batchSize)
  local x  = torch.randn(torch.LongStorage(sz_x))
  local h0 = torch.randn(torch.LongStorage(sz_h))

  local y = hidden:forward({x, h0})
  local dy = torch.randn(#y)

  hidden:zeroGradParameters()
  local dx, dh0 = unpack(hidden:backward({x, h0}, dy))
  dx = dx:clone()
  dh0 = dh0:clone()
  local dw = hidden.gradWeight:clone()
  local db = hidden.gradBias:clone()

  local function fx(x)   return hidden:forward{x, h0} end
  local function fh0(h0) return hidden:forward{x, h0} end

  local function fw(w)
    local old_w = hidden.weight
    hidden.weight = w
    local out = hidden:forward{x, h0}
    hidden.weight = old_w
    return out
  end

  local function fb(b)
    local old_b = hidden.bias
    hidden.bias = b
    local out = hidden:forward{x, h0}
    hidden.bias = old_b
    return out
  end

  local dx_num = gradcheck.numeric_gradient(fx, x, dy)
  local dh0_num = gradcheck.numeric_gradient(fh0, h0, dy)
  local dw_num = gradcheck.numeric_gradient(fw, hidden.weight, dy)
  local db_num = gradcheck.numeric_gradient(fb, hidden.bias, dy)

  local dx_error = gradcheck.relative_error(dx_num, dx);
  local dh0_error = gradcheck.relative_error(dh0_num, dh0)
  local dw_error = gradcheck.relative_error(dw_num, dw)
  local db_error = gradcheck.relative_error(db_num, db)

  tester:assertle(dh0_error, 1e-4)
  tester:assertle(dx_error, 1e-4)
  tester:assertle(dw_error, 1e-4)
  tester:assertle(db_error, 1e-4)
end


-- Make sure that everything works when we don't pass initial hidden state
-- in this case we only pass input sequence of vectors
function tests.noHiddenTest()

  local inputShape = {3, 4}
  local tensShape = {2, 2, 2}
  local nodeSize = 3
  local batchSize = 2

  local hidden = nn.TensHidden(tensShape, nodeSize, isBN)
  local sz_x, sz_h = GetInputAndInitStateSizes(inputShape, tensShape, nodeSize, batchSize)

  for t = 1, 3 do
    local x  = torch.randn(torch.LongStorage(sz_x))
    local dout = torch.randn(torch.LongStorage(sz_x))

    local out = hidden:forward(x)
    local din = hidden:backward(x, dout)

    tester:assert(torch.isTensor(din))
    check_size(din, sz_x)

    -- Make sure the initial cell state and initial hidden state are zero
    tester:assertTensorEq(hidden.h0, torch.zeros(torch.LongStorage(sz_h)), 0)
  end
end


function tests.rememberStatesTest()

  local inputShape = {3, 4}
  local tensShape = {2, 2, 2}
  local nodeSize = 3
  local batchSize = 2

  local hidden = nn.TensHidden(tensShape, nodeSize, isBN)
  hidden.remember_states = true
  local sz_x, sz_h = GetInputAndInitStateSizes(inputShape, tensShape, nodeSize, batchSize)

  local final_h = nil
  for t = 1, 4 do
    local x = torch.randn(torch.LongStorage(sz_x))
    local dout = torch.randn(torch.LongStorage(sz_x))
    local out = hidden:forward(x)
    local din = hidden:backward(x, dout)

    if t == 1 then
      tester:assertTensorEq(hidden.h0, torch.zeros(torch.LongStorage(sz_h)), 0)
    elseif t > 1 then
      tester:assertTensorEq(hidden.h0, final_h, 0)
    end
    final_h = hidden.h:select(1 + hidden.inputDim, hidden.inputShape[hidden.inputDim])
    final_h = final_h:select(final_h:dim() - 1, hidden.decompNum):clone()
  end

  -- Initial states should reset to zero after we call resetStates
  hidden:resetStates()
  local x = torch.randn(torch.LongStorage(sz_x))
  local dout = torch.randn(torch.LongStorage(sz_x))
  hidden:forward(x)
  hidden:backward(x, dout)
  tester:assertTensorEq(hidden.h0, torch.zeros(torch.LongStorage(sz_h)), 0)
end


tester:add(tests)
tester:run()
