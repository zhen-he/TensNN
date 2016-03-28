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

  local inputShape = {3,4}
  local tensShape = {2,3}
  local nodeSize = 3
  local batchSize = 2

  local hidden = nn.TensHidden(inputShape, tensShape, nodeSize, batchSize)

  local N, H = batchSize, nodeSize
  local sz_x = {N}
  for _, v in ipairs(hidden.inputShape) do
    table.insert(sz_x, v)
  end
  table.insert(sz_x, 2 * H)
  local x  = torch.randn(torch.LongStorage(sz_x))

  local sz_h = {N}
  for i, v in ipairs(hidden.hiddenShape) do
    if i ~= hidden.inputDim and i ~= hidden.hiddenDim then
      table.insert(sz_h, v)
    end
  end
  table.insert(sz_h, H)
  local h0 = torch.randn(torch.LongStorage(sz_h))
  local c0 = torch.randn(torch.LongStorage(sz_h))

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
  
  local inputShape = {3,3}
  local tensShape = {2,2}
  local nodeSize = 2
  local batchSize = 2

  local hidden = nn.TensHidden(inputShape, tensShape, nodeSize, batchSize)

  local N, H = batchSize, nodeSize
  local sz_x = {N}
  for _, v in ipairs(hidden.inputShape) do
    table.insert(sz_x, v)
  end
  table.insert(sz_x, 2 * H)
  local x  = torch.randn(torch.LongStorage(sz_x))

  local sz_h = {N}
  for i, v in ipairs(hidden.hiddenShape) do
    if i ~= hidden.inputDim and i ~= hidden.hiddenDim then
      table.insert(sz_h, v)
    end
  end
  table.insert(sz_h, H)
  local h0 = torch.randn(torch.LongStorage(sz_h))
  local c0 = torch.randn(torch.LongStorage(sz_h))

  local y = hidden:forward({x, h0, c0})
  local dy = torch.randn(#y)

  hidden:zeroGradParameters()
  local dx, dh0, dc0 = unpack(hidden:backward({x, h0, c0}, dy))
  dx = dx:clone()
  dh0 = dh0:clone()
  dc0 = dc0:clone()
  local dw = hidden.gradWeight:clone()
  local db = hidden.gradBias:clone()

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
  tester:assertle(dc0_error, 1e-4)
  tester:assertle(dx_error, 1e-4)
  tester:assertle(dw_error, 1e-4)
  tester:assertle(db_error, 1e-4)
end


-- Make sure that everything works correctly when we don't pass an initial cell
-- state; in this case we do pass an initial hidden state and an input sequence
function tests.noCellTest()

  local inputShape = {3,4}
  local tensShape = {2,3}
  local nodeSize = 3
  local batchSize = 2

  local hidden = nn.TensHidden(inputShape, tensShape, nodeSize, batchSize)

  local N, H = batchSize, nodeSize

  local sz_x = {N}
  for _, v in ipairs(hidden.inputShape) do
    table.insert(sz_x, v)
  end
  table.insert(sz_x, 2 * H)

  local sz_h = {N}
  for i, v in ipairs(hidden.hiddenShape) do
    if i ~= hidden.inputDim and i ~= hidden.hiddenDim then
      table.insert(sz_h, v)
    end
  end
  table.insert(sz_h, H)

  for t = 1, 3 do
    local x  = torch.randn(torch.LongStorage(sz_x))
    local h0 = torch.randn(torch.LongStorage(sz_h))
    local dout = torch.randn(torch.LongStorage(sz_x))

    local out = hidden:forward{x, h0}
    local din = hidden:backward({x, h0}, dout)

    tester:assert(torch.type(din) == 'table')
    tester:assert(#din == 2)
    check_size(din[1], sz_x)
    check_size(din[2], sz_h)

    -- Make sure the initial cell state got reset to zero
    tester:assertTensorEq(hidden.c0, torch.zeros(torch.LongStorage(sz_h)), 0)
  end
end


-- Make sure that everything works when we don't pass initial hidden or initial
-- cell state; in this case we only pass input sequence of vectors
function tests.noHiddenTest()

  local inputShape = {3,4}
  local tensShape = {2,3}
  local nodeSize = 3
  local batchSize = 2

  local hidden = nn.TensHidden(inputShape, tensShape, nodeSize, batchSize)

  local N, H = batchSize, nodeSize

  local sz_x = {N}
  for _, v in ipairs(hidden.inputShape) do
    table.insert(sz_x, v)
  end
  table.insert(sz_x, 2 * H)

  local sz_h = {N}
  for i, v in ipairs(hidden.hiddenShape) do
    if i ~= hidden.inputDim and i ~= hidden.hiddenDim then
      table.insert(sz_h, v)
    end
  end
  table.insert(sz_h, H)

  for t = 1, 3 do
    local x  = torch.randn(torch.LongStorage(sz_x))
    local dout = torch.randn(torch.LongStorage(sz_x))

    local out = hidden:forward(x)
    local din = hidden:backward(x, dout)

    tester:assert(torch.isTensor(din))
    check_size(din, sz_x)

    -- Make sure the initial cell state and initial hidden state are zero
    tester:assertTensorEq(hidden.c0, torch.zeros(torch.LongStorage(sz_h)), 0)
    tester:assertTensorEq(hidden.h0, torch.zeros(torch.LongStorage(sz_h)), 0)
  end
end


function tests.rememberStatesTest()

  local inputShape = {3,4}
  local tensShape = {2,3}
  local nodeSize = 3
  local batchSize = 2

  local hidden = nn.TensHidden(inputShape, tensShape, nodeSize, batchSize)
  hidden.remember_states = true

  local N, H = batchSize, nodeSize

  local sz_x = {N}
  for _, v in ipairs(hidden.inputShape) do
    table.insert(sz_x, v)
  end
  table.insert(sz_x, 2 * H)

  local sz_h = {N}
  for i, v in ipairs(hidden.hiddenShape) do
    if i ~= hidden.inputDim and i ~= hidden.hiddenDim then
      table.insert(sz_h, v)
    end
  end
  table.insert(sz_h, H)

  local final_h, final_c = nil, nil
  for t = 1, 4 do
    local x = torch.randn(torch.LongStorage(sz_x))
    local dout = torch.randn(torch.LongStorage(sz_x))
    local out = hidden:forward(x)
    local din = hidden:backward(x, dout)

    if t == 1 then
      tester:assertTensorEq(hidden.c0, torch.zeros(torch.LongStorage(sz_h)), 0)
      tester:assertTensorEq(hidden.h0, torch.zeros(torch.LongStorage(sz_h)), 0)
    elseif t > 1 then
      tester:assertTensorEq(hidden.c0, final_c, 0)
      tester:assertTensorEq(hidden.h0, final_h, 0)
    end
    final_c = hidden.c:select(1 + hidden.inputDim, hidden.inputShape[hidden.inputDim])
    final_c = final_c:select(final_c:dim() - 1, hidden.decompNum):clone()
    final_h = hidden.h:select(1 + hidden.inputDim, hidden.inputShape[hidden.inputDim])
    final_h = final_h:select(final_h:dim() - 1, hidden.decompNum):clone()
  end

  -- Initial states should reset to zero after we call resetStates
  hidden:resetStates()
  local x = torch.randn(torch.LongStorage(sz_x))
  local dout = torch.randn(torch.LongStorage(sz_x))
  hidden:forward(x)
  hidden:backward(x, dout)
  tester:assertTensorEq(hidden.c0, torch.zeros(torch.LongStorage(sz_h)), 0)
  tester:assertTensorEq(hidden.h0, torch.zeros(torch.LongStorage(sz_h)), 0)
end


tester:add(tests)
tester:run()
