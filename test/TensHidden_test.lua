require 'torch'
require 'nn'

require 'TensHidden'
local gradcheck = require 'util.gradcheck'

dropout = nil
local tests = torch.TestSuite()
local tester = torch.Tester()


local function check_size(x, dims)
  tester:assert(x:dim() == #dims)
  for i, d in ipairs(dims) do
    tester:assert(x:size(i) == d)
  end
end


local function GetInputAndInitStateSizes(inputShape, tensShape, nodeSize, batchSize)

  local N, H, E = batchSize, nodeSize, #tensShape

  local sz_x = {N}
  for _, v in ipairs(inputShape) do
    table.insert(sz_x, v)
  end
  table.insert(sz_x, 2 * E * H)

  local sz_h = {N}
  for i, v in ipairs(inputShape) do
    if i > 1 then
      table.insert(sz_h, v)
    end
  end
  for _, v in ipairs(tensShape) do
    table.insert(sz_h, v)
  end
  table.insert(sz_h, H)

  return sz_x, sz_h
end


function tests.gradcheck()
  
  local inputShape = {3}
  local tensShape = {3,3,3}
  local nodeSize = 3
  local batchSize = 3

  local hidden = nn.TensHidden(inputShape,tensShape, nodeSize, dropout)

  local sz_x, sz_h = GetInputAndInitStateSizes(inputShape, tensShape, nodeSize, batchSize)
  local x  = torch.randn(torch.LongStorage(sz_x))--:fill(2)
  local h0 = torch.randn(torch.LongStorage(sz_h))--:fill(2)
  local c0 = torch.randn(torch.LongStorage(sz_h))--:fill(2)
--  hidden.weight:fill(1)
--  hidden.bias:fill(1)
  local y = hidden:forward({x, h0, c0})
--  print(y)
  local dy = torch.randn(#y)--:fill(1)

  hidden:zeroGradParameters()
  local dx, dh0, dc0 = unpack(hidden:backward({x, h0, c0}, dy))
  dx = dx:clone()
  dh0 = dh0:clone()
  dc0 = dc0:clone()--;print(dc0)
  local dw = hidden.gradWeight:clone()--;print(dw)
  local db = hidden.gradBias:clone()--;print(db)

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

  if dropout then -- freeze the drop out noise to fix the network
    hidden.freezeNoise = true 
  end
  
  local dx_num = gradcheck.numeric_gradient(fx, x, dy)
  local dh0_num = gradcheck.numeric_gradient(fh0, h0, dy)
  local dc0_num = gradcheck.numeric_gradient(fc0, c0, dy)
  local dw_num = gradcheck.numeric_gradient(fw, hidden.weight, dy)
  local db_num = gradcheck.numeric_gradient(fb, hidden.bias, dy)

  local dx_error = gradcheck.relative_error(dx_num, dx)--;print(dx_num, dx)
  local dh0_error = gradcheck.relative_error(dh0_num, dh0)--;print(dh0_num, dh0)
  local dc0_error = gradcheck.relative_error(dc0_num, dc0)--;print(dc0_num, dc0)
  local dw_error = gradcheck.relative_error(dw_num, dw)--;print(dw_num, dw)
  local db_error = gradcheck.relative_error(db_num, db)--;print(db_num, db)

  tester:assertle(dh0_error, 1e-4)
  tester:assertle(dc0_error, 1e-4)
  tester:assertle(dx_error, 1e-4)
  tester:assertle(dw_error, 1e-4)
  tester:assertle(db_error, 1e-4)
end


-- Make sure that everything works correctly when we don't pass an initial cell
-- state; in this case we do pass an initial hidden state and an input sequence
function tests.noCellTest()

  local inputShape = {5}
  local tensShape = {3,3,3}
  local nodeSize = 3
  local batchSize = 2

  local hidden = nn.TensHidden(inputShape,tensShape, nodeSize, dropout)
  local sz_x, sz_h = GetInputAndInitStateSizes(inputShape, tensShape, nodeSize, batchSize)

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

  local inputShape = {5}
  local tensShape = {3, 3, 3}
  local nodeSize = 3
  local batchSize = 2

  local hidden = nn.TensHidden(inputShape,tensShape, nodeSize, dropout)
  local sz_x, sz_h = GetInputAndInitStateSizes(inputShape, tensShape, nodeSize, batchSize)

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

  local inputShape = {4}
  local tensShape = {3,3}
  local nodeSize = 3
  local batchSize = 2

  local hidden = nn.TensHidden(inputShape,tensShape, nodeSize, dropout)
  hidden.remember_states = true
  local sz_x, sz_h = GetInputAndInitStateSizes(inputShape, tensShape, nodeSize, batchSize)

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
    
    final_h, final_c = hidden:GetInitialStates()
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
