require 'torch'
require 'nn'


local hidden, parent = torch.class('nn.TensHidden', 'nn.Module')


function hidden:__init(inputShape, tensShape, nodeSize, batchSize)
  parent.__init(self)

  self.inputShape = inputShape -- table
  self.tensShape = tensShape -- table
  self.nodeSize = nodeSize
  self.batchSize = batchSize

  self.inputDim = #self.inputShape
  self.tensDim = #self.tensShape
  self.decompNum = self.inputDim + self.tensDim - 1

  self.hiddenShape = {} -- table
  for _, v in ipairs(self.inputShape) do
    table.insert(self.hiddenShape, v)
  end
  for _, v in ipairs(self.tensShape) do
    table.insert(self.hiddenShape, v)
  end
  table.insert(self.hiddenShape, self.decompNum)
  self.hiddenDim = #self.hiddenShape
  self.totalDim = self.hiddenDim + 2 -- one for batch and one for node vector

  self.nodeNum = 1
  for _, v in ipairs(self.hiddenShape) do
      self.nodeNum = self.nodeNum * v
  end
  
  local H, N = self.nodeSize, self.batchSize
  self.weight = torch.Tensor(2 * H, 5 * H) -- input gate, forget gate1, forget gate2, output gate, new content
  self.gradWeight = torch.Tensor(2 * H, 5 * H):zero()
  self.bias = torch.Tensor(5 * H)
  self.gradBias = torch.Tensor(5 * H):zero()
  
  local sz = {N}
  for _, v in ipairs(self.hiddenShape) do
    table.insert(sz, v + 1)
  end
  table.insert(sz, 2 * H)
  self.buffer1 = torch.Tensor(unpack(sz)):zero()

  local stateRegion, stateGradRegion = {1, N, 1, H}, {1, N, 1, H}
  local outputRegion, gradInputRegion = {1, N, 1, H}, {1, N, 1, H}
  for i, v in ipairs(self.hiddenShape) do
    table.insert(stateRegion, 1); table.insert(stateRegion, v)
    table.insert(gradRegion, 2); table.insert(gradRegion, v + 1)
    if  i <= self.inputDim then
      table.insert(outputRegion, 1); table.insert(outputRegion, v)
      table.insert(gradInputRegion, 2); table.insert(gradInputRegion, v + 1)
    else
      table.insert(outputRegion, v); table.insert(outputRegion, v)
      table.insert(gradInputRegion, 2); table.insert(gradInputRegion, 2)
    end
  end
  local states = buffer1:sub(unpack(stateRegion))
  local stateGrads = buffer1:sub(unpack(stateGradRegion))
  self.output = buffer1:sub(unpack(outputRegion))
  self.gradInput = buffer1:sub(unpack(gradInputRegion))

  local S = self.totalDim
  self.h = states:narrow(S, 1, H)
  self.c = states:narrow(S, H + 1, H)
  self.grad_h = stateGrads:narrow(S, 1, H)
  self.grad_c = stateGrads:narrow(S, H + 1, H)

  for i, v in ipairs(self.hiddenShape) do
    sz[1 + i] = v
  end
  sz[#sz] = 5 * H
  self.gates = torch.Tensor(unpack(sz)):zero() -- This will be (N, unpack(self.hiddenShape), 5H)

  self.grad_b_sum = torch.Tensor(1, 5 * H) -- This will be (1, 5H)
  self.grad_a = torch.Tensor(N, 5 * H) -- This will be (N, 5H)

  self.h0 = torch.Tensor()
  self.c0 = torch.Tensor()
  self.remember_states = false

  self:reset()
end

-- reset weights and bias
function hidden:reset(std)
  if not std then
    std = 1.0 / math.sqrt(self.nodeSize * 2)
  end
  self.bias:zero()
  self.bias[{{self.nodeSize + 1, 3 * self.nodeSize}}]:fill(1) -- set the bias of forget gates to 1
  self.weight:normal(0, std)
  return self
end

-- reset h0 and c0
function hidden:resetStates()
  self.h0 = self.h0.new()
  self.c0 = self.c0.new()
end


local function check_dims(x, dims)
  assert(x:dim() == #dims)
  for i, d in ipairs(dims) do
    assert(x:size(i) == d)
  end
end

function hidden:CheckSize(input, gradOutput)
  assert(input:dim() == self.inputDim + 2)
  assert(input:size(1) == self.batchSize)
  for i, v in ipairs(self.inputShape) do
    assert(input:size(i + 1) == v)
  end
  assert(input:size(input:dim()) == self.nodeSize * 2)

  if gradOutput then
    assert(gradOutput:dim() == input:dim())
    for i, v in ipairs(self.inputShape) do
      assert(gradOutput:size(i) == v)
    end
  end
end


function hidden:_unpack_input(input)
  local c0, h0, x = nil, nil, nil
  if torch.type(input) == 'table' and #input == 3 then
    c0, h0, x = unpack(input)
  elseif torch.type(input) == 'table' and #input == 2 then
    h0, x = unpack(input)
  elseif torch.isTensor(input) then
    x = input
  else
    assert(false, 'invalid input')
  end
  return c0, h0, x
end


function hidden:_get_sizes(input, gradOutput)
  local c0, h0, x = self:_unpack_input(input)
  local N, T = x:size(1), x:size(2)
  local H = self.nodeSize
  check_dims(x, {N, T, H})
  if h0 or c0 then
    local sz = {N}
    for _, v in ipairs(self.tensShape) do
      table.insert(sz, v)
    end
    table.insert(sz, H)
    if h0 then
      check_dims(h0, sz)
    end
    if c0 then
      check_dims(c0, sz)
    end
  end
  if gradOutput then
    check_dims(gradOutput, {N, T, H})
  end
  return N, T, H
end

function self:MoveCoor(currentCoor, step) -- step should be 1 or -1

  for i = #currentCoor, 1, -1 do
    currentCoor[i] = currentCoor[i] + step
    if currentCoor[i] > 0 and currentCoor[i] <= self.hiddenShape[i] then
      break
    elseif currentCoor[i] == 0 then
        currentCoor[i] = self.hiddenShape[i]
    else
        currentCoor[i] = 1
    end
  end
  
  return currentCoor
end


function hidden:updateOutput(input)
  
  self:CheckSize(input)
  local x = input
  local H, N = self.nodeSize, self.batchSize
  local h, c = self.h, self.c
  local h0, c0 = self.h0, self.c0

  local h0_ = h:select(1 + self.inputDim, self.inputShape[self.inputDim])
  if h0:nElement() == 0 or not self.remember_states then -- first run or don't remember
    h0:resizeAs(h0_):zero()
  else -- if remember, use the previous evaluated h as h0
    h0 = h0_:clone()
  end

  local c0_ = c:select(1 + self.inputDim, self.inputShape[self.inputDim])
  if c0:nElement() == 0 or not self.remember_states then -- first run or don't remember
    c0:resizeAs(c0_):zero()
  else -- if remember, use the previous evaluated c as c0
    c0 = c0_:clone()
  end

  local bias_expand = self.bias:view(1, 5 * H):expand(N, 5 * H) -- copy the bias for a batch
  local w1 = self.weight[{{1, H}}] -- weights for h1
  local w2 = self.weight[{{H + 1, 2 * H}}] -- weights for h2
  
  -- initialize the coordinate of current node
  local coor = {}
  for i = 1, self.hiddenDim do
    table.insert(coor, 1)
  end

  local h1, c1 = torch.Tensor(), torch.Tensor()
  local h2, c2 = torch.Tensor(), torch.Tensor()

  for i = 1, self.nodeNum / self.decompNum do
    for j = 0, self.decompNum do  

      -- find a previous node
      local changedDim = self.hiddenDim - 1 - j -- the dimension where the coordinate need to be minus 1
      if coor[changedDim] > 1 then -- the usual case
        -- point to the previous node
        coor[changedDim] = coor[changedDim] - 1
        local coorv = coor[self.hiddenDim]
        coor[self.hiddenDim] = -1
        h2 = h[{{}, unpack(coor)}] -- N * H
        c2 = c[{{}, unpack(coor)}] -- N * H
        -- recover to the current coordinate
        coor[changedDim] = coor[changedDim] + 1
        coor[self.hiddenDim] = coorv
      else -- the case that requires initial states (out of the network's shape)
        h2:resize(N, H):zero()
        c2:resize(N, H):zero()
        if changedDim == self.inputDim then -- get value from the last states of previous batch
          table.remove(coor, changedDim)
          h2 = h0[{{}, unpack(coor)}] -- N * H
          c2 = c0[{{}, unpack(coor)}] -- N * H
          table.insert(coor, changedDim, 1)
        elseif changedDim == self.inputDim + self.tensDim then -- get value from input
          local isFromInput = true
          for i = self.inputDim + 1, self.inputDim + self.tensDim - 1 do
            if coor[i] ~= 1 then
              isFromInput = false
              break
            end
          end
          if isFromInput then
            local inputCoor = {}
            for i = 1, self.inputDim do
              table.insert(inputCoor, coor[i])
            end
            local h2c2 = input[{{}, unpack(inputCoor)}] -- N * 2H
            h2 = h2c2[{{}, {1, H}}] -- N * H
            c2 = h2c2[{{}, {H + 1, 2 * H}}] -- N * H
          end
        end
      end
      -- update the current node
      if j > 0 then
        local hn = h[{{}, unpack(coor)}] -- N * H
        local cn = c[{{}, unpack(coor)}] -- N * H
        local gates = self.gates[{{}, unpack(coor)}] -- N * 5H
        gates:addmm(bias_expand, h1, w1) -- w1 * h1 + b
        gates:addmm(h2, w2) -- w1 * h1 + b + w2 * x2
        gates[{{}, {1, 4 * H}}]:sigmoid() -- for gates
        gates[{{}, {4 * H + 1, 5 * H}}]:tanh() -- for new content
        local i  = cur_gates[{{}, {1, H}}]
        local f1 = cur_gates[{{}, {H + 1, 2 * H}}]
        local f2 = cur_gates[{{}, {2 * H + 1, 3 * H}}]
        local o  = cur_gates[{{}, {3 * H + 1, 4 * H}}]
        local g  = cur_gates[{{}, {4 * H + 1, 5 * H}}]
        hn:cmul(i, g) -- gated new contents
        cn:cmul(f1, c1):addcmul(f2, c2):add(hn) -- new memories
        hn:tanh(cn):cmul(o) -- new hidden states
        h1, c1 = hn, cn -- save
        coor = self:MoveCoor(coor, 1)
      else
        h1 = h2:clone()
        c1 = c2:clone()
      end
    end
  end

  return self.output
end


function hidden:backward(input, gradOutput, scale)

  self:CheckSize(input, gradOutput)
  scale = scale or 1.0
  local x = input
  local H, N = self.nodeSize, self.batchSize
  local h, c = self.h, self.c
  local h0, c0 = self.h0, self.c0

  local grad_y = gradOutput
  local grad_x = self.gradInput
  
  local w1 = self.weight[{{1, H}}]
  local w2 = self.weight[{{H + 1, 2 * H}}]
  local grad_w1 = self.gradWeight[{{1, H}}]
  local grad_w2 = self.gradWeight[{{H + 1, 2 * H}}]
  local grad_b = self.gradBias
  local grad_b_sum = self.grad_b_sum

  -- initialize the coordinate of current node
  local coor = {}
  for i = 1, self.hiddenDim do
    table.insert(coor, self.hiddenShape[i])
  end

  for i = self.nodeNum / self.decompNum, 1, -1 do
    for j = self.decompNum, 0, -1 do


    end
  end

  grad_x:resizeAs(x):zero()
  local grad_hn = self.grad_hn:zero()
  local grad_cn = self.grad_cn:zero()

  for t = T, 1, -1 do
    local next_h, next_c = h[{{}, t}], c[{{}, t}]
    local prev_h, prev_c = nil, nil
    if t == 1 then
      prev_h, prev_c = h0, c0
    else
      prev_h, prev_c = h[{{}, t - 1}], c[{{}, t - 1}]
    end
    grad_hn:add(grad_y[{{}, t}]) -- add the gradient from upper layer (not the next time step)

    local i = self.gates[{{}, t, {1, H}}]
    local f = self.gates[{{}, t, {H + 1, 2 * H}}]
    local o = self.gates[{{}, t, {2 * H + 1, 3 * H}}]
    local g = self.gates[{{}, t, {3 * H + 1, 4 * H}}]
    
    local grad_a = self.grad_a:zero() -- gradients of activations
    local grad_ai = grad_a[{{}, {1, H}}]
    local grad_af = grad_a[{{}, {H + 1, 2 * H}}]
    local grad_ao = grad_a[{{}, {2 * H + 1, 3 * H}}]
    local grad_ag = grad_a[{{}, {3 * H + 1, 4 * H}}]
    
    -- We will use grad_ai, grad_af, and grad_ao as temporary buffers
    -- to compute grad_cn. We will need tanh_next_c (stored in grad_ai)
    -- to compute grad_ao; the other values can be overwritten after we compute
    -- grad_cn
    local tanh_next_c = grad_ai:tanh(next_c) -- grad_ai is used as a buffer
    local tanh_next_c2 = grad_af:cmul(tanh_next_c, tanh_next_c) -- grad_af is used as a buffer
    local my_grad_cn = grad_ao -- grad_ao is used as a buffer
    my_grad_cn:fill(1):add(-1, tanh_next_c2):cmul(o):cmul(grad_hn)
    grad_cn:add(my_grad_cn) -- accumulate the gradient of cell from hidden state (not the next time step)
    
    -- We need tanh_next_c (currently in grad_ai) to compute grad_ao; after that we can overwrite it.
    grad_ao:fill(1):add(-1, o):cmul(o):cmul(tanh_next_c):cmul(grad_hn)

    -- Use grad_ai as a temporary buffer for computing grad_ag
    local g2 = grad_ai:cmul(g, g)
    grad_ag:fill(1):add(-1, g2):cmul(i):cmul(grad_cn)

    -- We don't need any temporary storage for these so do them last
    grad_ai:fill(1):add(-1, i):cmul(i):cmul(g):cmul(grad_cn)
    grad_af:fill(1):add(-1, f):cmul(f):cmul(prev_c):cmul(grad_cn)
    
    grad_x[{{}, t}]:mm(grad_a, w1:t()) -- (N by 4H) * (4H by D) = N by D
    grad_w1:addmm(scale, x[{{}, t}]:t(), grad_a) -- temporally accumulate the gradients of parameters, as they are shared
    grad_w2:addmm(scale, prev_h:t(), grad_a)
    self.grad_b_sum:sum(grad_a, 1) -- directly accumulate grad_b (equal to grad_a) inside a batch
    grad_b:add(scale, self.grad_b_sum)

    grad_hn:mm(grad_a, w2:t()) -- (N by 4H) * (4H by H) = N by H
    grad_cn:cmul(f)
  end

  return self.gradInput
end


function hidden:clearState()
  self.cell:set()
  self.gates:set()
  self.grad_hn:set()
  self.grad_cn:set()
  self.grad_b_sum:set()
  self.grad_a:set()

  self.grad_x:set()
  self.output:set()
end


function hidden:updateGradInput(input, gradOutput)
  return self:backward(input, gradOutput, 0)
end


function hidden:accGradParameters(input, gradOutput, scale)
  self:backward(input, gradOutput, scale)
end

