require 'torch'
require 'nn'


local hidden, parent = torch.class('nn.TensHidden', 'nn.Module')


function hidden:__init(inputShape, tensShape, nodeSize, batchSize)

  assert(#inputShape > 0, 'invalid input shape')
  assert(#tensShape > 0, 'invalid tensorizing shape')
  assert(nodeSize > 0, 'invalid node size')
  assert(batchSize > 0, 'invalid batch size')

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
  
  local H = self.nodeSize
  self.weight = torch.Tensor(2 * H, 5 * H) -- input gate, forget gate1, forget gate2, output gate, new content
  self.gradWeight = torch.Tensor(2 * H, 5 * H):zero()
  self.bias = torch.Tensor(5 * H)
  self.gradBias = torch.Tensor(5 * H):zero()
  self:reset()
  
  self.stateAndGradBuff = torch.Tensor()
  self.gates = torch.Tensor() -- This will be (N, unpack(self.hiddenShape), 5H)
  self.grad_b_sum = torch.Tensor() -- This will be (1, 5H)
  self.grad_a = torch.Tensor() -- This will be (N, 5H)
  self.gradIndicator = torch.ByteTensor()

  self.h0 = torch.Tensor()
  self.c0 = torch.Tensor()
  self.remember_states = false
end


function hidden:reset(std)
  -- initialize weights and bias

  if not std then
    std = 1.0 / math.sqrt(self.nodeSize * 2)
  end
  self.bias:zero()
  self.bias[{{self.nodeSize + 1, 3 * self.nodeSize}}]:fill(1) -- set the bias of forget gates to 1
  self.weight:normal(0, std)
  return self
end


function hidden:resetStates()
  -- clear initial states h0 and c0

  self.h0 = self.h0.new()
  self.c0 = self.c0.new()
end


function hidden:InitState()

  local H, N = self.nodeSize, self.batchSize

  if self.stateAndGradBuff:nElement() == 0 then
    local sz = {N}
    for _, v in ipairs(self.hiddenShape) do
      table.insert(sz, v + 1)
    end
    table.insert(sz, 2 * H)
    self.stateAndGradBuff:resize(unpack(sz)):zero()

    local stateRegion, gradRegion = {1, N}, {1, N}
    local outputRegion, gradInputRegion, gradOutputRegion = {1, N}, {1, N}, {1, N}
    for i, v in ipairs(self.hiddenShape) do
      table.insert(stateRegion, 1); table.insert(stateRegion, v)
      table.insert(gradRegion, 1); table.insert(gradRegion, v + 1)
      if i <= self.inputDim then
        table.insert(outputRegion, 1); table.insert(outputRegion, v)
        table.insert(gradInputRegion, 2); table.insert(gradInputRegion, v + 1)
        table.insert(gradOutputRegion, 2); table.insert(gradOutputRegion, v + 1)
      else
        table.insert(outputRegion, v); table.insert(outputRegion, v)
        table.insert(gradInputRegion, 1); table.insert(gradInputRegion, 1)
        table.insert(gradOutputRegion, v + 1); table.insert(gradOutputRegion, v + 1)
      end
    end
    local states = self.stateAndGradBuff:sub(unpack(stateRegion))
    local grads = self.stateAndGradBuff:sub(unpack(gradRegion))
    self.output = self.stateAndGradBuff:sub(unpack(outputRegion))
    self.gradInput = self.stateAndGradBuff:sub(unpack(gradInputRegion))
    self.gradOutput = self.stateAndGradBuff:sub(unpack(gradOutputRegion))

    local S = self.totalDim
    self.h = states:narrow(S, 1, H)
    self.c = states:narrow(S, H + 1, H)
    self.grad_h = grads:narrow(S, 1, H)
    self.grad_c = grads:narrow(S, H + 1, H)
  end

  if self.gates:nElement() == 0 then
    local sz = {N}
    for _, v in ipairs(self.hiddenShape) do
      table.insert(sz, v)
    end
    table.insert(sz, 5 * H)
    self.gates:resize(unpack(sz)):zero() -- This will be (N, unpack(self.hiddenShape), 5H)
  end

  if self.grad_b_sum:nElement() == 0 then
    self.grad_b_sum:resize(1, 5 * H):zero() -- This will be (1, 5H)
  end

  if self.grad_a:nElement() == 0 then
    self.grad_a:resize(N, 5 * H):zero() -- This will be (N, 5H)
  end

  if self.gradIndicator:nElement() == 0 then
    sz = {}
    for _, v in ipairs(self.hiddenShape) do
      table.insert(sz, v + 1)
    end
    self.gradIndicator:resize(unpack(sz)):zero()

    local gradOutputRegion = {}
    for i, v in ipairs(self.hiddenShape) do
      if i <= self.inputDim then
        table.insert(gradOutputRegion, 2); table.insert(gradOutputRegion, v + 1)
      else
        table.insert(gradOutputRegion, v + 1); table.insert(gradOutputRegion, v + 1)
      end
    end
    self.gradIndicator:sub(unpack(gradOutputRegion)):fill(1)
  end
end


function hidden:clearState()
  -- clear intermediate variables (the original clearState() in 'nn.Module' is overloaded, 
  -- as it only clears 'output' and 'gradInput')

  self.stateAndGradBuff:set()
  self.gates:set()
  self.grad_b_sum:set()
  self.grad_a:set()
  self.gradIndicator:set()
end


function hidden:CheckSize(input, gradOutput)

  assert(input:dim() >= 3) -- batch, input dim, node vector
  assert(input:dim() == self.inputDim + 2)
  assert(input:size(1) == self.batchSize)
  for i, v in ipairs(self.inputShape) do
    assert(input:size(i + 1) == v)
  end
  assert(input:size(input:dim()) == self.nodeSize * 2) -- hidden vector and cell vector

  if gradOutput then
    assert(gradOutput:dim() == input:dim())
    for i, v in ipairs(input) do
      assert(gradOutput:size(i) == v)
    end
  end
end


function hidden:MoveCoor(curCoor, step) -- step must be 1 or -1

  for i = #curCoor, 1, -1 do
    curCoor[i] = curCoor[i] + step
    if curCoor[i] > 0 and curCoor[i] <= self.hiddenShape[i] then
      break
    elseif curCoor[i] == 0 then
      curCoor[i] = self.hiddenShape[i]
    else
      curCoor[i] = 1
    end
  end
end


function hidden:GetPredecessorState(input, curCoor, predecessorDim)

  local x = input
  local H, N = self.nodeSize, self.batchSize
  local h, c = self.h, self.c
  local h0, c0 = self.h0, self.c0
  local hp, cp = torch.Tensor(), torch.Tensor()

  local preCoor = {}
  for i, v in ipairs(curCoor) do
    preCoor[i] = v
  end

  if preCoor[predecessorDim] > 1 then
    -- point to the previous node
    preCoor[predecessorDim] = preCoor[predecessorDim] - 1
    if predecessorDim < self.hiddenDim then -- if not along the decomposing dimension
      preCoor[self.hiddenDim] = self.decompNum -- point to the last decompesed one of the other dimension
    end
    hp = h[{{}, unpack(preCoor)}] -- N * H
    cp = c[{{}, unpack(preCoor)}] -- N * H
  else -- the case that requires initial states (out of the network's shape)
    predecessorDim = math.min(predecessorDim, self.hiddenDim - 1)
    hp:resize(N, H):zero()
    cp:resize(N, H):zero()
    if predecessorDim == self.inputDim then -- get value from the last states of previous batch
      table.remove(preCoor, predecessorDim)
      hp = h0[{{}, unpack(preCoor)}] -- N * H
      cp = c0[{{}, unpack(preCoor)}] -- N * H
    elseif predecessorDim == self.inputDim + self.tensDim then -- get value from input
      local isFromInput = true
      for i = self.inputDim + 1, self.inputDim + self.tensDim - 1 do
        if preCoor[i] ~= 1 then
          isFromInput = false
          break
        end
      end
      if isFromInput then
        local inputCoor = {}
        for i = 1, self.inputDim do
          inputCoor[i] = preCoor[i]
        end
        local hpcp = x[{{}, unpack(inputCoor)}] -- N * 2H
        hp = hpcp[{{}, {1, H}}] -- N * H
        cp = hpcp[{{}, {H + 1, 2 * H}}] -- N * H
      end
    end
  end

  return hp, cp
end


function hidden:GetPredecessorGrad(curCoor, predecessorDim, gradIndicator)

  local preCoor = {}
  for i, v in ipairs(curCoor) do
    preCoor[i] = v
  end

  if preCoor[predecessorDim] == 1 then -- the case that requires initial states (out of the network's shape)
    predecessorDim = math.min(predecessorDim, self.hiddenDim - 1)
  end
  if predecessorDim < self.hiddenDim then -- if not along the decomposing dimension
    preCoor[self.hiddenDim] = self.decompNum -- point to the last decompesed one of the other dimension
  end
  preCoor[predecessorDim] = preCoor[predecessorDim] - 1

  -- as the memory of states and state gradients are shared, we shift each dimension's
  -- coordinate by 1 to avoid the states being overwritten by gradients 
  for i, v in ipairs(preCoor) do
    preCoor[i] = v + 1
  end

  local grad_hp = self.grad_h[{{}, unpack(preCoor)}] -- N * H
  local grad_cp = self.grad_c[{{}, unpack(preCoor)}] -- N * H
  
  -- if there's no gradient in a predecessor, we overwrite it with a back propagated gradient,
  -- otherwise we accumulate the back propagated gradient
  local isGrad = true
  if gradIndicator[preCoor] == 0 then -- no gradient in a predecessor
    isGrad = false
    gradIndicator[preCoor] = 1 
  end

  return grad_hp, grad_cp, isGrad
end


function hidden:updateOutput(input)
  
  self:CheckSize(input)
  self:InitState()
  
  local H, N = self.nodeSize, self.batchSize
  local h, c = self.h, self.c
  local h0, c0 = self.h0, self.c0

  local h0_ = h:select(1 + self.inputDim, self.inputShape[self.inputDim])
  if h0:nElement() == 0 or not self.remember_states then -- first run or don't remember
    h0:resizeAs(h0_):zero()
  else -- if remember, use the previous evaluated h as h0
    h0:copy(h0_)
  end

  local c0_ = c:select(1 + self.inputDim, self.inputShape[self.inputDim])
  if c0:nElement() == 0 or not self.remember_states then -- first run or don't remember
    c0:resizeAs(c0_):zero()
  else -- if remember, use the previous evaluated c as c0
    c0:copy(c0_)
  end

  local bias_expand = self.bias:view(1, 5 * H):expand(N, 5 * H) -- copy the bias for a batch
  local w1 = self.weight[{{1, H}}] -- weights for h1
  local w2 = self.weight[{{H + 1, 2 * H}}] -- weights for h2
  
  -- initialize the coordinate of current node
  local coor = {}
  for i = 1, self.hiddenDim do
    coor[i] = 1
  end

  for nodeId = 1, self.nodeNum do
    local decompNodeId = (nodeId - 1) % self.decompNum + 1

    -- get the predecessor states
    local h1, c1 = self:GetPredecessorState(input, coor, self.hiddenDim)
    local h2, c2 = self:GetPredecessorState(input, coor, self.hiddenDim - 1 - decompNodeId)

    -- update the current node
    local hn = h[{{}, unpack(coor)}]
    local cn = c[{{}, unpack(coor)}]
    local curGates = self.gates[{{}, unpack(coor)}] -- N * 5H
    curGates:addmm(bias_expand, h1, w1) -- w1 * h1 + b
    curGates:addmm(h2, w2) -- w1 * h1 + b + w2 * h2
    curGates:narrow(2, 1, 4 * H):sigmoid() -- for gates
    curGates:narrow(2, 4 * H + 1, H):tanh() -- for new content
    local i  = curGates:narrow(2, 1, H)
    local f1 = curGates:narrow(2, H + 1, H)
    local f2 = curGates:narrow(2, 2 * H + 1, H)
    local o  = curGates:narrow(2, 3 * H + 1, H)
    local g  = curGates:narrow(2, 4 * H + 1, H)
    hn:cmul(i, g) -- gated new contents
    cn:cmul(f1, c1):addcmul(f2, c2):add(hn) -- new memories
    hn:tanh(cn):cmul(o) -- new hidden states
    self:MoveCoor(coor, 1)
  end

  return self.output
end


function hidden:backward(input, gradOutput, scale)

  self:CheckSize(input, gradOutput)
  self.gradOutput:copy(gradOutput)
  scale = scale or 1.0

  local H, N = self.nodeSize, self.batchSize
  local h, c = self.h, self.c
  local h0, c0 = self.h0, self.c0
  local grad_h = self.grad_h
  local grad_c = self.grad_c

  local w1 = self.weight[{{1, H}}]
  local w2 = self.weight[{{H + 1, 2 * H}}]
  local grad_w1 = self.gradWeight[{{1, H}}]
  local grad_w2 = self.gradWeight[{{H + 1, 2 * H}}]
  local grad_b = self.gradBias
  local grad_b_sum = self.grad_b_sum
  local gradIndicator = self.gradIndicator:clone()

  -- initialize the coordinate of current node
  local coor = {}
  for i = 1, self.hiddenDim do
    coor[i] = self.hiddenShape[i]
  end

  for nodeId = self.nodeNum, 1, -1 do
    local decompNodeId = (nodeId - 1) % self.decompNum + 1

    -- get the predecessor states
    local h1, c1 = self:GetPredecessorState(input, coor, self.hiddenDim)
    local h2, c2 = self:GetPredecessorState(input, coor, self.hiddenDim - 1 - decompNodeId)

    -- get the predecessor gradients
    local grad_h1, grad_c1, isGrad1 = self:GetPredecessorGrad(coor, self.hiddenDim, gradIndicator)
    local grad_h2, grad_c2, isGrad2 = self:GetPredecessorGrad(coor, self.hiddenDim - 1 - decompNodeId, gradIndicator)
    
    -- back propagate the gradients to predecessors
    local cn = c[{{}, unpack(coor)}] -- N * H
    local grad_hn = grad_h[{{}, unpack(coor)}] -- N * H
    local grad_cn = grad_c[{{}, unpack(coor)}] -- N * H
    local curGates = self.gates[{{}, unpack(coor)}] -- N * 5H
    local i  = curGates:narrow(2, 1, H)
    local f1 = curGates:narrow(2, H + 1, H)
    local f2 = curGates:narrow(2, 2 * H + 1, H)
    local o  = curGates:narrow(2, 3 * H + 1, H)
    local g  = curGates:narrow(2, 4 * H + 1, H)

    local grad_a = self.grad_a:zero() -- gradients of activations
    local grad_ai  = grad_a:narrow(2, 1, H)
    local grad_af1 = grad_a:narrow(2, H + 1, H)
    local grad_af2 = grad_a:narrow(2, 2 * H + 1, H)
    local grad_ao  = grad_a:narrow(2, 3 * H + 1, H)
    local grad_ag  = grad_a:narrow(2, 4 * H + 1, H)

    -- We will use grad_ai, grad_af, and grad_ao as temporary buffers to compute grad_cn. 
    -- We will need tanh_next_c (stored in grad_ai) to compute grad_ao; 
    -- the other values can be overwritten after we compute grad_cn
    local tanh_next_c = grad_ai:tanh(next_c) -- grad_ai is used as a buffer
    local tanh_next_c2 = grad_af1:cmul(tanh_next_c, tanh_next_c) -- grad_af1 is used as a buffer
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
    grad_af1:fill(1):add(-1, f1):cmul(f1):cmul(c1):cmul(grad_cn)
    grad_af2:fill(1):add(-1, f2):cmul(f2):cmul(c2):cmul(grad_cn)

    -- temporally accumulate the gradients of parameters, as they are shared
    grad_w1:addmm(scale, h1:t(), grad_a)
    grad_w2:addmm(scale, h2:t(), grad_a)

    grad_b_sum:sum(grad_a, 1) -- directly accumulate grad_b (equal to grad_a) inside a batch
    grad_b:add(scale, grad_b_sum)

    if not isGrad1 then -- if no previous gradient, we overwrite it
      grad_h1:mm(grad_a, w1:t())
      grad_c1:mm(grad_cn, f1)
    else -- if previous gradient exists, we accumulate it
      grad_h1:addmm(grad_a, w1:t())
      grad_c1:addmm(grad_cn, f1)
    end

    if not isGrad2 then -- if no previous gradient, we overwrite it 
      grad_h2:mm(grad_a, w1:t())
      grad_c2:mm(grad_cn, f2)
    else -- if previous gradient exists, we accumulate it
      grad_h2:addmm(grad_a, w1:t())
      grad_c2:addmm(grad_cn, f2)
    end
    
    self:MoveCoor(coor, -1)
  end

  return self.gradInput
end


function hidden:updateGradInput(input, gradOutput)
  return self:backward(input, gradOutput, 0)
end


function hidden:accGradParameters(input, gradOutput, scale)
  self:backward(input, gradOutput, scale)
end
