require 'torch'
require 'nn'


local hidden, parent = torch.class('nn.TensHidden', 'nn.Module')


function hidden:__init(inputShape, tensShape, nodeSize, dropout)

  assert(#inputShape > 0, 'invalid input shape')
  assert(#tensShape > 0, 'invalid tensorizing shape')
  assert(nodeSize > 0, 'invalid node size')
  if dropout then
    assert(dropout > 0 and dropout < 1, 'invalid dropout ratio')
  end

  parent.__init(self)

  self.inputShape = nil
  self.tensShape = tensShape -- table
  self.nodeSize = nodeSize
  self.batchSize = nil
  self.dropout = dropout

  local H = nodeSize
  self.hiddenDim = #inputShape + #tensShape
  self.gateNum = 3 + self.hiddenDim -- new content, output gate, input gate, forget gates
  self.weight = torch.Tensor(self.hiddenDim * H, self.gateNum * H) 
  self.gradWeight = torch.Tensor(self.hiddenDim * H, self.gateNum * H):zero()
  self.bias = torch.Tensor(self.gateNum * H)
  self.gradBias = torch.Tensor(self.gateNum * H):zero()
  self:reset()

  self.buff1 = torch.Tensor() -- This will be (N, H), a small buffer
  self.buff2 = torch.Tensor() -- This will be (N, H), a small buffer
  self.stateAndGradBuff = torch.Tensor()

  self.gates = torch.Tensor() -- This will be (N, unpack(self.hiddenShape), gateNum * H)
  self.grad_b_sum = torch.Tensor() -- This will be (1, gateNum * H)
  self.grad_a = torch.Tensor() -- This will be (N, gateNum * H)
  self.gradIndicator = torch.ByteTensor() -- for gradients accumulation

  self.noise = torch.Tensor() -- This will be (N, unpack(self.hiddenShape), tensDim * H), for drop out
  self.h_dp = torch.Tensor() -- This will be (N, unpack(self.hiddenShape), tensDim * H), for drop out
  self.freezeNoise = false -- in gradient checking, we freeze the drop out noise to fix the network

  self.h0 = torch.Tensor()
  self.c0 = torch.Tensor()
  self.remember_states = false
  self.train = self.train or true
end


function hidden:reset(std)
  local H = self.nodeSize
  if not std then
    std = 1.0 / math.sqrt(self.hiddenDim * H)
  end
  self.weight:normal(0, std)
  self.bias:zero()
  self.bias[{{3 * H + 1, self.gateNum * H}}]:fill(1) -- set the bias of forget gates to 1

  return self
end


function hidden:resetStates()
  -- clear initial states h0 and c0
  self.h0 = self.h0.new()
  self.c0 = self.c0.new()
end


function hidden:InitState(input)

  local isInputShapeChanged = false
  if self.inputShape then
    if self.inputDim + 2 ~= input:dim() then
      isInputShapeChanged = true
    else
      for i, v in ipairs(self.inputShape) do
        if input:size(i + 1) ~= v then
          isInputShapeChanged = true
          break
        end
      end
    end
  end
  if self.batchSize then
    if input:size(1) ~= self.batchSize then
      isInputShapeChanged = true
    end
  end

  if not self.inputShape or isInputShapeChanged then
    self.batchSize = input:size(1)

    self.inputShape = {}
    for i = 2, input:dim() - 1 do
      table.insert(self.inputShape, input:size(i))
    end

    self.inputDim = #self.inputShape
    self.tensDim = #self.tensShape

    self.hiddenShape = {} -- table
    for _, v in ipairs(self.inputShape) do
      table.insert(self.hiddenShape, v)
    end
    for _, v in ipairs(self.tensShape) do
      table.insert(self.hiddenShape, v)
    end
    assert(self.hiddenDim == #self.hiddenShape, 'the dimension of of input mismatched with inputShape')
    self.totalDim = self.hiddenDim + 2 -- one for batch and one for node vector

    self.nodeNum = 1
    for _, v in ipairs(self.hiddenShape) do
        self.nodeNum = self.nodeNum * v
    end
  end

  local H, N = self.nodeSize, self.batchSize

  if self.stateAndGradBuff:nElement() == 0 or isInputShapeChanged then
    local sz = {N}
    for _, v in ipairs(self.hiddenShape) do
      table.insert(sz, v + 1)
    end
    table.insert(sz, 2 * H)
    self.buff1:resize(N, H):zero()
    self.buff2:resize(N, H):zero()
    self.stateAndGradBuff:resize(torch.LongStorage(sz)):zero()
    local stateRegion, gradRegion = {{}}, {{}}
    local outputRegion, gradInputRegion, gradOutputRegion = {{}}, {{}}, {{}}
    local gradInitStateRegion = {{}}
    for i, v in ipairs(self.hiddenShape) do
      table.insert(stateRegion, {1, v})
      table.insert(gradRegion, {1, v + 1})
      if i < self.inputDim then -- the input dimension (except the last one)
        table.insert(outputRegion, {1, v})
        table.insert(gradInputRegion, {2, v + 1})
        table.insert(gradOutputRegion, {2, v + 1})
        table.insert(gradInitStateRegion, {2, v + 1})
      elseif i == self.inputDim then -- the last input dimension
        table.insert(outputRegion, {1, v})
        table.insert(gradInputRegion, {2, v + 1})
        table.insert(gradOutputRegion, {2, v + 1})
        table.insert(gradInitStateRegion, 1)
      elseif i < self.hiddenDim then -- the tensorized dimension (except the last one)
        table.insert(outputRegion, v)
        table.insert(gradInputRegion, 2)
        table.insert(gradOutputRegion, v + 1)
        table.insert(gradInitStateRegion, {2, v + 1})
      else -- the last tensorized dimension
        table.insert(outputRegion, v)
        table.insert(gradInputRegion, 1)
        table.insert(gradOutputRegion, v + 1)
        table.insert(gradInitStateRegion, {2, v + 1})
      end
    end

    local states = self.stateAndGradBuff[stateRegion]
    local grads = self.stateAndGradBuff[gradRegion]
    self._output = self.stateAndGradBuff[outputRegion]
    self._grad_x = self.stateAndGradBuff[gradInputRegion]
    self.gradOutput = self.stateAndGradBuff[gradOutputRegion]
    local grad_h0c0 = self.stateAndGradBuff[gradInitStateRegion]

    local S = self.totalDim
    self.h = states:narrow(S, 1, H)
    self.c = states:narrow(S, H + 1, H)
    self.grad_h = grads:narrow(S, 1, H)
    self.grad_c = grads:narrow(S, H + 1, H)
    self._grad_h0 = grad_h0c0:narrow(grad_h0c0:dim(), 1, H)
    self._grad_c0 = grad_h0c0:narrow(grad_h0c0:dim(), H + 1, H)
  end

  if self.dropout then
    if self.noise:nElement() == 0 or isInputShapeChanged then
      local sz = {N}
      for _, v in ipairs(self.hiddenShape) do
        table.insert(sz, v)
      end
      table.insert(sz, self.tensDim * H)
      self.noise:resize(torch.LongStorage(sz)):zero()
      self.h_dp:resize(torch.LongStorage(sz)):zero()
    end
  end

  if self.gates:nElement() == 0 or isInputShapeChanged then
    local sz = {N}
    for _, v in ipairs(self.hiddenShape) do
      table.insert(sz, v)
    end
    table.insert(sz, self.gateNum * H)
    self.gates:resize(torch.LongStorage(sz)):zero()
  end

  if self.grad_b_sum:nElement() == 0 then
    self.grad_b_sum:resize(1, self.gateNum * H):zero()
  end

  if self.grad_a:nElement() == 0 then
    self.grad_a:resize(N, self.gateNum * H):zero()
  end

  if self.gradIndicator:nElement() == 0 or isInputShapeChanged then
    local sz = {}
    for _, v in ipairs(self.hiddenShape) do
      table.insert(sz, v + 1)
    end
    self.gradIndicator:resize(torch.LongStorage(sz)):zero()

    local gradOutputRegion = {}
    for i, v in ipairs(self.hiddenShape) do
      if i <= self.inputDim then
        table.insert(gradOutputRegion, {2, v + 1})
      else
        table.insert(gradOutputRegion, v + 1)
      end
    end
    self.gradIndicator[gradOutputRegion]:fill(1)
  end
end


function hidden:clearState()
  -- clear intermediate variables (the original clearState() in 'nn.Module' is overloaded, 
  -- as it only clears 'output' and 'gradInput')

  self.buff1:set()
  self.buff2:set()
  self.stateAndGradBuff:set()
  self.h:set()
  self.c:set()
  self.output:set()
  self._output:set()
  self.gates:set()

  self.grad_x:set()
  self._grad_x:set()
  self.grad_h:set()
  self.grad_c:set()
  self.grad_h0:set()
  self.grad_c0:set()
  self._grad_h0:set()
  self._grad_c0:set()
  self.gradOutput:set()
  self.grad_b_sum:set()
  self.grad_a:set()
  self.gradIndicator:set()

  self.noise:set()
  self.h_dp:set()
end


function hidden:CheckSize(input, gradOutput)

  assert(torch.isTensor(input))
  assert(input:dim() >= 3) -- batch, input dim, node vector
  assert(input:size(input:dim()) == self.nodeSize * 2) -- hidden vector and cell vector

  if gradOutput then
    assert(gradOutput:dim() == input:dim())
    for i = 1, input:dim() do
      assert(gradOutput:size(i) == input:size(i))
    end
  end
end


function hidden:UnpackInput(input)
  local x, h0, c0 = nil, nil, nil
  if torch.type(input) == 'table' and #input == 3 then
    x, h0, c0 = unpack(input)
  elseif torch.type(input) == 'table' and #input == 2 then
    x, h0 = unpack(input)
  elseif torch.isTensor(input) then
    x = input
  else
    assert(false, 'invalid input')
  end
  return x, h0, c0
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


function hidden:GetPredecessorState(input, curCoor)

  local H, N = self.nodeSize, self.batchSize
  local hps = {}
  local cps = {}
  local noiseps = {}
  local h_dpps = {}

  for predecessorDim = 1, self.hiddenDim do
    local hp = torch.Tensor():type(self.weight:type())
    local cp = torch.Tensor():type(self.weight:type())

    local preCoor = {}
    for i, v in ipairs(curCoor) do
      preCoor[i] = v
    end

    if preCoor[predecessorDim] > 1 then
      -- point to the previous node
      preCoor[predecessorDim] = preCoor[predecessorDim] - 1
      hp = self.h[{{}, unpack(preCoor)}] -- N * H
      cp = self.c[{{}, unpack(preCoor)}] -- N * H
    else -- the case that requires initial states (out of the network's shape)
      hp:resize(N, H):zero()
      cp:resize(N, H):zero()
      if predecessorDim == self.inputDim then -- get value from the last states of previous batch
        -- as h0 (c0) are the last slice on both input dimension h (c), we remove the corresponding coordinate
        table.remove(preCoor, self.inputDim) -- remove the input dimension
        hp = self.h0[{{}, unpack(preCoor)}] -- N * H
        cp = self.c0[{{}, unpack(preCoor)}] -- N * H
      elseif predecessorDim == self.hiddenDim then -- get value from input
        local isFromInput = true
        for i = self.inputDim + 1, self.hiddenDim - 1 do
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
          local hpcp = input[{{}, unpack(inputCoor)}] -- N * 2H
          hp = hpcp[{{}, {1, H}}] -- N * H
          cp = hpcp[{{}, {H + 1, 2 * H}}] -- N * H
        end
      end
    end

    if self.dropout and predecessorDim > self.inputDim then
      local i = predecessorDim - self.inputDim - 1
      local noisep = self.noise[{{}, unpack(curCoor)}]:narrow(2, i * H + 1, H)
      local h_dpp = self.h_dp[{{}, unpack(curCoor)}]:narrow(2, i * H + 1, H)
      table.insert(noiseps, noisep)
      table.insert(h_dpps, h_dpp)
    end

    table.insert(hps, hp)
    table.insert(cps, cp)
  end

  return hps, cps, noiseps, h_dpps
end


function hidden:GetPredecessorGrad(curCoor, gradIndicator)
  local grad_hps = {}
  local grad_cps = {}
  local isGrads = {}

  for predecessorDim = 1, self.hiddenDim do
    local preCoor = {}
    for i, v in ipairs(curCoor) do
      preCoor[i] = v
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

    table.insert(grad_hps, grad_hp)
    table.insert(grad_cps, grad_cp)
    table.insert(isGrads, isGrad)
  end

  return grad_hps, grad_cps, isGrads
end


function hidden:DropoutForward(input, noise)
  local p = 1 - self.dropout
  if self.train then
    if not self.freezeNoise then
      noise:bernoulli(p):div(p)
    end
    input:cmul(noise)
  end
end


function hidden:DropoutBackward(noise, gradOutput)
  if self.train then
    gradOutput:cmul(noise)
  end
end


function hidden:SoftmaxForward(input)
  local H, N = self.nodeSize, self.batchSize
  assert(input:size(1) == N)
  assert(input:size(2) == (self.hiddenDim + 1) * H)

  local a_max = self.buff1:copy(input:narrow(2, 1, H))
  for i = 1, self.hiddenDim do
    local a = input:narrow(2, i * H + 1, H)
    a_max:cmax(a)
  end

  local exp_sum = self.buff2:zero()
  for i = 0, self.hiddenDim do
    local a = input:narrow(2, i * H + 1, H)
    a:add(-1, a_max):exp()
    exp_sum:add(a)
  end

  for i = 0, self.hiddenDim do
    local a = input:narrow(2, i * H + 1, H)
    a:cdiv(exp_sum)
  end
end


function hidden:SoftmaxBackward(output, gradOutput)
  local H, N = self.nodeSize, self.batchSize
  assert(output:size(1) == N)
  assert(output:size(2) == (self.hiddenDim + 1) * H)
  assert(gradOutput:size(1) == output:size(1))
  assert(gradOutput:size(2) == output:size(2))

  local product_sum = self.buff1:zero()
  for i = 0, self.hiddenDim do
    local output_every = output:narrow(2, i * H + 1, H)
    local gradOutput_every = gradOutput:narrow(2, i * H + 1, H)
    gradOutput_every:cmul(output_every)
    product_sum:add(gradOutput_every)
  end

  for i = 0, self.hiddenDim do
    local output_every = output:narrow(2, i * H + 1, H)
    local gradOutput_every = gradOutput:narrow(2, i * H + 1, H)
    gradOutput_every:addcmul(-1, output_every, product_sum)
  end
end


function hidden:updateOutput(input)
  self.recompute_backward = true
  local x, h0, c0 = self:UnpackInput(input)
  self:CheckSize(x)
  self.isReturnGradH0 = (h0 ~= nil)
  self.isReturnGradC0 = (c0 ~= nil)
  self:InitState(x)
  local h, c = self.h, self.c
  local H, N = self.nodeSize, self.batchSize
  local buff1 = self.buff1

  if h0 then
    self.h0 = h0:clone()
  else
    h0 = self.h0
    local h0_ = h:select(1 + self.inputDim, self.inputShape[self.inputDim]) -- the last slice on the last input dimension
    if h0:nElement() == 0 or not self.remember_states then -- first run or don't remember
      h0:resizeAs(h0_):zero()
    else -- if remember, use the previous evaluated h as h0
      assert(x:size(1) == self.batchSize, 'batch sizes must be the same to remember states')
      h0:copy(h0_)
    end
  end

  if c0 then
    self.c0 = c0:clone()
  else
    c0 = self.c0
    local c0_ = c:select(1 + self.inputDim, self.inputShape[self.inputDim]) -- the last slice on the last input dimension
    if c0:nElement() == 0 or not self.remember_states then -- first run or don't remember
      c0:resizeAs(c0_):zero()
    else -- if remember, use the previous evaluated c as c0
      assert(x:size(1) == self.batchSize, 'batch sizes must be the same to remember states')
      c0:copy(c0_)
    end
  end
  
  local gNum = self.gateNum
  local bias_expand = self.bias:view(1, gNum * H):expand(N, gNum * H) -- copy the bias for a batch

  -- initialize the coordinate of current node
  local coor = {}
  for i = 1, self.hiddenDim do
    coor[i] = 1
  end

  for nodeId = 1, self.nodeNum do

    -- get the predecessor states
    local hs, cs, noises, h_dps = self:GetPredecessorState(x, coor)

    -- update the current node
    local hn = h[{{}, unpack(coor)}]
    local cn = c[{{}, unpack(coor)}]
    local curGates = self.gates[{{}, unpack(coor)}] -- N * (gNum * H)
    curGates:copy(bias_expand) -- b
    for i = 1, self.hiddenDim do
      local wi = self.weight[{{i * H - H + 1, i * H}}] -- weights for hi
      local hi = hs[i]
      if self.dropout and i > self.inputDim then
        local idx = i - self.inputDim
        hi = h_dps[idx]:copy(hi)
        self:DropoutForward(hi, noises[idx])
      end
      curGates:addmm(hi, wi) -- sum(wi * hi) + b
    end
    local g = curGates:narrow(2, 1, H):tanh() -- new content
    local o = curGates:narrow(2, H + 1, H):sigmoid() -- output gate
    local i_f = curGates:narrow(2, 2 * H + 1, (gNum - 2) * H) -- input gate and forget gates
    self:SoftmaxForward(i_f)
    local i = i_f:narrow(2, 1, H) -- input gate
    hn:cmul(i, g) -- gated new content
    cn:copy(hn)
    for i = 1, self.hiddenDim do
      local f = i_f:narrow(2, i * H + 1, H) -- forget gate
      cn:addcmul(f, cs[i]) -- accumulate to new memory
    end
    hn:tanh(cn):cmul(o) -- output

    self:MoveCoor(coor, 1)
  end

  self.output = self._output:contiguous()
  return self.output
end


function hidden:backward(input, gradOutput, scale)

  self.recompute_backward = false
  local x, h0, c0 = self:UnpackInput(input)
  self:CheckSize(x, gradOutput)
  self.gradOutput:copy(gradOutput)
  scale = scale or 1.0
  assert(scale == 1.0, 'must have scale=1')

  local h, c = self.h, self.c
  local grad_h, grad_c = self.grad_h, self.grad_c
  if not c0 then c0 = self.c0 end
  if not h0 then h0 = self.h0 end

  local H, N = self.nodeSize, self.batchSize
  local grad_b = self.gradBias
  local grad_b_sum = self.grad_b_sum
  local gradIndicator = self.gradIndicator:clone()
  local buff1 = self.buff1

  -- initialize the coordinate of current node
  local coor = {}
  for i = 1, self.hiddenDim do
    coor[i] = self.hiddenShape[i]
  end

  for nodeId = self.nodeNum, 1, -1 do
    -- get the predecessor states
    local hs, cs, noises, h_dps = self:GetPredecessorState(x, coor)
    -- get the predecessor gradients
    local grad_hs, grad_cs, isGrads = self:GetPredecessorGrad(coor, gradIndicator)
    
    -- back propagate the gradients to predecessors
    local hn = h[{{}, unpack(coor)}] -- N * H
    local cn = c[{{}, unpack(coor)}] -- N * H
    local coorg = {}
    for i, v in ipairs(coor) do
      coorg[i] = v + 1
    end
    local grad_hn = grad_h[{{}, unpack(coorg)}] -- N * H
    local grad_cn = grad_c[{{}, unpack(coorg)}] -- N * H
    local curGates = self.gates[{{}, unpack(coor)}] -- N * (gNum * H)
    local g = curGates:narrow(2, 1, H)
    local o = curGates:narrow(2, H + 1, H)
    local i = curGates:narrow(2, 2 * H + 1, H)
    local i_f = curGates:narrow(2, 2 * H + 1, (self.gateNum - 2) * H) -- input gate and forget gates

    local grad_a = self.grad_a:zero() -- gradients of activations
    local grad_ag  = grad_a:narrow(2, 1, H)
    local grad_ao = grad_a:narrow(2, H + 1, H)
    local grad_aif = grad_a:narrow(2, 2 * H + 1, (self.gateNum - 2) * H)

    local tanh_next_c = buff1:tanh(cn)
    local tanh_next_c2 = grad_ao:cmul(tanh_next_c, tanh_next_c)
    local my_grad_cn = grad_ag
    my_grad_cn:fill(1):add(-1, tanh_next_c2):cmul(o):cmul(grad_hn)
    grad_cn:add(my_grad_cn) -- accumulate the gradient of cell from current hidden state

    -- gradients of ao and ag
    grad_ao:fill(1):add(-1, o):cmul(o):cmul(tanh_next_c):cmul(grad_hn)
    local g2 = buff1:cmul(g, g)
    grad_ag:fill(1):add(-1, g2):cmul(i):cmul(grad_cn)

    -- gradients of ai and all af
    for i = 0, self.hiddenDim do
      local grad_aif_every = grad_aif:narrow(2, i * H + 1, H) -- the gradient at input gate (i == 0) or forget gate
      local content = g -- new content (i == 0) or the i-th memory cell
      if (i > 0) then 
        content = cs[i]
      end
      grad_aif_every:cmul(content, grad_cn)
    end
    self:SoftmaxBackward(i_f, grad_aif)

    -- gradients of parameters
    for i = 1, self.hiddenDim do
      local grad_wi = self.gradWeight[{{i * H - H + 1, i * H}}]
      local hi = hs[i]
      if self.dropout and i > self.inputDim then
        hi = h_dps[i - self.inputDim]
      end
      grad_wi:addmm(scale, hi:t(), grad_a)
    end
    grad_b_sum:sum(grad_a, 1) -- directly accumulate grad_b (equal to grad_a) inside a batch
    grad_b:add(scale, grad_b_sum)

    -- gradients of hidden states and memory cells
    for i = 1, self.hiddenDim do
      local wi = self.weight[{{i * H - H + 1, i * H}}] -- weights for hi
      local f = curGates:narrow(2, (i + 2) * H + 1, H) -- forget gate

      local grad_hi = buff1:mm(grad_a, wi:t())
      if self.dropout and i > self.inputDim then
        self:DropoutBackward(noises[i - self.inputDim], grad_hi)
      end

      if not isGrads[i] then -- if no previous gradient, we overwrite it
        grad_hs[i]:copy(grad_hi)
        grad_cs[i]:cmul(grad_cn, f)
      else -- if previous gradient exists, we accumulate it
        grad_hs[i]:add(grad_hi)
        grad_cs[i]:addcmul(grad_cn, f)
      end
    end
    
    self:MoveCoor(coor, -1)
  end

  self.grad_x = self._grad_x:contiguous()
  self.grad_h0 = self._grad_h0:contiguous()
  self.grad_c0 = self._grad_c0:contiguous()
  if self.isReturnGradH0 and self.isReturnGradC0 then
    self.gradInput = {self.grad_x, self.grad_h0, self.grad_c0}
  elseif self.isReturnGradH0 then
    self.gradInput = {self.grad_x, self.grad_h0}
  else
    self.gradInput = self.grad_x
  end

  return self.gradInput
end


function hidden:updateGradInput(input, gradOutput)
  if self.recompute_backward then
    self:backward(input, gradOutput, 1.0)
  end
  return self.gradInput
end


function hidden:accGradParameters(input, gradOutput, scale)
  if self.recompute_backward then
    self:backward(input, gradOutput, scale)
  end
end
