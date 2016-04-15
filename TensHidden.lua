require 'torch'
require 'nn'


local hidden, parent = torch.class('nn.TensHidden', 'nn.Module')


function hidden:__init(tensShape, nodeSize, isBatchNorm)

  assert(#tensShape > 0, 'invalid tensorizing shape')
  assert(nodeSize > 0, 'invalid node size')

  parent.__init(self)
  
  self.inputShape = nil
  self.tensShape = tensShape -- table
  self.nodeSize = nodeSize
  self.batchSize = nil
  self.isBatchNorm = (isBatchNorm or 0) == 1

  local H = self.nodeSize
  self.weight = torch.Tensor(2 * H, 3 * H) -- forget gate, select gate, new content
  self.gradWeight = torch.Tensor(2 * H, 3 * H):zero()

  self.bias = torch.Tensor(9 * H)
  self.gamma1 = self.bias[{{1, 3 * H}}]
  self.gamma2 = self.bias[{{3 * H + 1, 6 * H}}]
  self.beta = self.bias[{{6 * H + 1, 9 * H}}]

  self.gradBias = torch.Tensor(9 * H):zero()
  self.gradGamma1 = self.gradBias[{{1, 3 * H}}]
  self.gradGamma2 = self.gradBias[{{3 * H + 1, 6 * H}}]
  self.gradBeta = self.gradBias[{{6 * H + 1, 9 * H}}]
  self.gamma_init = 1
  self.ep = 0.00001

  self:reset()

  self.buff3H = torch.Tensor() -- This will be (1, 3 * H), a small buffer
  self.buffNH = torch.Tensor() -- This will be (N, H), a small buffer
  self.buffN3H = torch.Tensor() -- This will be (N, 3 * H), a small buffer
  self.stateAndGradBuff = torch.Tensor()

  self.gates = torch.Tensor() -- This will be (N, unpack(self.hiddenShape), 3H)
  self.grad_a = torch.Tensor() -- This will be (N, 3H)
  self.gradIndicator = torch.ByteTensor() -- for gradients accumulation

  self.bn_in1 = torch.Tensor() -- This will be (N, unpack(self.hiddenShape), 3H), the input of batch normalization
  self.bn_in2 = torch.Tensor() -- This will be (N, unpack(self.hiddenShape), 3H), the input of batch normalization
  self.bn_out1 = torch.Tensor() -- This will be (N, unpack(self.hiddenShape), 3H), the output of batch normalization
  self.bn_out2 = torch.Tensor() -- This will be (N, unpack(self.hiddenShape), 3H), the output of batch normalization
  self.means1 = torch.Tensor() -- This will be (unpack(self.hiddenShape), 3 * H)
  self.means2 = torch.Tensor() -- This will be (unpack(self.hiddenShape), 3 * H)
  self.vars1 = torch.Tensor() -- This will be (unpack(self.hiddenShape), 3 * H)
  self.vars2 = torch.Tensor() -- This will be (unpack(self.hiddenShape), 3 * H)

  self.h0 = torch.Tensor()
  self.remember_states = false
  
end


function hidden:reset(std)
  -- initialize weights and bias

  local H = self.nodeSize
  if not std then
    std = 1.0 / math.sqrt(H * 2)
  end
  self.weight:normal(0, std) --:add(torch.eye(H):repeatTensor(2, 3))

  self.bias:fill(self.gamma_init)
  self.beta:zero()
  -- self.bias[{{1, self.nodeSize}}]:fill(1) -- set the bias of forget gates to 1
  return self
end


function hidden:resetStates()
  -- clear initial states h0
  self.h0 = self.h0.new()
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
  end

  local H, N = self.nodeSize, self.batchSize

  if self.stateAndGradBuff:nElement() == 0 or isInputShapeChanged then
    local sz = {N}
    for _, v in ipairs(self.hiddenShape) do
      table.insert(sz, v + 1)
    end
    table.insert(sz, H)
    self.buffNH:resize(N, H):zero()
    self.buffN3H:resize(N, 3 * H):zero()
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
      elseif i < self.hiddenDim - 1 then -- the tensorized dimension (except the last one)
        table.insert(outputRegion, v)
        table.insert(gradInputRegion, 2)
        table.insert(gradOutputRegion, v + 1)
        table.insert(gradInitStateRegion, {2, v + 1})
      elseif i == self.hiddenDim - 1 then -- the last tensorized dimension
        table.insert(outputRegion, v)
        table.insert(gradInputRegion, 1)
        table.insert(gradOutputRegion, v + 1)
        table.insert(gradInitStateRegion, {2, v + 1})
      else -- the decomposing dimension
        table.insert(outputRegion, v)
        table.insert(gradInputRegion, v + 1)
        table.insert(gradOutputRegion, v + 1)
        table.insert(gradInitStateRegion, v + 1)
      end
    end

    self.h = self.stateAndGradBuff[stateRegion]
    self.grad_h = self.stateAndGradBuff[gradRegion]
    self._output = self.stateAndGradBuff[outputRegion]
    self._grad_x = self.stateAndGradBuff[gradInputRegion]
    self.gradOutput = self.stateAndGradBuff[gradOutputRegion]
    self._grad_h0 = self.stateAndGradBuff[gradInitStateRegion]
  end

  if self.isBatchNorm then
    if self.bn_in1:nElement() == 0 or isInputShapeChanged then
      local sz = {1}
      for _, v in ipairs(self.hiddenShape) do
        table.insert(sz, v)
      end
      table.insert(sz, 3 * H)
      self.means1:resize(torch.LongStorage(sz)):zero()
      self.means2:resize(torch.LongStorage(sz)):zero()
      self.vars1:resize(torch.LongStorage(sz)):zero()
      self.vars2:resize(torch.LongStorage(sz)):zero()
      sz[1] = N
      self.bn_in1:resize(torch.LongStorage(sz)):zero()
      self.bn_in2:resize(torch.LongStorage(sz)):zero()
      self.bn_out1:resize(torch.LongStorage(sz)):zero()
      self.bn_out2:resize(torch.LongStorage(sz)):zero()
    end
  end

  if self.gates:nElement() == 0 or isInputShapeChanged then
    local sz = {N}
    for _, v in ipairs(self.hiddenShape) do
      table.insert(sz, v)
    end
    table.insert(sz, 3 * H)
    self.gates:resize(torch.LongStorage(sz)):zero() -- This will be (N, unpack(self.hiddenShape), 3H)
  end

  if self.buff3H:nElement() == 0 then
    self.buff3H:resize(1, 3 * H):zero() -- This will be (1, 3H)
  end

  if self.grad_a:nElement() == 0 then
    self.grad_a:resize(N, 3 * H):zero() -- This will be (N, 3H)
  end

  if self.gradIndicator:nElement() == 0 or isInputShapeChanged then
    sz = {}
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

  self.buff3H:set()
  self.buffNH:set()
  self.buffN3H:set()
  self.stateAndGradBuff:set()
  self.gates:set()
  self.grad_a:set()
  self.gradIndicator:set()

  self.output:set()
  self.grad_x:set()
  self.grad_h0:set()

  self.bn_in1:set()
  self.bn_in2:set()
  self.bn_out1:set()
  self.bn_out2:set()
  self.means1:set()
  self.means2:set()
  self.vars1:set()
  self.vars2:set()
end


function hidden:CheckSize(input, gradOutput)

  assert(torch.isTensor(input))
  assert(input:dim() >= 3) -- batch, input dim, node vector
  assert(input:size(input:dim()) == self.nodeSize) -- hidden vector

  if gradOutput then
    assert(gradOutput:dim() == input:dim())
    for i = 1, input:dim() do
      assert(gradOutput:size(i) == input:size(i))
    end
  end
end


function hidden:UnpackInput(input)
  local x, h0 = nil, nil
  if torch.type(input) == 'table' and #input == 2 then
    x, h0 = unpack(input)
  elseif torch.isTensor(input) then
    x = input
  else
    assert(false, 'invalid input')
  end
  return x, h0
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


function hidden:GetPredecessorState(input, curCoor, predecessorDim, id)

  local x = input
  local H, N = self.nodeSize, self.batchSize
  local h = self.h
  local h0 = self.h0
  local hp = torch.Tensor():type(self.weight:type())

  local preCoor = {}
  for i, v in ipairs(curCoor) do
    preCoor[i] = v
  end

  if predecessorDim == self.hiddenDim and preCoor[predecessorDim] == 1 then
    predecessorDim = self.hiddenDim - 1
  end
  if predecessorDim < self.hiddenDim then -- if not along the decomposing dimension
    preCoor[self.hiddenDim] = self.decompNum -- point to the last decompesed one of the other dimension
  end

  if preCoor[predecessorDim] > 1 then
    -- point to the previous node
    preCoor[predecessorDim] = preCoor[predecessorDim] - 1
    hp = h[{{}, unpack(preCoor)}] -- N * H
  else -- the case that requires initial states (out of the network's shape)
    hp:resize(N, H):zero()
    if predecessorDim == self.inputDim then -- get value from the last states of previous batch
      -- as h0 are the last slice on both input dimension and decomposed dimension of h (c), we remove the corresponding coordinates
      table.remove(preCoor, self.inputDim) -- remove the input dimension
      table.remove(preCoor, #preCoor) -- remove the decomposing dimension
      hp = h0[{{}, unpack(preCoor)}] -- N * H
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
        hp = x[{{}, unpack(inputCoor)}] -- N * H
      end
    end
  end

  local meanp, varp, bn_inp, bn_outp = nil, nil, nil, nil
  if self.isBatchNorm then
    local means, vars, bn_in, bn_out = self.means1, self.vars1, self.bn_in1, self.bn_out1
    if id == 2 then
      means, vars, bn_in, bn_out = self.means2, self.vars2, self.bn_in2, self.bn_out2
    end
    meanp = means[{{}, unpack(curCoor)}] -- 3 * H
    varp = vars[{{}, unpack(curCoor)}] -- 3 * H
    bn_inp = bn_in[{{}, unpack(curCoor)}] -- N * 3H
    bn_outp = bn_out[{{}, unpack(curCoor)}] -- N * 3H
  end
  
  return hp, meanp, varp, bn_inp, bn_outp
end


function hidden:GetPredecessorGrad(curCoor, predecessorDim, gradIndicator)

  local preCoor = {}
  for i, v in ipairs(curCoor) do
    preCoor[i] = v
  end

  if predecessorDim == self.hiddenDim and preCoor[predecessorDim] == 1 then
    predecessorDim = self.hiddenDim - 1
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
  
  -- if there's no gradient in a predecessor, we overwrite it with a back propagated gradient,
  -- otherwise we accumulate the back propagated gradient
  local isGrad = true
  if gradIndicator[preCoor] == 0 then -- no gradient in a predecessor
    isGrad = false
    gradIndicator[preCoor] = 1 
  end

  return grad_hp, isGrad
end


function hidden:batchNormForward(input, mean, var, output)
  local N, W = input:size(1), input:size(2)
  local buffN3H = self.buffN3H

  mean:mean(input, 1)
  var:var(input, 1, true)
  output:copy(mean:expand(N, W)):mul(-1):add(input)
  buffN3H:copy(var:expand(N, W)):add(self.ep):sqrt()
  output:cdiv(buffN3H)
end


function hidden:batchNormBackward(input, mean, var, output, gradOutput)
  local N, W = input:size(1), input:size(2)
  local buffN3H = self.buffN3H

  local input_minus_mean = mean:expand(N, W):clone():mul(-1):add(input)
  local sqrt_var = var:clone():add(self.ep):sqrt()
  local grad_var = sqrt_var:clone():pow(-3):div(-2)
  grad_var = buffN3H:cmul(gradOutput, input_minus_mean):sum(1):cmul(grad_var)

  local grad_mean = buffN3H:copy(input_minus_mean):mul(-2):div(N):sum(1):cmul(grad_var)
  grad_mean = gradOutput:sum(1):mul(-1):cdiv(sqrt_var):add(grad_mean)

  buffN3H:mul(-1):cmul(grad_var:expand(N, W)):addcdiv(gradOutput, sqrt_var:expand(N, W)):add(1/N, grad_mean:expand(N, W))
  gradOutput:copy(buffN3H)
end


function hidden:updateOutput(input)
  self.recompute_backward = true
  local x, h0 = self:UnpackInput(input)
  self:CheckSize(x)
  self.isReturnGradH0 = (h0 ~= nil)
  self:InitState(x)
  local h = self.h
  local H, N = self.nodeSize, self.batchSize

  if h0 then
    self.h0 = h0:clone()
  else
    h0 = self.h0
    local h0_ = h:select(1 + self.inputDim, self.inputShape[self.inputDim]) -- the last slice on the last input dimension
    h0_ = h0_:select(h0_:dim() - 1, self.decompNum) -- the last slice on the decomposing dimension
    if h0:nElement() == 0 or not self.remember_states then -- first run or don't remember
      h0:resizeAs(h0_):zero()
    else -- if remember, use the previous evaluated h as h0
      assert(x:size(1) == self.batchSize, 'batch sizes must be the same to remember states')
      h0:copy(h0_)
    end
  end
  
  local beta_expand = self.beta:view(1, 3 * H):expand(N, 3 * H) -- copy the beta for a batch
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
    local h1, mean1, var1, bn_in1, bn_out1 = self:GetPredecessorState(x, coor, self.hiddenDim, 1)
    local h2, mean2, var2, bn_in2, bn_out2 = self:GetPredecessorState(x, coor, self.hiddenDim - 1 - decompNodeId, 2)

    -- update the current node
    local hn = h[{{}, unpack(coor)}]
    local curGates = self.gates[{{}, unpack(coor)}] -- N * 3H

    if self.isBatchNorm then
      bn_in1:mm(h1, w1) -- w1 * h1
      self:batchNormForward(bn_in1, mean1, var1, bn_out1)
      bn_in2:mm(h2, w2) -- w2 * h2
      self:batchNormForward(bn_in2, mean2, var2, bn_out2)
      local gamma1_expand = self.gamma1:view(1, 3 * H):expand(N, 3 * H)
      local gamma2_expand = self.gamma2:view(1, 3 * H):expand(N, 3 * H)
      curGates:addcmul(beta_expand, bn_out1, gamma1_expand):addcmul(bn_out2, gamma2_expand) -- BN(w1 * h1) * gamma1 + BN(w2 * h2) * gamma2 + beta
    else
      curGates:addmm(beta_expand, h1, w1):addmm(h2, w2) -- w1 * h1 + w2 * h2 + b
    end

    curGates:narrow(2, 1, 2 * H):sigmoid() -- for gates
    curGates:narrow(2, 2 * H + 1, H):tanh() -- for new content
    local f  = curGates:narrow(2, 1, H)
    local s = curGates:narrow(2, H + 1, H)
    local g = curGates:narrow(2, 2 * H + 1, H)
    hn:cmul(s, h1):add(h2):addcmul(-1, s, h2):cmul(f):add(g):addcmul(-1, f, g) -- new hidden states
    self:MoveCoor(coor, 1)
  end

  self.output = self._output:contiguous()
  return self.output
end


function hidden:backward(input, gradOutput, scale)

  self.recompute_backward = false
  local x, h0 = self:UnpackInput(input)
  self:CheckSize(x, gradOutput)
  self.gradOutput:copy(gradOutput)
  scale = scale or 1.0
  assert(scale == 1.0, 'must have scale=1')

  local h = self.h
  local grad_h = self.grad_h
  if not h0 then h0 = self.h0 end

  local H, N = self.nodeSize, self.batchSize
  local w1 = self.weight[{{1, H}}]
  local w2 = self.weight[{{H + 1, 2 * H}}]
  local grad_w1 = self.gradWeight[{{1, H}}]
  local grad_w2 = self.gradWeight[{{H + 1, 2 * H}}]
  local gradIndicator = self.gradIndicator:clone()
  local buff3H = self.buff3H
  local buffNH = self.buffNH
  local buffN3H = self.buffN3H

  -- initialize the coordinate of current node
  local coor = {}
  for i = 1, self.hiddenDim do
    coor[i] = self.hiddenShape[i]
  end

  for nodeId = self.nodeNum, 1, -1 do
    local decompNodeId = (nodeId - 1) % self.decompNum + 1

    -- get the predecessor states
    local h1, mean1, var1, bn_in1, bn_out1 = self:GetPredecessorState(x, coor, self.hiddenDim, 1)
    local h2, mean2, var2, bn_in2, bn_out2 = self:GetPredecessorState(x, coor, self.hiddenDim - 1 - decompNodeId, 2)
    local h1_bn, h2_bn = h1, h2

    -- get the predecessor gradients
    local grad_h1, isGrad1 = self:GetPredecessorGrad(coor, self.hiddenDim, gradIndicator)
    local grad_h2, isGrad2 = self:GetPredecessorGrad(coor, self.hiddenDim - 1 - decompNodeId, gradIndicator)
    
    -- back propagate the gradients to predecessors
    local coorg = {}
    for i, v in ipairs(coor) do
      coorg[i] = v + 1
    end
    local grad_hn = grad_h[{{}, unpack(coorg)}] -- N * H
    local curGates = self.gates[{{}, unpack(coor)}] -- N * 3H
    local f  = curGates:narrow(2, 1, H)
    local s = curGates:narrow(2, H + 1, H)
    local g = curGates:narrow(2, 2 * H + 1, H)

    local grad_a = self.grad_a:zero() -- gradients of activations
    local grad_af  = grad_a:narrow(2, 1, H)
    local grad_as = grad_a:narrow(2, H + 1, H)
    local grad_ag = grad_a:narrow(2, 2 * H + 1, H)

    local one_minus_f = grad_ag:fill(1):add(-1, f) -- grad_ag is used as a buffer to store (1 - f)
    buffNH = h1:clone():add(-1, h2):cmul(s) -- a buffer to store (h1 - h2) * s

    -- gradients of gate activations
    grad_af:copy(buffNH):add(h2):add(-1, g):cmul(f):cmul(one_minus_f):cmul(grad_hn)
    grad_as:fill(1):add(-1, s):cmul(buffNH):cmul(f):cmul(grad_hn)
    buffNH:fill(1):addcmul(-1, g, g)
    grad_ag:cmul(buffNH):cmul(grad_hn)

    -- gradients of beta
    buff3H:sum(grad_a, 1) -- directly accumulate gradBeta (equal to grad_a) inside a batch
    self.gradBeta:add(scale, buff3H)

    -- gradients of the scaled batch normalization outputs
    local grad_h1w1, grad_h2w2 = grad_a, grad_a

    if self.isBatchNorm then
      -- gradients of gamma1
      buffN3H:cmul(grad_a, bn_out1)
      buff3H:sum(buffN3H, 1)
      self.gradGamma1:add(scale, buff3H)
      -- gradients of gamma2
      buffN3H:cmul(grad_a, bn_out2)
      buff3H:sum(buffN3H, 1)
      self.gradGamma2:add(scale, buff3H)
      -- gradients of batch normalization outputs
      local gamma1_expand = self.gamma1:view(1, 3 * H):expand(N, 3 * H)
      local gamma2_expand = self.gamma2:view(1, 3 * H):expand(N, 3 * H)
      grad_h1w1 = torch.cmul(grad_a, gamma1_expand)
      grad_h2w2 = torch.cmul(grad_a, gamma2_expand)
      -- gradients of batch normalization inputs
      self:batchNormBackward(bn_in1, mean1, var1, bn_out1, grad_h1w1)
      self:batchNormBackward(bn_in2, mean2, var2, bn_out2, grad_h2w2)
    end

    -- gradients of weights
    grad_w1:addmm(scale, h1:t(), grad_h1w1)
    grad_w2:addmm(scale, h2:t(), grad_h2w2)

    -- gradient of h1
    buffNH:cmul(f, s):cmul(grad_hn):addmm(grad_h1w1, w1:t())
    if not isGrad1 then -- if no previous gradient, we overwrite it
      grad_h1:copy(buffNH)
    else -- if previous gradient exists, we accumulate it
      grad_h1:add(buffNH)
    end

    -- gradient of h2
    buffNH:fill(1):add(-1, s):cmul(f):cmul(grad_hn):addmm(grad_h2w2, w2:t())
    if not isGrad2 then -- if no previous gradient, we overwrite it 
      grad_h2:copy(buffNH)
    else -- if previous gradient exists, we accumulate it
      grad_h2:add(buffNH)
    end
    
    self:MoveCoor(coor, -1)
  end

  self.grad_x = self._grad_x:contiguous()
  self.grad_h0 = self._grad_h0:contiguous()
  if self.isReturnGradH0 then
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
