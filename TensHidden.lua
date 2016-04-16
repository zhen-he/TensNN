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
  self.bias = torch.Tensor(3 * H)
  self.gradBias = torch.Tensor(3 * H):zero()
  self:reset()

  self.buff = torch.Tensor() -- This will be (N, H), a small buffer
  self.stateAndGradBuff = torch.Tensor()

  self.gates = torch.Tensor() -- This will be (N, unpack(self.hiddenShape), 3H)
  self.grad_b_sum = torch.Tensor() -- This will be (1, 3H)
  self.grad_a = torch.Tensor() -- This will be (N, 3H)
  self.gradIndicator = torch.ByteTensor() -- for gradients accumulation

  self.norms = torch.Tensor()
  self.means = torch.Tensor() -- This will be (unpack(self.hiddenShape), H)
  self.vars = torch.Tensor() -- This will be (unpack(self.hiddenShape), H)
  self.meansAvg = torch.Tensor() -- This will be (unpack(self.hiddenShape), H)
  self.varsAvg = torch.Tensor() -- This will be (unpack(self.hiddenShape), H)
  self.normIndicator = torch.ByteTensor() -- for batch normalization
  self.gam = 1
  self.ep = 0.00001

  self.h0 = torch.Tensor()
  self.remember_states = false
  self.train = self.train or true
end


function hidden:reset(std)
  local H = self.nodeSize
  if not std then
    std = 1.0 / math.sqrt(H * 2)
  end
  self.weight:normal(0, std) --:add(torch.eye(H):repeatTensor(2, 3))
  self.bias:zero()
  self.bias[{{1, self.nodeSize}}]:fill(1) -- set the bias of forget gates to 1
  
  if self.meansAvg then
    self.meansAvg:zero()
  end
  if self.varsAvg then
    self.varsAvg:fill(1)
  end

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
    self.buff:resize(N, H):zero()
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
    if self.means:nElement() == 0 or isInputShapeChanged then
      local sz = {}
      for _, v in ipairs(self.hiddenShape) do
        table.insert(sz, v + 1)
      end
      self.normIndicator:resize(torch.LongStorage(sz)):zero()
      table.insert(sz, 1, 1)
      table.insert(sz, H)
      self.means:resize(torch.LongStorage(sz)):zero()
      self.vars:resize(torch.LongStorage(sz)):zero()
      sz[1] = N
      self.norms:resize(torch.LongStorage(sz)):zero()
    end
    if self.meansAvg:nElement() == 0 or isInputShapeChanged then
      local sz = {1}
      for _, v in ipairs(self.hiddenShape) do
        table.insert(sz, v + 1)
      end
      table.insert(sz, H)
      self.meansAvg:resize(torch.LongStorage(sz)):zero()
      self.varsAvg:resize(torch.LongStorage(sz)):fill(1)
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

  if self.grad_b_sum:nElement() == 0 then
    self.grad_b_sum:resize(1, 3 * H):zero() -- This will be (1, 3H)
  end

  if self.grad_a:nElement() == 0 then
    self.grad_a:resize(N, 3 * H):zero() -- This will be (N, 3H)
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

  self.buff:set()
  self.stateAndGradBuff:set()
  self.h:set()
  self.output:set()
  self._output:set()
  self.gates:set()

  self.grad_x:set()
  self._grad_x:set()
  self.grad_h:set()
  self.grad_h0:set()
  self._grad_h0:set()
  self.gradOutput:set()
  self.grad_b_sum:set()
  self.grad_a:set()
  self.gradIndicator:set()

  self.norms:set()
  self.means:set()
  self.vars:set()
  self.normIndicator:set()
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


function hidden:GetPredecessorState(input, curCoor, predecessorDim)

  local H, N = self.nodeSize, self.batchSize
  local hp = torch.Tensor():type(self.weight:type())
  local normAllowed = false

  local preCoor = {}
  for i, v in ipairs(curCoor) do
    preCoor[i] = v
  end

  if predecessorDim == self.hiddenDim and preCoor[self.hiddenDim] == 1 then
    predecessorDim = self.hiddenDim - 1
  end
  if predecessorDim < self.hiddenDim then -- if not along the decomposing dimension
    preCoor[self.hiddenDim] = self.decompNum -- point to the last decompesed one of the other dimension
    if predecessorDim > self.inputDim then -- batch normalization is only allowed along the tensorized dimension
      normAllowed = true
    end
  end

  local meanp, varp, normp, meanAvgp, varAvgp = nil, nil, nil, nil, nil
  local preNormCoor = {} -- for the normalization of the predecessor
  if self.isBatchNorm then
    for i, v in ipairs(preCoor) do
      preNormCoor[i] = v + 1
    end
    preNormCoor[predecessorDim] = preNormCoor[predecessorDim] - 1
    meanp = self.means[{{}, unpack(preNormCoor)}] -- 1 * H
    varp = self.vars[{{}, unpack(preNormCoor)}] -- 1 * H
    meanAvgp = self.meansAvg[{{}, unpack(preNormCoor)}] -- 1 * H
    varAvgp = self.varsAvg[{{}, unpack(preNormCoor)}] -- 1 * H
    normp = self.norms[{{}, unpack(preNormCoor)}] -- N * H
  end

  if preCoor[predecessorDim] > 1 then
    -- point to the previous node
    preCoor[predecessorDim] = preCoor[predecessorDim] - 1
    hp = self.h[{{}, unpack(preCoor)}] -- N * H
  else -- the case that requires initial states (out of the network's shape)
    hp:resize(N, H):zero()
    if predecessorDim == self.inputDim then -- get value from the last states of previous batch
      -- as h0 are the last slice on both input dimension and decomposed dimension of h (c), we remove the corresponding coordinates
      table.remove(preCoor, self.inputDim) -- remove the input dimension
      table.remove(preCoor, #preCoor) -- remove the decomposing dimension
      hp = self.h0[{{}, unpack(preCoor)}] -- N * H
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
        hp = input[{{}, unpack(inputCoor)}] -- N * H
      end
    end
  end

  local isNormed = nil
  if self.isBatchNorm then
    isNormed = true
    if self.normIndicator[preNormCoor] == 0 then
      isNormed = false
      self.normIndicator[preNormCoor] = 1 
    end
  end

  return hp, meanp, varp, normp, normAllowed, isNormed, meanAvgp, varAvgp
end


function hidden:GetPredecessorGrad(curCoor, predecessorDim, gradIndicator)

  local preCoor = {}
  for i, v in ipairs(curCoor) do
    preCoor[i] = v
  end

  if predecessorDim == self.hiddenDim and preCoor[self.hiddenDim] == 1 then
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


function hidden:batchNormForward(input, mean, var, meanAvg, varAvg, output)
  local N, H = input:size(1), input:size(2)
  local buff = self.buff
  
  local meanp, varp = meanAvg, varAvg
  if self.train then
    meanp = mean:mean(input, 1)
    varp = var:var(input, 1, true) -- biased var
    meanAvg:mul(0.9):add(0.1, mean)
    varAvg:mul(0.9):add(N / (N - 1) * 0.1, var) -- unbiased var
  end
  output:copy(meanp:expand(N, H)):mul(-1):add(input)
  buff:copy(varp:expand(N, H)):add(self.ep):sqrt()
  output:cdiv(buff):mul(self.gam)
end


function hidden:batchNormBackward(input, mean, var, output, gradOutput)
  local N, H = input:size(1), input:size(2)
  local buff = self.buff

  buff:copy(gradOutput):div(N)
  local grad_output_mean = buff:sum(1):expand(N, H)
  local product_mean = buff:cmul(output):sum(1):expand(N, H)
  local std_div_gam = buff:copy(var:expand(N, H)):add(self.ep):sqrt():div(self.gam)
  gradOutput:add(-1, grad_output_mean):addcmul(-1, output, product_mean):cdiv(std_div_gam)
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
  
  local bias_expand = self.bias:view(1, 3 * H):expand(N, 3 * H) -- copy the bias for a batch
  local w1 = self.weight[{{1, H}}] -- weights for h1
  local w2 = self.weight[{{H + 1, 2 * H}}] -- weights for h2

  if self.isBatchNorm then
    self.normIndicator:zero()
  end

  -- initialize the coordinate of current node
  local coor = {}
  for i = 1, self.hiddenDim do
    coor[i] = 1
  end

  for nodeId = 1, self.nodeNum do
    local decompNodeId = (nodeId - 1) % self.decompNum + 1

    -- get the predecessor states
    local h1, mean1, var1, norm1, normAllowed1, isNormed1, meanAvg1, varAvg1
      = self:GetPredecessorState(x, coor, self.hiddenDim)
    local h2, mean2, var2, norm2, normAllowed2, isNormed2, meanAvg2, varAvg2 
      = self:GetPredecessorState(x, coor, self.hiddenDim - 1 - decompNodeId)
    local h1_bn, h2_bn = h1, h2

   -- batch normaliztion
    if self.isBatchNorm then
      if normAllowed1 then
        if not isNormed1 then
          self:batchNormForward(h1, mean1, var1, meanAvg1, varAvg1, norm1)
        end
        h1_bn = norm1 
      end
      if normAllowed2 then 
        if not isNormed2 then
          self:batchNormForward(h2, mean2, var2, meanAvg2, varAvg2, norm2)
        end
        h2_bn = norm2 
      end
    end

    -- update the current node
    local hn = h[{{}, unpack(coor)}]
    local curGates = self.gates[{{}, unpack(coor)}] -- N * 3H
    curGates:addmm(bias_expand, h1_bn, w1) -- w1 * h1 + b
    curGates:addmm(h2_bn, w2) -- w1 * h1 + b + w2 * h2
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
  local grad_b = self.gradBias[{{1, 3 * H}}]
  local grad_b_sum = self.grad_b_sum
  local gradIndicator = self.gradIndicator:clone()
  local buff = self.buff

  -- initialize the coordinate of current node
  local coor = {}
  for i = 1, self.hiddenDim do
    coor[i] = self.hiddenShape[i]
  end

  for nodeId = self.nodeNum, 1, -1 do
    local decompNodeId = (nodeId - 1) % self.decompNum + 1

    -- get the predecessor states
    local h1, mean1, var1, norm1, normAllowed1 = self:GetPredecessorState(x, coor, self.hiddenDim)
    local h2, mean2, var2, norm2, normAllowed2 = self:GetPredecessorState(x, coor, self.hiddenDim - 1 - decompNodeId)
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

    local grad_a = self.grad_a:zero()
    local grad_af  = grad_a:narrow(2, 1, H)
    local grad_as = grad_a:narrow(2, H + 1, H)
    local grad_ag = grad_a:narrow(2, 2 * H + 1, H)

    local one_minus_f = grad_ag:fill(1):add(-1, f) -- grad_ag is used as a buffer to store (1 - f)
    buff = h1:clone():add(-1, h2):cmul(s) -- a buffer to store (h1 - h2) * s

    -- gradients of activations
    grad_af:copy(buff):add(h2):add(-1, g):cmul(f):cmul(one_minus_f):cmul(grad_hn)
    grad_as:fill(1):add(-1, s):cmul(buff):cmul(f):cmul(grad_hn)
    buff:fill(1):addcmul(-1, g, g)
    grad_ag:cmul(buff):cmul(grad_hn)

    -- gradients of batch normalization outputs
    local grad_h1_bn = buff:mm(grad_a, w1:t()):clone()
    local grad_h2_bn = buff:mm(grad_a, w2:t()):clone()

    -- gradients of batch normalization inputs
    if self.isBatchNorm then
      if normAllowed1 then
        self:batchNormBackward(h1, mean1, var1, norm1, grad_h1_bn)
        h1_bn = norm1
      end
      if normAllowed2 then
        self:batchNormBackward(h2, mean2, var2, norm2, grad_h2_bn)
        h2_bn = norm2
      end
    end

    -- gradients of parameters
    grad_w1:addmm(scale, h1_bn:t(), grad_a)
    grad_w2:addmm(scale, h2_bn:t(), grad_a)
    grad_b_sum:sum(grad_a, 1) -- directly accumulate grad_b (equal to grad_a) inside a batch
    grad_b:add(scale, grad_b_sum)

    -- gradients of h1
    buff:cmul(f, s):cmul(grad_hn):add(grad_h1_bn)
    if not isGrad1 then -- if no previous gradient, we overwrite it
      grad_h1:copy(buff)
    else -- if previous gradient exists, we accumulate it
      grad_h1:add(buff)
    end

    -- gradients of h2
    buff:fill(1):add(-1, s):cmul(f):cmul(grad_hn):add(grad_h2_bn)
    if not isGrad2 then -- if no previous gradient, we overwrite it 
      grad_h2:copy(buff)
    else -- if previous gradient exists, we accumulate it
      grad_h2:add(buff)
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
