require 'torch'
require 'nn'


local hidden, parent = torch.class('nn.TensHidden', 'nn.Module')


function hidden:__init(inputShape, tensShape, nodeSize, dropout)

  assert(#inputShape > 0, 'invalid input shape')
  assert(#tensShape > 0, 'invalid tensorizing shape')
  local layerNum = 1
  for i, v in ipairs(tensShape) do
    layerNum = layerNum + v - 1
  end
  assert(layerNum >= 3, 'the network must be at least 3 layers: x -> h -> y')
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
  self.tensDim = #self.tensShape
  self.hiddenDim = #inputShape + self.tensDim
  self.gateNum = 3 + self.hiddenDim -- new content, output gate, input gate, forget gates
  self.weight = torch.Tensor(self.hiddenDim * H, self.gateNum * H) 
  self.gradWeight = torch.Tensor(self.hiddenDim * H, self.gateNum * H):zero()
  self.bias = torch.Tensor(self.gateNum * H)
  self.gradBias = torch.Tensor(self.gateNum * H):zero()
  self:reset()

  self.buf_H_1 = torch.Tensor()
  self.buf_H_2 = torch.Tensor()
  self.buf_H_3 = torch.Tensor()
  self.buf_DH_1 = torch.Tensor()
  self.buf_DH_2 = torch.Tensor()
  self.buf_D1H_1 = torch.Tensor() -- (D + 1) * H
  self.buf_2DH_1 = torch.Tensor() -- 2 * D * H
  self.buf_2DH_2 = torch.Tensor()
  self.buf_GH_1 = torch.Tensor() -- G * H
  self.buf_GH_2 = torch.Tensor()
  self.buf_GH_3 = torch.Tensor()

  self.states = torch.Tensor()
  self.grads = torch.Tensor()

  self.gates = torch.Tensor() -- This will be (N, unpack(self.hiddenShape), gateNum * H)
  self.initialStateMask = torch.Tensor() -- This will be (N, unpack(self.hiddenShape), H)
  self.gradActivationMask = torch.Tensor() -- This will be (N, unpack(self.hiddenShape), H)
  self.grad_b_sum = torch.Tensor() -- This will be (1, gateNum * H)

  self.noise = torch.Tensor() -- This will be (N, unpack(self.hiddenShape), tensDim * H), for drop out
  self.h_dp = torch.Tensor() -- This will be (N, unpack(self.hiddenShape), tensDim * H), for drop out
  self.freezeNoise = false -- in gradient checking, we freeze the drop out noise to fix the network

  self.h0 = torch.Tensor()
  self.c0 = torch.Tensor()
  self.grad_x = torch.Tensor()
  self.grad_h0 = torch.Tensor()
  self.grad_c0 = torch.Tensor()
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
  -- self.bias[{{3 * H + 1, self.gateNum * H}}]:fill(1) -- set the bias of forget gates to 1
  
  return self
end


function hidden:resetStates()
  -- clear initial states h0 and c0
  self.h0 = self.h0.new()
  self.c0 = self.c0.new()
end


function hidden:InitState(input)

  -- check if the input is changed, in which case we need to reinitialize the network
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

  -- get the size of the network
  if not self.inputShape or isInputShapeChanged then
    self.batchSize = input:size(1)

    self.inputShape = {}
    for i = 2, input:dim() - 1 do
      table.insert(self.inputShape, input:size(i))
    end

    self.inputDim = #self.inputShape

    self.hiddenShape = {} -- the shape of the skewed block
    local l = 0
    for _, v in ipairs(self.inputShape) do
      table.insert(self.hiddenShape, v)
      l = l + v
    end
    for _, v in ipairs(self.tensShape) do
      table.insert(self.hiddenShape, v)
      l = l + v
    end
    assert(self.hiddenDim == #self.hiddenShape, 'the dimension of of input mismatched with inputShape')
    self.hiddenShape[1] = l - self.hiddenDim + 2
  end 

  -- initialize the network states
  local N = self.batchSize
  local H = self.nodeSize
  local I = self.inputDim
  local T = self.tensDim
  local D = self.hiddenDim
  local G = self.gateNum
  local S = D + 2
  local L = self.hiddenShape[1]

  if self.states:nElement() == 0 or isInputShapeChanged then

    -- intialize states and gradients
    local sz = {N}
    for i, v in ipairs(self.hiddenShape) do
      table.insert(sz, v)
    end
    table.insert(sz, 2 * D * H)
    self.states:resize(torch.LongStorage(sz)):zero()
    self.grads:resize(torch.LongStorage(sz)):zero()
    sz[#sz] = G * H
    self.gates:resize(torch.LongStorage(sz)):zero()
    sz[#sz] = H
    self.gradActivationMask:resize(torch.LongStorage(sz)):zero()
    self.initialStateMask:resize(torch.LongStorage(sz)):zero()

    -- define variables (with '_') sharing memory with states or gradients
    self._h = self.states:narrow(S, 1, D * H)
    self._c = self.states:narrow(S, D * H + 1, D * H)
    self._grad_h = self.grads:narrow(S, 1, D * H)
    self._grad_c = self.grads:narrow(S, D * H + 1, D * H)
    
    local initialState_region = {{}} -- whole batch
    for i, v in ipairs(self.hiddenShape) do
      if i == 1 then
        table.insert(initialState_region, 1) -- first slice on the first input dimension
      else
        table.insert(initialState_region, {})
      end
    end
    table.insert(initialState_region, {1, H})-- states of the first input dimension
    self._h0 = self._h[initialState_region]
    self._c0 = self._c[initialState_region]
    self._grad_h0 = self._grad_h[initialState_region]
    self._grad_c0 = self._grad_c[initialState_region]

    -- initialize the initial state mask
    self.initialStateMask[initialState_region]:fill(1)
    local before_input_region = {{}}
    local after_output_region = {{}} -- whole batch
    for i, v in ipairs(self.hiddenShape) do
      if i == 1 then
        table.insert(before_input_region, {1, self.inputShape[1] + 1}) -- except the first slice of the first input dimension
        table.insert(after_output_region, {1, self.inputShape[1] + 1}) -- except the first slice of the first input dimension
      elseif i <= I then
        table.insert(before_input_region, {})
        table.insert(after_output_region, {})
      else
        table.insert(before_input_region, 1) -- first slice on the tensorized dimensions
        table.insert(after_output_region, v) -- first slice on the tensorized dimensions
      end
    end
    table.insert(before_input_region, {})
    table.insert(after_output_region, {})
    -- we should mask the input predecessors and output successors
    self.initialStateMask[before_input_region]:zero()
    self.initialStateMask[after_output_region]:zero()
    self:SkewBlock(self.initialStateMask)

    -- mask for gate activation gradients (to prevent the unnecessary accumulating of the parameter gradients)
    self.gradActivationMask:narrow(2, 2, self.inputShape[1]):fill(1) -- pass the gradients except the initial and expanded ones
    self.gradActivationMask[before_input_region]:zero() -- mask the input predecessors
    self.gradActivationMask[after_output_region]:zero() -- mask the output successors
    self:SkewBlock(self.gradActivationMask) -- skew the mask

    local input_region = {{}} -- whole batch
    for i, v in ipairs(self.hiddenShape) do
      if i == 1 then
        table.insert(input_region, {2, self.inputShape[1] + 1}) -- except the first slice of the first input dimension
      elseif i <= I then
        table.insert(input_region, {})
      else
        table.insert(input_region, 1) -- first slice on the tensorized dimensions
      end
    end
    table.insert(input_region, {I * H + 1, D * H}) -- for the tensorized dimensions
    self._gradInput_h = self._grad_h[input_region]
    self._gradInput_c = self._grad_c[input_region]
    self._input_h = {}
    self._input_c = {}
    for d = 1, T do
      -- set the input region in the states at where we should place the original input
      input_region[2] = {2, self.inputShape[1] + 1} -- the first input dimension
      input_region[1 + I + d] = 2 -- the d-th tensorized dimension
      input_region[#input_region] = {(I + d - 1) * H + 1, (I + d - 1) * H + H}
      -- place the original input in the right place and recover the index
      self._input_h[d] = self._h[input_region]
      self._input_c[d] = self._c[input_region]
      input_region[1 + I + d] = 1 -- recover the d-th tensorized dimension
    end

    local output_region = {{}} -- whole batch
    for i, v in ipairs(self.hiddenShape) do
      if i == 1 then
        table.insert(output_region, {}) -- this will be modified
      elseif i <= I then
        table.insert(output_region, {})
      else
        table.insert(output_region, v) -- this will be modified
      end
    end
    table.insert(output_region, {}) -- this will be modified
    self._output_h = {}
    self._output_c = {}
    self._gradOutput_h = {}
    self._gradOutput_c = {}
    for d = 1, T do
      -- set the output region in the states at where we should place the original output
      output_region[2] = {L - 1 - self.inputShape[1] + 1, L - 1} -- the first input dimension
      output_region[1 + I + d] = self.tensShape[d] - 1 -- the d-th tensorized dimension
      output_region[#output_region] = {1, H} -- states of the first input dimension
      self._output_h[d] = self._h[output_region]
      self._output_c[d] = self._c[output_region]
      output_region[#output_region] = {(I + d - 1) * H + 1, (I + d - 1) * H + H} -- gradients of the last tensorized dimension
      self._gradOutput_h[d] = self._grad_h[output_region]
      self._gradOutput_c[d] = self._grad_c[output_region]
      output_region[1 + I + d] = self.tensShape[d] -- recover the d-th tensorized dimension
    end

    -- layer buffers
    table.remove(sz, 2) -- a slice along the first input dimension
    sz[#sz] = H
    self.buf_H_1:resize(torch.LongStorage(sz)):zero()
    self.buf_H_2:resize(torch.LongStorage(sz)):zero()
    self.buf_H_3:resize(torch.LongStorage(sz)):zero()
    sz[#sz] = D * H
    self.buf_DH_1:resize(torch.LongStorage(sz)):zero()
    self.buf_DH_2:resize(torch.LongStorage(sz)):zero()
    sz[#sz] = (D + 1) * H
    self.buf_D1H_1:resize(torch.LongStorage(sz)):zero()
    sz[#sz] = 2 * D * H
    self.buf_2DH_1:resize(torch.LongStorage(sz)):zero()
    self.buf_2DH_2:resize(torch.LongStorage(sz)):zero()
    sz[#sz] = G * H
    self.buf_GH_1:resize(torch.LongStorage(sz)):zero()
    self.buf_GH_2:resize(torch.LongStorage(sz)):zero()
    self.buf_GH_3:resize(torch.LongStorage(sz)):zero()
  end

  if self.grad_b_sum:nElement() == 0 then
    self.grad_b_sum:resize(1, G * H):zero()
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

end


function hidden:clearState()
  -- clear intermediate variables (the original clearState() in 'nn.Module' is overloaded, as it only clears 'output' and 'gradInput')

  -- network states
  self.states:set()
  self.grads:set()
  self.gates:set()
  self.initialStateMask:set()
  self.gradActivationMask:set()

  -- shared variables
  self._h:set()
  self._c:set()
  self._grad_h:set()
  self._grad_c:set()
  self._h0:set()
  self._c0:set()
  self._grad_h0:set()
  self._grad_c0:set()
  for d = 1, self.tensDim do
    self._input_h[d]:set()
    self._input_c[d]:set()
  end
  self._gradInput_h:set()
  self._gradInput_c:set()
  for d = 1, self.tensDim do
    self._output_h[d]:set()
    self._output_c[d]:set()
    self._gradOutput_h[d]:set()
    self._gradOutput_c[d]:set()
  end
  self._inputMask:set()

  -- layer buffers
  self.buf_H_1:set()
  self.buf_H_2:set()
  self.buf_H_3:set()
  self.buf_DH_1:set()
  self.buf_DH_2:set()
  self.buf_D1H_1:set()
  self.buf_2DH_1:set()
  self.buf_2DH_2:set()
  self.buf_GH_1:set()
  self.buf_GH_2:set()
  self.buf_GH_3:set()
  self.grad_b_sum:set()

  -- module interfaces
  self.output:set()
  self.grad_x:set()
  self.grad_h0:set()
  self.grad_c0:set()
  
  -- for drop out
  self.noise:set()
  self.h_dp:set()
end


function hidden:CheckSize(input, gradOutput)

  assert(torch.isTensor(input))
  assert(input:dim() >= 3) -- batch, input dim, node vector
  assert(input:size(input:dim()) == self.nodeSize * self.tensDim * 2) -- hidden vector and cell vector

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


function hidden:updateOutput(input)
  self.recompute_backward = true
  local x, h0, c0 = self:UnpackInput(input)
  self:CheckSize(x)
  self.isReturnGradH0 = (h0 ~= nil)
  self.isReturnGradC0 = (c0 ~= nil)
-- timer = torch.Timer()
  self:InitState(x)
-- print('\nA: ' .. timer:time().real * 1e3 .. ' ms');timer = torch.Timer()
  self:SetInitialStates(h0, c0)

-- print('\nB: ' .. timer:time().real * 1e3 .. ' ms');timer = torch.Timer()
  self:SetInput(x)

-- print('\nC: ' .. timer:time().real * 1e3 .. ' ms');timer = torch.Timer()
  self:UpdateStates()
-- print('\nD: ' .. timer:time().real * 1e3 .. ' ms');timer = torch.Timer()
  self:GetOutput()
-- print('\nE: ' .. timer:time().real * 1e3 .. ' ms');timer = torch.Timer()
  return self.output
end


function hidden:SetInitialStates(h0, c0)
  local H = self.nodeSize
  local D = self.hiddenDim
  local tmp = self.states:narrow(self.states:dim(), 1, H):select(2, 1)
  local isRemember = false

  if h0 then
    self._h:zero()
    self.h0 = h0:clone()
    self._h0:copy(self.h0)
    self:SkewBlock(self._h)
  else
    if self.h0:nElement() == 0 or not self.remember_states then
      self._h:zero()
      self.h0:resizeAs(tmp):zero()
    else
      isRemember = true
    end
  end

  if c0 then
    self._c:zero()
    self.c0 = c0:clone()
    self._c0:copy(self.c0)
    self:SkewBlock(self._c)
  else
    if self.c0:nElement() == 0 or not self.remember_states then
      self._c:zero()
      self.c0:resizeAs(tmp):zero()
    else
      isRemember = true
    end
  end

  -- extract initial states from previous batch
  if isRemember then
    local len = self.hiddenShape[1] - self.inputShape[1]
    local buf = self.grads:narrow(2, 1, len) -- use the grads as a buffer temporally 
    buf:copy(self.states:narrow(2, self.inputShape[1] + 1, len))
    self.states:narrow(2, 1, len):copy(buf)
  end

  -- mask other regions
  self._h:narrow(D + 2, 1, H):cmul(self.initialStateMask)
  self._h:narrow(D + 2, H + 1, (D - 1) * H):zero()
  self._c:narrow(D + 2, 1, H):cmul(self.initialStateMask)
  self._c:narrow(D + 2, H + 1, (D - 1) * H):zero()

end


function hidden:GetInitialStates()
  local H = self.nodeSize
  self:RecoverBlock(self.states)
  local h0 = self._h:narrow(self._h:dim(), 1, H):select(2, self.inputShape[1] + 1):clone()
  local c0 = self._c:narrow(self._c:dim(), 1, H):select(2, self.inputShape[1] + 1):clone()
  self.h0 = h0:clone()
  self.c0 = c0:clone()
  self:SkewBlock(self.states)

  return h0, c0
end


function hidden:SetInput(input)
  local H = self.nodeSize
  local I = self.inputDim
  local T = self.tensDim

  for d = 1, T do
    -- the original input for the d-th tensorized dimension
    local input_h_d = input:narrow(input:dim(), (d - 1) * H + 1, H)
    local input_c_d = input:narrow(input:dim(), (T + d - 1) * H + 1, H)
    -- place the original input in the right place
    self._input_h[d]:copy(input_h_d)
    self._input_c[d]:copy(input_c_d)
  end

  -- the input mask
  self._inputMask = self.grads:zero():narrow(self.grads:dim(), 1, T * H)
  local input_region = {{}} -- whole batch
  for i, v in ipairs(self.hiddenShape) do
    if i == 1 then
      table.insert(input_region, {2, self.inputShape[1] + 1}) -- except the first slice of the first input dimension
    elseif i <= I then
      table.insert(input_region, {})
    else
      table.insert(input_region, 1) -- first slice on the tensorized dimensions
    end
  end
  table.insert(input_region, {})
  for d = 1, T do
    input_region[1 + I + d] = 2 -- the d-th tensorized dimension
    input_region[#input_region] = {(d - 1) * H + 1, (d - 1) * H + H}
    self._inputMask[input_region]:fill(1)
    input_region[1 + I + d] = 1 -- recover the d-th tensorized dimension
  end

end


function hidden:SkewBlock(t)
  local L = t:size(2)
  for d = 2, self.hiddenDim do
    for i = 2, self.hiddenShape[d] do
      local slice = t:select(1 + d, i) -- the i-th slice along the d-th dimension (take account the batch dimension so +1)
      local slice_cp = slice[{{}, {1, L - i + 1}}]:clone()
      slice:zero()
      slice[{{}, {i, L}}]:copy(slice_cp)
    end
  end
end


function hidden:RecoverBlock(t)
  local L = t:size(2)
  for d = 2, self.hiddenDim do
    for i = 2, self.hiddenShape[d] do
      local slice = t:select(1 + d, i) -- the i-th slice along the d-th dimension (take account the batch dimension so +1)
      local slice_cp = slice[{{}, {i, L}}]:clone()
      slice:zero()
      slice[{{}, {1, L - i + 1}}]:copy(slice_cp)
    end
  end
end


function hidden:UpdateStates()
  local H = self.nodeSize
  local I = self.inputDim
  local T = self.tensDim
  local D = self.hiddenDim
  local G = self.gateNum
  local L = self.states:size(2)

  local buf_H_1 = self.buf_H_1:view(-1, H)
  local buf_H_2 = self.buf_H_2:view(-1, H)
  local buf_H_3 = self.buf_H_3
  local buf_DH_1 = self.buf_DH_1:view(-1, D * H)
  local buf_TH_1 = self.buf_DH_2:narrow(self.buf_DH_2:dim(), 1, T * H)
  local buf_2DH_1 = self.buf_2DH_1
  local buf_2DH_2 = self.buf_2DH_2
  local buf_GH_1 = self.buf_GH_1
  local bias_expand = self.bias:view(1, G * H):expand(buf_H_1:size(1), G * H)

  for l = 2, L do
    -- extract predecessor states for current layer
    local s_prev_all = buf_2DH_1:copy(self.states:select(2, l - 1)):view(-1, 2 * D * H) -- a slice along the fisrt input dimension
    local h_prev_all = s_prev_all:narrow(2, 1, D * H)
    local c_prev_all = s_prev_all:narrow(2, D * H + 1, D * H)

    -- update gates
    local gates = buf_GH_1:copy(self.gates:select(2, l)):view(-1, G * H)
    gates:addmm(bias_expand, h_prev_all, self.weight)
    local g = gates:narrow(2, 1, H):tanh() -- new content
    local o = gates:narrow(2, H + 1, H) -- output gate
    local i = gates:narrow(2, 2 * H + 1, H) -- input gate
    local f = gates:narrow(2, 3 * H + 1, D * H) -- forget gates
    gates:narrow(2, H + 1, (D + 2) * H):sigmoid()
    self.gates:select(2, l):copy(buf_GH_1) -- save the gate values for backward

    -- update current states
    local c_mem_all = buf_DH_1:cmul(f, c_prev_all):view(-1, D, H)
    local c_mem = buf_H_1:view(-1, 1, H):sum(c_mem_all, 2):view(-1, H)
    local c_new_ = buf_H_2:cmul(i, g):add(c_mem) -- new memory cells
    local h_new = buf_H_1:tanh(c_new_):cmul(o):viewAs(self.buf_H_1) -- new hidden states
    local c_new = c_new_:viewAs(self.buf_H_1)

    -- prepare predecessor states for the next layer
    local s_cur_all = buf_2DH_1:copy(self.states:select(2, l)) -- a slice along the fisrt input dimension
    local h_cur_all = s_cur_all:narrow(s_cur_all:dim(), 1, D * H)
    local c_cur_all = s_cur_all:narrow(s_cur_all:dim(), D * H + 1, D * H)
    for d = 1, D do
      local h_cur_d = h_cur_all:narrow(h_cur_all:dim(), (d - 1) * H + 1, H) -- h of the d-th dimension
      local c_cur_d = c_cur_all:narrow(c_cur_all:dim(), (d - 1) * H + 1, H) -- c of the d-th dimension
      if d == 1 then
        h_cur_d:copy(h_new)
        c_cur_d:copy(c_new)
      else
        local len = h_cur_d:size(d)
        if len > 1 then
          h_cur_d:narrow(d, 2, len - 1):copy(h_new:narrow(d, 1, len - 1))
          c_cur_d:narrow(d, 2, len - 1):copy(c_new:narrow(d, 1, len - 1))
        end
      end
    end

     -- for the node where initial state exists, we block the updated state for the first input dimension and use the original one
    local mask_all = buf_2DH_2:zero() -- initial state mask for all directions of h and c
    local mask_1 = buf_H_3:copy(self.initialStateMask:select(2, l)) -- initial state mask for the first input dimension
    mask_all:narrow(mask_all:dim(), 1, H):copy(mask_1)
    mask_all:narrow(mask_all:dim(), D * H + 1, H):copy(mask_1)
    -- for the node where input exists, we block the updated state for the last tensorized dimension and use the original one
    local mask_T = buf_TH_1:copy(self._inputMask:select(2, l)) -- input mask for the all the tensorized dimensions
    mask_all:narrow(mask_all:dim(), I * H + 1, (D - 1) * H):copy(mask_T)
    mask_all:narrow(mask_all:dim(), (D + I) * H + 1, (D - 1) * H):copy(mask_T)

    -- final states = reserved states + filtered new states
    local s_cur_all_res = self.states:select(2, l):cmul(mask_all) -- the reserved states
    local s_cur_all_new = mask_all:mul(-1):add(1):cmul(s_cur_all) -- the filtered new states
    local s_cur_all_final = s_cur_all_res:add(s_cur_all_new)
  
  end

end


function hidden:GetOutput() -- operated on skewed block
  local I = self.inputDim
  self.output = torch.cat(torch.cat(self._output_h, I + 2), torch.cat(self._output_c, I + 2), I + 2)
end


function hidden:backward(input, gradOutput, scale)

  self.recompute_backward = false
  local x, h0, c0 = self:UnpackInput(input)
  self:CheckSize(x, gradOutput)
-- timer = torch.Timer()

  self:SetGradOutput(gradOutput)
-- print('\nA: ' .. timer:time().real * 1e3 .. ' ms');timer:reset()

  self:UpdateGrads(scale)
-- print('\nB: ' .. timer:time().real * 1e3 .. ' ms');timer:reset()

  self:GetGradInput()
-- print('\nC: ' .. timer:time().real * 1e3 .. ' ms');timer:reset()

  if self.isReturnGradH0 or self.isReturnGradC0 then
    self:GetGradInitialStates()
  end

-- print('\nD: ' .. timer:time().real * 1e3 .. ' ms');timer:reset()
  if self.isReturnGradH0 and self.isReturnGradC0 then
    self.gradInput = {self.grad_x, self.grad_h0, self.grad_c0}
  elseif self.isReturnGradH0 then
    self.gradInput = {self.grad_x, self.grad_h0}
  else
    self.gradInput = self.grad_x
  end
  return self.gradInput

end


function hidden:SetGradOutput(gradOutput)
  local H = self.nodeSize
  local T = self.tensDim

  self.grads:zero()
  for d = 1, T do
    -- the original gradOutput region for the d-th tensorized dimension
    local gradOutput_h_d = gradOutput:narrow(gradOutput:dim(), (d - 1) * H + 1, H)
    local gradOutput_c_d = gradOutput:narrow(gradOutput:dim(), (T + d - 1) * H + 1, H)
    -- place the original gradOutput in the right place
    self._gradOutput_h[d]:copy(gradOutput_h_d)
    self._gradOutput_c[d]:copy(gradOutput_c_d)
  end
end


function hidden:UpdateGrads(scale)
  scale = scale or 1.0
  assert(scale == 1.0, 'must have scale=1')
  local H = self.nodeSize
  local D = self.hiddenDim
  local G = self.gateNum
  local L = self.grads:size(2)

  local buf_H_1 = self.buf_H_1:view(-1, H)
  local buf_H_2 = self.buf_H_2:view(-1, H)
  local buf_H_3 = self.buf_H_3:view(-1, H)
  local buf_DH_1 = self.buf_DH_1:view(-1, D * H)
  local buf_DH_2 = self.buf_DH_2:view(-1, D * H)
  local buf_D1H_1 = self.buf_D1H_1:view(-1, (D + 1) * H)
  local buf_2DH_1 = self.buf_2DH_1
  local buf_2DH_2 = self.buf_2DH_2
  local buf_GH_1 = self.buf_GH_1
  local buf_GH_2 = self.buf_GH_2:view(-1, G * H)
  local buf_GH_3 = self.buf_GH_3:view(-1, G * H)

  for l = L - 1, 3, -1 do

-- local timer = torch.Timer()
    -- extract states of previous layer
    local s_prev_all = buf_2DH_1:copy(self.states:select(2, l - 1)):view(-1, 2 * D * H) -- a slice along the fisrt input dimension
    local h_prev_all = s_prev_all:narrow(2, 1, D * H)
    local c_prev_all = s_prev_all:narrow(2, D * H + 1, D * H)
-- print('\nA: ' .. timer:time().real * 1e3 .. ' ms');timer:reset()
    -- extract gradients of current layer (unsummed gradients from all directions)
    local grad_s_cur_all = buf_2DH_2:copy(self.grads:select(2, l)):view(-1, 2 * D * H)
    local grad_h_cur_all = grad_s_cur_all:narrow(2, 1, D * H)
    local grad_c_cur_all = grad_s_cur_all:narrow(2, D * H + 1, D * H)
-- print('\n' ..l..': ' .. timer:time().real * 1e3 .. ' ms');timer:reset()
    -- sum the gradients from all directions
    local grad_h_cur_all_ = buf_DH_1:copy(grad_h_cur_all):view(-1, D, H)
    local grad_h_cur = buf_H_1:view(-1, 1, H):sum(grad_h_cur_all_, 2):view(-1, H)
    local grad_c_cur_all_ = buf_DH_1:copy(grad_c_cur_all):view(-1, D, H)
    local grad_c_cur = buf_H_2:view(-1, 1, H):sum(grad_c_cur_all_, 2):view(-1, H)

    -- extract gates of current layer
    local gates = buf_GH_1:copy(self.gates:select(2, l)):view(-1, G * H)
    local g = gates:narrow(2, 1, H)
    local o = gates:narrow(2, H + 1, H)
    local i = gates:narrow(2, 2 * H + 1, H)
    local f = gates:narrow(2, 3 * H + 1, D * H)
    local o_i_f = gates:narrow(2, H + 1, (G - 1) * H)

    -- gradients of gate activations
    local grad_a = buf_GH_2:zero() -- gradients of activations
    local grad_ag = grad_a:narrow(2, 1, H)
    local grad_ao = grad_a:narrow(2, H + 1, H)
    local grad_ai = grad_a:narrow(2, 2 * H + 1, H)
    local grad_af = grad_a:narrow(2, 3 * H + 1, D * H)
    local grad_aif = grad_a:narrow(2, 2 * H + 1, (G - 2) * H)
    local grad_aoif = grad_a:narrow(2, H + 1, (G - 1) * H)

    local s_cur_all = buf_2DH_2:copy(self.states:select(2, l)):view(-1, 2 * D * H) -- current states
    local c_cur = s_cur_all:narrow(2, D * H + 1, H) -- current c
    local tanh_next_c = buf_H_3:tanh(c_cur)
    local tanh_next_c2 = grad_ao:cmul(tanh_next_c, tanh_next_c)
    local grad_from_h = grad_ag
    grad_from_h:fill(1):add(-1, tanh_next_c2):cmul(o):cmul(grad_h_cur)
    grad_c_cur:add(grad_from_h) -- accumulate the gradient of cell from current hidden state

    -- gradients of ao, ag, ai, af
    grad_aoif:fill(1):add(-1, o_i_f):cmul(o_i_f)
    grad_ao:cmul(tanh_next_c):cmul(grad_h_cur) -- grad ao
    local g2 = buf_H_3:cmul(g, g)
    grad_ag:fill(1):add(-1, g2):cmul(i):cmul(grad_c_cur) -- grad ag
    local grad_c_cur_expand = buf_D1H_1
    for d = 0, D do
      grad_c_cur_expand:narrow(2, d * H + 1, H):copy(grad_c_cur)
    end
    grad_aif:cmul(grad_c_cur_expand)
    grad_ai:cmul(g) -- grad ai
    grad_af:cmul(c_prev_all) -- grad af

    -- set the activation gradients to zero in some skewed region where we don't want to accumulate the parametre gradients
    local mask_d = self.buf_H_1:copy(self.gradActivationMask:select(2, l)):view(-1, H) -- gradient mask for one gate activations
    local mask_all = buf_GH_3:zero() -- gradient mask for all gate activations
    for d = 1, G do
      mask_all:narrow(2, (d - 1) * H + 1, H):copy(mask_d) -- to save the memory, but will cost some time
    end
    grad_a:cmul(mask_all)

    -- gradients of parameters
    self.gradWeight:addmm(scale, h_prev_all:t(), grad_a)
    self.grad_b_sum:sum(grad_a, 1) -- directly accumulate grad_b (equal to grad_a) inside a batch
    self.gradBias:add(scale, self.grad_b_sum)

    -- gradients of previous hidden states and memory cells (for all the direction)
    grad_h_prev_all_new = buf_DH_1:mm(grad_a, self.weight:t())
    grad_c_prev_all_new = buf_DH_2
    for d = 1, D do
      grad_c_prev_all_new:narrow(2, (d - 1) * H + 1, H):copy(grad_c_cur)
    end
    grad_c_prev_all_new:cmul(f)

    -- extract gradients of previous layer
    local grad_s_prev_all = self.grads:select(2, l - 1) -- gradients including the input gradients
    local S = grad_s_prev_all:dim()
    local grad_h_prev_all = grad_s_prev_all:narrow(S, 1, D * H)
    local grad_c_prev_all = grad_s_prev_all:narrow(S, D * H + 1, D * H)

    -- propagate the gradients to previous layer (align the gradients)
    for d = 1, D do

      -- extract gradients in each direction
      local grad_h_prev_d_new = buf_H_1:copy(grad_h_prev_all_new:narrow(2, (d - 1) * H + 1, H)):viewAs(self.buf_H_1)
      local grad_c_prev_d_new = buf_H_2:copy(grad_c_prev_all_new:narrow(2, (d - 1) * H + 1, H)):viewAs(self.buf_H_1)
      local grad_h_prev_d = grad_h_prev_all:narrow(S, (d - 1) * H + 1, H)
      local grad_c_prev_d = grad_c_prev_all:narrow(S, (d - 1) * H + 1, H)
      -- shift the gradients' position along each direction 
      if d == 1 then -- directly copy along the first input dimension
        grad_h_prev_d:add(grad_h_prev_d_new)
        grad_c_prev_d:add(grad_c_prev_d_new)
      else -- shifted copy along the other dimensions (the gradients will flow out)
        local len = grad_h_prev_d:size(d)
        if len > 1 then
          grad_h_prev_d:narrow(d, 1, len - 1):add(grad_h_prev_d_new:narrow(d, 2, len - 1))
          grad_c_prev_d:narrow(d, 1, len - 1):add(grad_c_prev_d_new:narrow(d, 2, len - 1))
        end
      end
    end

  end
-- print(self.grads:select(2, 2))
end


function hidden:GetGradInput()
  self.grad_x = torch.cat(self._gradInput_h, self._gradInput_c, self.inputDim + 2)
end


function hidden:GetGradInitialStates()
  self:RecoverBlock(self.grads)
  self.grad_h0 = self._grad_h0:clone()
  self.grad_c0 = self._grad_c0:clone()
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
