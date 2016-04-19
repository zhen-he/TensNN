require 'torch'
require 'nn'
require 'optim'

require 'TensLM'
require 'util.DataLoader'

local utils = require 'util.utils'
local unpack = unpack or table.unpack

local cmd = torch.CmdLine()

-- Dataset options
cmd:option('-input_h5', 'data/enwik8.h5')
cmd:option('-input_json', 'data/enwik8.json')
cmd:option('-batch_size', 100)
cmd:option('-seq_length', 100)

-- Model options
cmd:option('-init_from', '')
cmd:option('-rnn_size', 700)
cmd:option('-tensShape', {1})
cmd:option('-batchnorm', 'no') -- no, input, tensor, all

-- Optimization options
cmd:option('-max_epochs', 50)
cmd:option('-learning_rate', 1e-3)
cmd:option('-grad_clip', 1)
cmd:option('-lr_decay_every', 5)
cmd:option('-lr_decay_factor', 1.0)

-- Output options
cmd:option('-print_every', 10)
cmd:option('-checkpoint_every', 1000)
cmd:option('-result_dir', 'result/')

-- Benchmark options
cmd:option('-speed_benchmark', 0) -- record the time consuming
cmd:option('-memory_benchmark', 0) -- record the memory usage

-- Backend options
cmd:option('-gpu', 0)
cmd:option('-gpu_backend', 'cuda')

local opt = cmd:parse(arg)


-- directory names for saving
local filenamehd = 't'
for _, v in ipairs(opt.tensShape) do filenamehd = filenamehd .. v end
filenamehd = filenamehd .. '_s' .. opt.rnn_size .. '_' .. opt.batchnorm .. 'BN'
filenamehd = opt.result_dir .. filenamehd .. '/' .. filenamehd .. '_'


-- Set up GPU stuff
local dtype = 'torch.FloatTensor'
if opt.gpu >= 0 and opt.gpu_backend == 'cuda' then
  require 'cutorch'
  require 'cunn'
  cutorch.setDevice(opt.gpu + 1)
  dtype = 'torch.CudaTensor'
  print(string.format('Running with CUDA on GPU %d', opt.gpu))
elseif opt.gpu >= 0 and opt.gpu_backend == 'opencl' then
  -- Memory benchmarking is only supported in CUDA mode
  -- TODO: Time benchmarking is probably wrong in OpenCL mode.
  require 'cltorch'
  require 'clnn'
  cltorch.setDevice(opt.gpu + 1)
  dtype = torch.Tensor():cl():type()
  print(string.format('Running with OpenCL on GPU %d', opt.gpu))
else
  -- Memory benchmarking is only supported in CUDA mode
  opt.memory_benchmark = 0
  print 'Running in CPU mode'
end


-- Initialize the DataLoader and vocabulary
local loader = DataLoader(opt)
local vocab = utils.read_json(opt.input_json)
local idx_to_token = {}
for k, v in pairs(vocab.idx_to_token) do
  idx_to_token[tonumber(k)] = v
end


-- Initialize the model and criterion
local model = nil  -- model
if opt.init_from ~= '' then
  print('Initializing from ', opt.init_from)
  model = torch.load(opt.init_from).model:type(dtype)
else
  local opt_clone = torch.deserialize(torch.serialize(opt)) -- used as copy
  opt_clone.idx_to_token = idx_to_token
  model = nn.TensLM(opt_clone):type(dtype)
end
local params, grad_params = model:getParameters() -- parameters
local crit = nn.CrossEntropyCriterion():type(dtype) -- criterion (the output is score so we use this one)


-- Set up some variables we will use below
local N, T = opt.batch_size, opt.seq_length
local train_loss_history = {}
local val_loss_history = {}
local val_loss_history_it = {}
local forward_backward_times = {} -- record the spent time of each forward backward pass
local init_memory_usage, memory_usage = nil, {} -- record the memory usage

if opt.memory_benchmark == 1 then
  -- This should only be enabled in GPU mode
  assert(cutorch)
  cutorch.synchronize()
  local free, total = cutorch.getMemoryUsage(cutorch.getDevice())
  init_memory_usage = total - free
end


-- Loss function that we pass to an optim method
local function f(w)
  assert(w == params)
  grad_params:zero() -- clear the gradients of parameters

  -- Get a minibatch and run the model forward, maybe timing it
  local timer
  local x, y = loader:nextBatch('train')
  x, y = x:type(dtype), y:type(dtype)
  if opt.speed_benchmark == 1 then
    if cutorch then cutorch.synchronize() end
    timer = torch.Timer()
  end
  local scores = model:forward(x)

  -- Use the Criterion to compute loss; we need to reshape the scores to be
  -- two-dimensional before doing so. Annoying.
  local scores_view = scores:view(N * T, -1)
  local y_view = y:view(N * T)
  local loss = crit:forward(scores_view, y_view)

  -- Run the Criterion and model backward to compute gradients, maybe timing it
  local grad_scores = crit:backward(scores_view, y_view):view(N, T, -1)
  model:backward(x, grad_scores)
  if timer then
    if cutorch then cutorch.synchronize() end
    local time = timer:time().real
    print('Forward / Backward pass took ', time)
    table.insert(forward_backward_times, time)
  end

  -- Maybe record memory usage
  if opt.memory_benchmark == 1 then
    assert(cutorch)
    if cutorch then cutorch.synchronize() end
    local free, total = cutorch.getMemoryUsage(cutorch.getDevice())
    local memory_used = total - free - init_memory_usage
    local memory_used_mb = memory_used / 1024 / 1024
    print(string.format('Using %dMB of memory', memory_used_mb))
    table.insert(memory_usage, memory_used)
  end

  if opt.grad_clip > 0 then
    grad_params:clamp(-opt.grad_clip, opt.grad_clip)
  end

  return loss, grad_params
end


-- Train the model!
local optim_config = {learningRate = opt.learning_rate, beta2 = 0.99}
local num_train = loader.split_sizes['train']
local num_iterations = opt.max_epochs * num_train -- the maximum iteration number
model:training()
for i = 1, num_iterations do

  -- Take a gradient step and maybe print, note that adam returns a singleton array of losses
  local _, loss = optim.adam(f, params, optim_config)
  loss[1] = loss[1] / math.log(2) -- using the bits-per-character (BPC) metric
  table.insert(train_loss_history, loss[1])
  if opt.print_every > 0 and i % opt.print_every == 0 then
    local float_epoch = i / num_train
    local msg = 'Epoch %.2f / %d, i = %d / %d, loss = %f'
    local args = {msg, float_epoch, opt.max_epochs, i, num_iterations, loss[1]}
    print(string.format(unpack(args)))
  end

  -- Maybe save a checkpoint
  local check_every = opt.checkpoint_every
  if (check_every > 0 and i % check_every == 0) or i == num_iterations then
  
    -- save current model to disk
    local tmpFileName = string.format('%stmp.t7', filenamehd)
    paths.mkdir(paths.dirname(tmpFileName))
    model:float()
    torch.save(tmpFileName, model)
    model:type(dtype)
    
    -- Evaluate loss on the validation set
    model:evaluate() -- switch to validation mode
    model:resetStates() -- clear h0
    local num_val = loader.split_sizes['val']
    local val_loss = 0
    for j = 1, num_val do
      local xv, yv = loader:nextBatch('val')
      xv = xv:type(dtype)
      yv = yv:type(dtype):view(N * T)
      local scores = model:forward(xv):view(N * T, -1)
      if j > 1 then
        val_loss = val_loss + crit:forward(scores, yv) / math.log(2)
      end
    end
    val_loss = val_loss / (num_val - 1)
    model:resetStates() -- clear h0
    model:training() -- switch back to training
    print('val_loss = ', val_loss)
    table.insert(val_loss_history, val_loss)
    table.insert(val_loss_history_it, i)

    -- First save a JSON checkpoint, excluding the model
    local checkpoint = {
      opt = opt,
      train_loss_history = train_loss_history,
      val_loss_history = val_loss_history,
      val_loss_history_it = val_loss_history_it,
      forward_backward_times = forward_backward_times,
      memory_usage = memory_usage,
    }
    local filename = string.format('%s%d.json', filenamehd, i)
    paths.mkdir(paths.dirname(filename))
    utils.write_json(filename, checkpoint)

    -- Save a torch checkpoint with the model
    model:clearState() -- clear the intermiate states for saving
    local filename = string.format('%s%d.t7', filenamehd, i)
    paths.mkdir(paths.dirname(filename))
    model:float() -- cast the model to float before saving so it can be used on CPU
    checkpoint.model = model
    torch.save(filename, checkpoint)
    model = torch.load(tmpFileName) -- recover the model for training from disk
    model:type(dtype) -- convert back the type
    params, grad_params = model:getParameters()
    collectgarbage()
  end
  
  -- When at the end of an epoch
  if i % num_train == 0 then
    model:resetStates() -- Reset hidden states
    -- Maybe decay learning rate
    local epoch = math.ceil(i / num_train)
    if epoch % opt.lr_decay_every == 0 then
      local old_lr = optim_config.learningRate
      optim_config = {learningRate = old_lr * opt.lr_decay_factor}
    end
  end
end
