require 'torch'
require 'nn'

require 'TensLM'

isBN = 1
local tests = torch.TestSuite()
local tester = torch.Tester()


local function check_dims(x, dims)
  tester:assert(x:dim() == #dims)
  for i, d in ipairs(dims) do
    tester:assert(x:size(i) == d)
  end
end

-- training test
-- Just a smoke test to make sure model can run forward / backward
function tests.simpleTest()
  local inputShape = {3}
  local tensShape = {2,2,2}
  local nodeSize = 5
  local batchSize = 2
  local vocabSize = 6

  local N, T, H, V = batchSize, inputShape[1], nodeSize, vocabSize
  local idx_to_token = {[1]='a', [2]='b', [3]='c', [4]='d', [5]='e', [6]='f'}
  local LM = nn.TensLM{
    idx_to_token = idx_to_token,
    rnn_size = H,
    tensShape = tensShape,
    batchnorm = isBN
  }
  local crit = nn.CrossEntropyCriterion()
  local params, grad_params = LM:getParameters()

  local x = torch.Tensor(N, T):random(V)
  local y = torch.Tensor(N, T):random(V)
  local scores = LM:forward(x)
  check_dims(scores, {N, T, V})
  local scores_view = scores:view(N * T, V)
  local y_view = y:view(N * T)
  local loss = crit:forward(scores_view, y_view)
  local dscores = crit:backward(scores_view, y_view):view(N, T, V)
  LM:backward(x, dscores)
end

-- sampling test
function tests.sampleTest()
  local tensShape = {2,2,2}
  local nodeSize = 5

  local H = nodeSize
  local idx_to_token = {[1]='a', [2]='b', [3]='c', [4]='d', [5]='e', [6]='f'}
  local LM = nn.TensLM{
    idx_to_token = idx_to_token,
    rnn_size = H,
    tensShape = tensShape,
    batchnorm = isBN
  }
  
  local TT = 100
  local start_text = 'bad'
  local sampled = LM:sample{start_text=start_text, length=TT}
  tester:assert(torch.type(sampled) == 'string')
  tester:assert(string.len(sampled) == TT)
end

-- encode/decode test
function tests.encodeDecodeTest()
  local tensShape = {2,2,2}
  local nodeSize = 5

  local H = nodeSize
  local idx_to_token = {
    [1]='a', [2]='b', [3]='c', [4]='d',
    [5]='e', [6]='f', [7]='g', [8]=' ',
  }
  local LM = nn.TensLM{
    idx_to_token=idx_to_token,
    rnn_size = H,
    tensShape = tensShape,
    batchnorm = isBN
  }

  local s = 'a bad feed'
  local encoded = LM:encode_string(s)
  local expected_encoded = torch.LongTensor{1, 8, 2, 1, 4, 8, 6, 5, 5, 4}
  tester:assert(torch.all(torch.eq(encoded, expected_encoded)))

  local s2 = LM:decode_string(encoded)
  tester:assert(s == s2)
end


tester:add(tests)
tester:run()

