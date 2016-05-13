import 'torch'

local N = 100
local inputShape = {100}
local tensShape = {2,2}
local H = 846
local V = 205

local hiddenShape = {} -- the shape of the skewed block
local l = 0
for _, v in ipairs(inputShape) do
  table.insert(hiddenShape, v)
  l = l + v 
end
for _, v in ipairs(tensShape) do
  table.insert(hiddenShape, v)
  l = l + v
end
local D = #hiddenShape
local L = l - D + 2
hiddenShape[1] = L
local G = D + 3

local states = N
for i, v in ipairs(hiddenShape) do
  states = states * v
  
end
states = states * 2 * D * H * 8 / (1024 * 1024)-- MB, double type takes 8 Bytes

local grads = states / hiddenShape[D] * (hiddenShape[D] + 1)

local gates = states / (2 * D) * G

local masks = states / (2 * D) * 2 -- 2 masks

local buffs = states / (2 * D * L) * (
1 * 3 + 
D * 2 + 
(D + 1) * 1 + 
(2 * D) * 2 +
G * 3)

local paras = (4 * V * H + D * G * H * H) * 8 / (1024 * 1024)

local mem = (states + grads + gates + masks + buffs + paras) -- MB

print('\ntotal: ' .. mem)
print('states: ' .. states)
print('grads: ' .. grads)
print('gates: ' .. gates)
print('masks: ' .. masks)
print('buffs: ' .. buffs)
print('paras: ' .. paras)




