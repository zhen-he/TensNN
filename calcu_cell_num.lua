import 'torch'

-- source
local d0 = 3
local m0 = 700
local v = 205

-- target
local d1 = 3

local para_num = 4 * v * (d0 - 1) * m0 + (d0 * (d0 + 3)) * m0 * m0 + (d0 + 3) * m0
--local shared_para_num = 4 * v * (d0 - 1) * m0 + (1 * (d0 + 3)) * m0 * m0 + (d0 + 3) * m0

local B = 4 * v * (d1 - 1) + (d1 + 3)
local A = d1 * (d1 + 3)
local C = -para_num
local m1_multi = (-B + math.sqrt(B * B - 4 * A * C)) / (2 * A)

B = 4 * v * (d1 - 1) + (d1 + 3)
A = 2 * (d1 + 3)
C = -para_num
local m1_share = (-B + math.sqrt(B * B - 4 * A * C)) / (2 * A)

d1 = 2
B = 3 * v * (d1 - 1) + (d1 + 2)
A = d1 * (d1 + 2)
C = -para_num
local m1_stack = (-B + math.sqrt(B * B - 4 * A * C)) / (2 * A)



print('para_num: ' .. para_num)
--print('shared_para_num: ' .. shared_para_num)
print('m1_multi: ' .. m1_multi)
print('m1_share: ' .. m1_share)
print('m1_stack: ' .. m1_stack)


--print(4 * v * (d0 - 1) * m0)
--print((d0 * (d0 + 3)) * m0 * m0 + (d0 + 3) * m0)
--
--print(4 * v * (d1 - 1) * m1)
--print((d1 * (d1 + 3)) * m1 * m1 + (d1 + 3) * m1)

