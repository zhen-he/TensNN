import 'torch'

-- source
local d0 = 2
local m0 = 1000
local v = 205

-- target
local d1 = 3

local v = 65
local para_num = 4 * v * m0 + d0 * (d0 + 3) * m0 * m0
local m1 = (-4 * v + math.sqrt(16 * v * v + 4 * d1 * (d1 + 3) * para_num)) / (2 * d1 * (d1 + 3))
print(para_num, m1)

