import 'torch'

-- source
local d0 = 3
local m0 = 256
local v = 65

-- target
local d1 = 2

local para_num = 4 * v * d0 * m0 + (d0 * (d0 + 3)) * m0 * m0
local m1 = (-4 * v * d1 + math.sqrt(16 * v * v * d1 * d1 + 4 * d1 * (d1 + 3) * para_num)) 
/ (2 * d1 * (d1 + 3))
print(para_num, m1)

