import 'torch'

-- source
local d0 = 2
local m0 = 700
local v = 65

-- target
local d1 = 3

local para_num = 4 * v * (d0 - 1) * m0 + (d0 * (d0 + 3)) * m0 * m0 + (d0 + 3) * m0

local B = 4 * v * (d1 - 1) + (d1 + 3)
local A = d1 * (d1 + 3)
local C = -para_num

local m1 = (-B + math.sqrt(B * B - 4 * A * C)) / (2 * A)
print(para_num, m1)

--print(4 * v * (d0 - 1) * m0)
--print((d0 * (d0 + 3)) * m0 * m0 + (d0 + 3) * m0)
--
--print(4 * v * (d1 - 1) * m1)
--print((d1 * (d1 + 3)) * m1 * m1 + (d1 + 3) * m1)

