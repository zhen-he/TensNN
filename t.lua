

a=torch.Tensor(1):fill(1)
local b=a
local c=b[1]
b[1]=c+1
print(a, b, c)

