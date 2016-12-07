local nn = require 'nn'


local Convolution = nn.SpatialConvolution
local Max = nn.SpatialMaxPooling
local View = nn.View
local Linear = nn.Linear

local model  = nn.Sequential()

model:add(Convolution(3,100,7,7))
model:add(nn.ReLU(true))
model:add(Max(2,2,2,2))
model:add(Convolution(100, 150, 4, 4))
model:add(nn.ReLU(true))
model:add(Max(2,2,2,2))
model:add(Convolution(150, 250, 4, 4))
model:add(nn.ReLU(true))
model:add(Max(2,2,2,2))
model:add(View(250*3*3))
model:add(nn.Dropout(0.5))
model:add(Linear(250*3*3, 300))
model:add(nn.BatchNormalization(300))
model:add(nn.ReLU(true))
model:add(Linear(300, 43))
model:add(nn.SoftMax())

return model
