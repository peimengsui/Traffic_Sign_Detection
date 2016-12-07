require 'torch'
require 'optim'
require 'os'
require 'optim'
require 'xlua'
local dbg = require("debugger")
require 'cunn'
require 'cutorch'
require 'cudnn' -- faster convolutions
require 'image'
require 'nnx' 
--[[
--  Hint:  Plot as much as you can.  
--  Look into torch wiki for packages that can help you plot.
--]]

local tnt = require 'torchnet'
local image = require 'image'
local optParser = require 'opts'
local opt = optParser.parse(arg)

--change width height default
local WIDTH, HEIGHT = 48, 48
local DATA_PATH = (opt.data ~= '' and opt.data or './data/')
local trainLogger = optim.Logger('train.log')
local valLogger = optim.Logger('val.log')
torch.setdefaulttensortype('torch.DoubleTensor')

-- torch.setnumthreads(1)
torch.manualSeed(opt.manualSeed)
-- cutorch.manualSeedAll(opt.manualSeed)
function crop(img)
    return image.crop(img,x1,y1,x2,y2)
end 
function resize(img)
    return image.scale(img, WIDTH,HEIGHT,"bilinear")
end
function randomDistortion(img)
    local image = require 'image'
    local res=torch.FloatTensor(3, 48, 48):fill(0)
    --local distImg=torch.FloatTensor(3, 54, 54):fill(0)
    local image = require 'image'
    t = image.translate(img,torch.uniform(-0.1*48,0.1*48),torch.uniform(-0.1*48,0.1*48))
    r = image.rotate(t, torch.uniform(-3.14/36,3.14/36))
    scale = torch.uniform(0.9,1.1)
    sz = torch.floor(scale*48)
    s = image.scale(r, sz, sz)
    res:copy(image.scale(s,48,48,"bilinear"))
    return res
end
-- function conorm(img)
--     local neighborhood = image.gaussian1D(5) -- 5 for face detector training
--     local channels = {'r','g','b'}
--     float = torch.FloatTensor(img:size()):copy(img)
-- -- Define our local normalization operator (It is an actual nn module,
-- -- which could be inserted into a trainable model):
--     local normalization = nn.SpatialContrastiveNormalization(1, neighborhood, 1):float()
-- -- Normalize all channels locally:
--     for c in ipairs(channels) do
--         float[{{c},{},{} }] = normalization:forward(float[{{c},{},{} }])
--     end
--     img = torch.DoubleTensor(float:size()):copy(float)
--     return img
-- end

--[[
-- Hint:  Should we add some more transforms? shifting, scaling?
-- Should all images be of size 32x32?  Are we losing 
-- information by resizing bigger images to a smaller size?
--]]
function transformInput(inp,x1,y1,x2,y2)
    f = tnt.transform.compose{
        [1] = crop,
        [2] = resize,
        --[3] = conorm,
        [3] = randomDistortion
    }
    return f(inp)
end

--dataset idx is t7 file, image load, get the crop window
function getTrainSample(dataset, idx)
    r = dataset[idx]
    classId, track, file = r[9], r[1], r[2]
    width,height = r[3],r[4]
    x1,y1,x2,y2 = r[5],r[6],r[7],r[8]
    file = string.format("%05d/%05d_%05d.ppm", classId, track, file)
    return transformInput(image.load(DATA_PATH .. '/train_images/'..file),x1,y1,x2,y2)
end

function getTrainLabel(dataset, idx)
    return torch.LongTensor{dataset[idx][9] + 1}
end

function getTestSample(dataset, idx)
    r = dataset[idx]
    x1,y1,x2,y2 = r[4],r[5],r[6],r[7]
    file = DATA_PATH .. "/test_images/" .. string.format("%05d.ppm", r[1])
    img = image.load(file)
    return transformInput(image.load(file),x1,y1,x2,y2)
end
if opt.parallel =='true' then
    function getIterator(mode)
        print ('getting iterator')
      return tnt.ParallelDatasetIterator{
        nthread = opt.nThreads,
        init    = function() 
            local tnt = require 'torchnet'
            local torch = require 'torch'
            local xlua = require 'xlua'
            local dbg = require("debugger")
            local image = require 'image'
            local tnt = require 'torchnet'
            local image = require 'image'

        end,
        closure = function()
                local tnt = require 'torchnet'
                local torch = require 'torch'
                local xlua = require 'xlua'
                local dbg = require("debugger")
                local tnt = require 'torchnet'
                local image = require 'image'
                getTrainSample = function (dataset, idx)
                    r = dataset[idx]
                    classId, track, file = r[9], r[1], r[2]
                    width,height = r[3],r[4]
                    x1,y1,x2,y2 = r[5],r[6],r[7],r[8]
                    file = string.format("%05d/%05d_%05d.ppm", classId, track, file)
                    return transformInput(image.load(DATA_PATH .. '/train_images/'..file),x1,y1,x2,y2)
                    end
                getTrainLabel = function (dataset, idx)
                            local torch = require 'torch'
                            return torch.LongTensor{dataset[idx][9] + 1}
                        end
                transformInput = function (inp,x1,y1,x2,y2)
                            local tnt = require 'torchnet'
                            f = tnt.transform.compose{
                                [1] = crop,
                                [2] = resize,
                                --[3] = conorm,
                                [3] = randomDistortion
                            }
                            return f(inp)
                        end
                getTestSample = function (dataset, idx)
                    local image = require 'image'
                    r = dataset[idx]
                    x1,y1,x2,y2 = r[4],r[5],r[6],r[7]
                    file = DATA_PATH .. "/test_images/" .. string.format("%05d.ppm", r[1])
                    img = image.load(file)
                    return transformInput(image.load(file),x1,y1,x2,y2)
                end
                crop = function (img)
                            local image = require 'image'
                            return image.crop(img,x1,y1,x2,y2)
                        end
                resize = function (img)
                            local image = require 'image'
                            return image.scale(img, WIDTH,HEIGHT,"bilinear")
                        end
                -- conorm = function (img)
                --     local neighborhood = image.gaussian1D(5) -- 5 for face detector training
                --     local channels = {'r','g','b'}
                --     float = torch.FloatTensor(img:size()):copy(img)
                -- -- Define our local normalization operator (It is an actual nn module,
                -- -- which could be inserted into a trainable model):
                --     local normalization = nn.SpatialContrastiveNormalization(1, neighborhood, 1):float()
                -- -- Normalize all channels locally:
                --     for c in ipairs(channels) do
                --         float[{{c},{},{} }] = normalization:forward(float[{{c},{},{} }])
                --     end
                --     img = torch.DoubleTensor(float:size()):copy(float)
                --     return img
                -- end
                randomDistortion = function (img)
                            local image = require 'image'
                            local res=torch.FloatTensor(3, 48, 48):fill(0)
                            --local distImg=torch.FloatTensor(3, 54, 54):fill(0)
                            local image = require 'image'
                            local t = image.translate(img,torch.uniform(-0.1*48,0.1*48),torch.uniform(-0.1*48,0.1*48))
                            local r = image.rotate(t, torch.uniform(-3.14/36,3.14/36))
                            local scale = torch.uniform(0.9,1.1)
                            local sz = torch.floor(scale*48)
                            local s = image.scale(r, sz, sz)
                            res:copy(image.scale(s,48,48,"bilinear"))
                            return res
                        end
                local trainData = torch.load(DATA_PATH..'train.t7')
                trainDataset = tnt.SplitDataset{
                    partitions = {train=0.9, val=0.1},
                    initialpartition = 'train',
                    --[[
                    --  Hint:  Use a resampling strategy that keeps the 
                    --  class distribution even during initial training epochs 
                    --  and then slowly converges to the actual distribution 
                    --  in later stages of training.
                    --]]
                    dataset = tnt.ShuffleDataset{
                        --replacement = true,
                        --size = 50000,
                        dataset = tnt.ListDataset{
                            list = torch.range(1, trainData:size(1)):long(),
                            load = function(idx)
                                return {
                                    input =  getTrainSample(trainData, idx),
                                    target = getTrainLabel(trainData, idx)
                                }
                            end
                        }
                    }
                }
                trainDataset:select(mode)
                --dbg()
                return tnt.BatchDataset{
                            batchsize = opt.batchsize,
                            dataset = trainDataset
                }

        end
      }
    end
else 
    function getIterator(dataset)
        --[[
        -- Hint:  Use ParallelIterator for using multiple CPU cores
        --]]
        return tnt.DatasetIterator{
            dataset = tnt.BatchDataset{
                batchsize = opt.batchsize,
                dataset = dataset
            }
        }
    end
end
function getTestIterator(dataset)
        --[[
        -- Hint:  Use ParallelIterator for using multiple CPU cores
        --]]
        return tnt.DatasetIterator{
            dataset = tnt.BatchDataset{
                batchsize = opt.batchsize,
                dataset = dataset
            }
        }
    end




local testData = torch.load(DATA_PATH..'test.t7')
local trainData = torch.load(DATA_PATH..'train.t7')
trainDataset = tnt.SplitDataset{
    partitions = {train=0.9, val=0.1},
    initialpartition = 'train',
    --[[
    --  Hint:  Use a resampling strategy that keeps the 
    --  class distribution even during initial training epochs 
    --  and then slowly converges to the actual distribution 
    --  in later stages of training.
    --]]
    dataset = tnt.ShuffleDataset{
        --replacement = true,
        --size = 50000,
        dataset = tnt.ListDataset{
            list = torch.range(1, trainData:size(1)):long(),
            load = function(idx)
                return {
                    input =  getTrainSample(trainData, idx),
                    target = getTrainLabel(trainData, idx)
                }
            end
        }
    }
}


testDataset = tnt.ListDataset{
    list = torch.range(1, testData:size(1)):long(),
    load = function(idx)
        return {
            input = getTestSample(testData, idx),
            sampleId = torch.LongTensor{testData[idx][1]}
        }
    end
}

--[[
-- Hint:  Use :cuda to convert your model to use GPUs
--]]
local model = require("models/".. opt.model)
if opt.cuda == 'true' then 
    model = model:cuda()
end
local engine = tnt.OptimEngine()
local meter = tnt.AverageValueMeter()
local criterion = nn.CrossEntropyCriterion()
if opt.cuda == 'true' then
    criterion = criterion:cuda()
end
local clerr = tnt.ClassErrorMeter{topk = {1}}
local timer = tnt.TimeMeter()
local batch = 1

-- print(model)

engine.hooks.onStart = function(state)
    meter:reset()
    clerr:reset()
    timer:reset()
    batch = 1
    if state.training then
        mode = 'Train'
    else
        mode = 'Val'
    end
end

--[[
-- Hint:  Use onSample function to convert to 
--        cuda tensor for using GPU
--]]
if opt.cuda == 'true' then 
    local input = torch.CudaTensor()
    local target = torch.CudaTensor()
    engine.hooks.onSample = function(state)
        local image = require 'image'
        input:resize(state.sample.input:size()):copy(state.sample.input)
        state.sample.input = input
        if state.sample.target ~= nil then
            target:resize(state.sample.target:size()):copy(state.sample.target)
            state.sample.target = target
        end
    end
end
engine.hooks.onForwardCriterion = function(state)
    meter:add(state.criterion.output)
    clerr:add(state.network.output, state.sample.target)
    --dbg()
    trainLogger:add{['% avg error (train set)'] = clerr:value{k=1}}
    if opt.verbose == 'true' then
        print(string.format('avg. loss: %2.4f; avg. error: %2.4f',
            meter:value(), clerr:value{k = 1}))
    --     print(string.format("%s Batch: %d/%d; avg. loss: %2.4f; avg. error: %2.4f",
    --             mode, batch, state.iterator.dataset:size(), meter:value(), clerr:value{k = 1}))
    else
        xlua.progress(batch, 352)
    end
    batch = batch + 1 -- batch increment has to happen here to work for train, val and test.
    timer:incUnit()
end

engine.hooks.onEnd = function(state)
    valLogger:add{['% avg error (val set)'] = clerr:value{k=1}}
    print(string.format("%s: avg. loss: %2.4f; avg. error: %2.4f, time: %2.4f",
    mode, meter:value(), clerr:value{k = 1}, timer:value()))
     torch.save('kaggle_model1.t7',state.network:clearState(),'ascii')
end

local epoch = 1

while epoch <= opt.nEpochs do
    --trainDataset:select('train')
    if opt.parallel=='true' then
        engine:train{
            network = model,
            criterion = criterion,
            iterator = getIterator('train'),
            optimMethod = optim.sgd,
            maxepoch = 1,
            config = {
                learningRate = opt.LR,
                --learningRateDecay = 5.0e-6,
                momentum = opt.momentum,
                weightDecay = opt.weightDecay
                --learningRateDecay = opt.weightDecay
            }
        }
        --trainDataset:select('val')
        engine:test{
            network = model,
            criterion = criterion,
            iterator = getIterator('val')
        }
    else 
        trainDataset:select('train')
        engine:train{
            network = model,
            criterion = criterion,
            iterator = getIterator(trainDataset),
            optimMethod = optim.sgd,
            maxepoch = 1,
            config = {
                learningRate = opt.LR,
                --learningRateDecay = 5.0e-6,
                momentum = opt.momentum,
                weightDecay = opt.weightDecay
                --learningRateDecay = opt.weightDecay
            }
        }
        trainDataset:select('val')
        engine:test{
            network = model,
            criterion = criterion,
            iterator = getIterator(trainDataset)
        }
    end
    trainLogger:style{['% avg error (train set)'] = '-'}
    valLogger:style{['% avg error (val set)'] = '-'}
    trainLogger:plot()
    valLogger:plot()
    print('Done with Epoch '..tostring(epoch))
    epoch = epoch + 1
end

local submission = assert(io.open(opt.logDir .. "/submissionensemble1.csv", "w"))
submission:write("Filename,ClassId\n")
batch = 1

--[[
--  This piece of code creates the submission
--  file that has to be uploaded in kaggle.
--]]
engine.hooks.onForward = function(state)
    local fileNames  = state.sample.sampleId
    local _, pred = state.network.output:max(2)
    pred = pred - 1
    for i = 1, pred:size(1) do
        submission:write(string.format("%05d,%d\n", fileNames[i][1], pred[i][1]))
    end
    xlua.progress(batch, state.iterator.dataset:size())
    batch = batch + 1
end

engine.hooks.onEnd = function(state)
    submission:close()
end

engine:test{
    network = model,
    iterator = getTestIterator(testDataset)
}

print("The End!")