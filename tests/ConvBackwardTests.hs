{-
Full disclosure: I could NOT be bothered to run through manual test cases on paper to verify, 
so I got claude to generate these test cases. It said that it verified the numbers by running through
JAX; I'm trusting it for now because my numbers match it.

There is a better way of testing:
https://medium.com/@stergiadis777/convolutional-neural-networks-implementation-and-testing-9ac0a57ac5f1
Basically you just check the end result by calculating ∂L/∂W yourself for each weight, then compare
with the propagations' result, because most of the work is intermediate and we only care about the final vals.  
If something ends up going wrong with learning MNIST then I can do this to check again. 
-}

module Main where

import NNets

type ConvNetwork = (InputLayer :+: DenseLayer :+: ConvLayer :+: PoolLayer :+: FlattenLayer)

denseW = toMatrix [[-0.8,-0.07,0.2,0.17]]   -- same for all 3 cases
denseB = toVector [0.97]

-- ===================== Case A: 2x2 conv filter, non-overlapping pool =====================
networkA :: Free ConvNetwork a
networkA = do
    denseLayer denseW denseB
    flattenLayer 2 2
    poolLayer 2 2
    convLayer (toMatrix [[1,-1],[0,1]]) (-0.5)
    inputLayer

inputA :: Tensor
inputA = TMat $ toMatrix [[1,2,0,1,3],[0,1,2,1,0],[2,0,1,0,2],[1,1,0,2,1],[0,2,1,0,1]]

desiredOutputA :: Vector
desiredOutputA = toVector [1.0]

-- expected pool_out (-> flatten input): [3.5,2.5,2.5,2.5]
-- expected dense output al = sigmoid(-1.08) = 0.25351
-- --- after trainCNN: ---
-- expected denseW': [[-0.60223,0.07127,0.34127,0.31127]]   expected denseB': [1.02651]
-- expected convFilter': [[0.93784,-1.0],[-0.03390,0.93219]]  expected convBias': -0.52825

-- ===================== Case B: 3x3 conv filter, pool ties =====================
networkB :: Free ConvNetwork a
networkB = do
    denseLayer denseW denseB
    flattenLayer 2 2
    poolLayer 2 2
    convLayer (toMatrix [[0,0,1],[0,1,0],[1,0,0]]) (-1.0)
    inputLayer

inputB :: Tensor  -- your original diamond input
inputB = TMat $ toMatrix [[0,0,1,1,0,0],[0,1,0,0,1,0],[1,0,0,0,0,1],[1,0,0,0,0,1],[0,1,0,0,1,0],[0,0,1,1,0,0]]

desiredOutputB :: Vector
desiredOutputB = toVector [1.0]

-- expected pool_out (both 0-windows are 4-way ties -> flatten input): [2,0,0,2]
-- expected dense output al = sigmoid(-0.29) = 0.42800
-- --- after trainCNN: ---
-- expected denseW': [[-0.68797,-0.07,0.2,0.28203]]   expected denseB': [1.02601]
-- expected convFilter': [[0,0,0.96471],[0,0.96471,0],[0.96471,0,0]]  expected convBias': -1.03529

-- ===================== Case C: 4x4 conv filter, OVERLAPPING pool (stride 1 < spatExt 2) =====================
networkC :: Free ConvNetwork a
networkC = do
    denseLayer denseW denseB
    flattenLayer 2 2
    poolLayer 2 1   -- overlapping: centre conv-out cell feeds all 4 pool windows
    convLayer (toMatrix [[1,0,0,-1],[0,1,0,0],[0,0,1,0],[-1,0,0,1]]) 0.0
    inputLayer

inputC :: Tensor
inputC = TMat $ toMatrix [[1,0,2,1,0,1],[0,1,0,2,1,0],[2,0,1,0,2,1],[1,2,0,1,0,2],[0,1,2,0,1,0],[1,0,1,2,0,1]]

desiredOutputC :: Vector
desiredOutputC = toVector [1.0]

-- expected pool_out (-> flatten input): [2,7,7,2]
-- expected dense output al = sigmoid(0.62) = 0.65022
-- --- after trainCNN: ---
-- expected denseW': [[-0.73636,0.15275,0.42275,0.23364]]   expected denseB': [1.00182]
-- expected convFilter':
--   [[0.98823,-0.00223,-0.03373,-1.02228],[0.00636,0.98823,-0.00223,-0.03373],
--    [-0.04232,0.00636,0.98823,-0.00223],[-1.01368,-0.04232,0.00636,0.98823]]
-- expected convBias': -0.01591

main :: IO ()
main = do
    putStrLn "=== Case A ==="
    printNetwork networkA
    let networkA' = trainCNN (inputA, desiredOutputA) networkA
    printNetwork networkA'

    putStrLn "=== Case B ==="
    printNetwork networkB
    let networkB' = trainCNN (inputB, desiredOutputB) networkB
    printNetwork networkB'

    putStrLn "=== Case C ==="
    printNetwork networkC
    let networkC' = trainCNN (inputC, desiredOutputC) networkC
    printNetwork networkC'