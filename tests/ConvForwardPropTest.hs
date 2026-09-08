-- testing forward prop for conv before implementing the backprop stuff.
module ConvForwardPropTest where

import NNets

type ConvNetwork =  (InputLayer :+: DenseLayer :+: ConvLayer :+: PoolLayer :+: FlattenLayer)

-- convNetwork :: Free ConvNetwork a
-- convNetwork = do
--     denseLayer (toMatrix [[-0.8,-0.07,0.2,0.17]]) (toVector [0.97]) -- the previous flattenLayer is basically an inputLayer from this thing's pov
--     flattenLayer 2 2
--     poolLayer 2 2
--     convLayer (toMatrix [[0.0,0.0,1.0],[0.0,1.0,0.0],[1.0,0.0,0.0]]) (-2.0)
--     inputLayer

-- convTestInput :: Tensor = TMat $ toMatrix
--                                 (map (map fromIntegral) [[0,0,1,1,0,0],
--                                                         [0,1,0,0,1,0],
--                                                         [1,0,0,0,0,1],
--                                                         [1,0,0,0,0,1],
--                                                         [0,1,0,0,1,0],
--                                                         [0,0,1,1,0,0]])
-- main :: IO ()
-- main = return ()

-- Ground-truth fixtures for ConvLayer + PoolLayer forward/backward. Computed and
-- cross-checked independently (finite-difference gradient check), so treat these
-- numbers as the oracle. Seed = dL/d(pool output), standing in for whatever your
-- flatten+dense+loss backward would normally hand down -- lets you unit-test
-- conv+pool in isolation before wiring in the rest of the network.
--
-- conv backward (dLdFilter/dLdBias/dLdInput) isn't written yet on your end, so
-- those are given as target values to check once you have it: dLdBias = sum of
-- (dLdConvOut masked by ReLU'), dLdFilter = correlate(input, masked grad),
-- dLdInput = "full" convolution of the masked grad with the filter (pad by
-- filterSize-1, correlate with rotatePi filt) -- rotatePi/padMatrixN look built
-- for exactly this.

-- ===================== Case A: 2x2 filter, non-overlapping pool =====================
inputA  = toMatrix [[1,2,0,1,3],[0,1,2,1,0],[2,0,1,0,2],[1,1,0,2,1],[0,2,1,0,1]]
filterA = toMatrix [[1,-1],[0,1]]
biasA   = -0.5
seedA   = toMatrix [[1,-2],[0.5,1]]      -- dL/d(pool output)

convNetA :: Free ConvNetwork a
convNetA = do
    flattenLayer 2 2
    poolLayer 2 2
    convLayer filterA biasA
    inputLayer

-- expected conv_out:  [[0,3.5,0,0],[0,0,0.5,2.5],[2.5,0,2.5,0],[1.5,1.5,0,1.5]]
-- expected pool_out (2x2, stride 2): [[3.5,2.5],[2.5,2.5]]
-- expected dLdConvOut = reversePool convOutA poolOutA seedA 2 2:
--   [[0,1,0,0],[0,0,0,-2],[0.5,0,1,0],[0,0,0,0]]
-- expected dLdFilter: [[2,0],[1.5,0.5]]        expected dLdBias: 0.5
-- expected dLdInput:
--   [[0,1,-1,0,0],[0,0,1,-2,2],[0.5,-0.5,1,-1,-2],[0,0.5,0,1,0],[0,0,0,0,0]]

-- ===================== Case B: 3x3 filter, pool ties =====================
-- (your original diamond input) -- both zero-windows are 4-way ties: tests equal-split
inputB  = toMatrix [[0,0,1,1,0,0],[0,1,0,0,1,0],[1,0,0,0,0,1],[1,0,0,0,0,1],[0,1,0,0,1,0],[0,0,1,1,0,0]]
filterB = toMatrix [[0,0,1],[0,1,0],[1,0,0]]
biasB   = -1.0
seedB   = toMatrix [[2,-1],[1,0.5]]

-- expected conv_out:  [[2,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,2]]
-- expected pool_out: [[2,0],[0,2]]
-- expected dLdConvOut = reversePool convOutB poolOutB seedB 2 2:
--   [[2,0,-0.25,-0.25],[0,0,-0.25,-0.25],[0.25,0.25,0,0],[0.25,0.25,0,0.5]]
-- expected dLdFilter: [[0,0,2.5],[0,2.5,0],[2.5,0,0]]     expected dLdBias: 2.5
-- expected dLdInput:
--   [[0,0,2,0,0,0],[0,2,0,0,0,0],[2,0,0,0,0,0],[0,0,0,0,0,0.5],[0,0,0,0,0.5,0],[0,0,0,0.5,0,0]]
-- note: this case sits exactly on the ReLU zero boundary at several positions (by
-- design, to force ties), so a naive finite-diff check is unstable right at this
-- point -- these values were confirmed by hand instead, not just numerically.

-- ===================== Case C: 4x4 filter, OVERLAPPING pool (stride 1 < spatExt 2) =====================
inputC  = toMatrix [[1,0,2,1,0,1],[0,1,0,2,1,0],[2,0,1,0,2,1],[1,2,0,1,0,2],[0,1,2,0,1,0],[1,0,1,2,0,1]]
filterC = toMatrix [[1,0,0,-1],[0,1,0,0],[0,0,1,0],[-1,0,0,1]]
biasC   = 0.0
seedC   = toMatrix [[1,1],[-1,2]]

-- expected conv_out: [[2,0,7],[0,2,0],[7,0,2]]
-- expected pool_out (2x2, stride 1): [[2,7],[7,2]]
-- expected dLdConvOut = reversePool convOutC poolOutC seedC 2 1:
--   [[0.5,0,1],[0,1.5,0],[-1,0,1]]   -- centre cell is in all 4 windows: watch it accumulate
-- expected dLdFilter: [[3,1,5,4],[-1,3,1,5],[7,-1,3,1],[2,7,-1,3]]   expected dLdBias: 3.0
-- expected dLdInput:
--   [[0.5,0,1,-0.5,0,-1],[0,2,0,1,-1.5,0],[-1,0,3,1,1,-1],
--    [-0.5,-1,-1,3,0,1],[0,-1.5,-1,0,2.5,0],[1,0,-1,-1,0,1]]

main :: IO ()
main = do
    let convOutA = convolution inputA filterA biasA
        poolOutA = pool convOutA 2 2
        dLdConvOutA = reversePool convOutA poolOutA seedA 2 2
    print convOutA >> print poolOutA >> print dLdConvOutA

    let convOutB = convolution inputB filterB biasB
        poolOutB = pool convOutB 2 2
        dLdConvOutB = reversePool convOutB poolOutB seedB 2 2
    print convOutB >> print poolOutB >> print dLdConvOutB

    let convOutC = convolution inputC filterC biasC
        poolOutC = pool convOutC 2 1
        dLdConvOutC = reversePool convOutC poolOutC seedC 2 1
    print convOutC >> print poolOutC >> print dLdConvOutC