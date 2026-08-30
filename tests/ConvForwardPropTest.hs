-- testing forward prop for conv before implementing the backprop stuff.
module Main where

import NNets

type ConvNetwork =  (InputLayer :+: DenseLayer :+: ConvLayer :+: PoolLayer :+: FlattenLayer)

convNetwork :: Free ConvNetwork a
convNetwork = do
    denseLayer (toMatrix [[-0.8,-0.07,0.2,0.17]]) (toVector [0.97]) -- the previous flattenLayer is basically an inputLayer from this thing's pov
    flattenLayer 2 2
    poolLayer 2 2
    convLayer (toMatrix [[0.0,0.0,1.0],[0.0,1.0,0.0],[1.0,0.0,0.0]]) (-2.0)
    inputLayer

convTestInput :: Tensor = TMat $ toMatrix
                                (map (map fromIntegral) [[0,0,1,1,0,0],
                                                        [0,1,0,0,1,0],
                                                        [1,0,0,0,0,1],
                                                        [1,0,0,0,0,1],
                                                        [0,1,0,0,1,0],
                                                        [0,0,1,1,0,0]])
main :: IO ()
main = return ()