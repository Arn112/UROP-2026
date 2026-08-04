{-#LANGUAGE PatternSynonyms, ViewPatterns#-}

module CnnTest where

import CnnCode
import Helpers

type ConvNetwork =  (InputLayer :+: DenseLayer :+: ConvLayer :+: PoolLayer :+: FlattenLayer)

convNetwork :: Free ConvNetwork a
convNetwork = do
    denseLayer [[-0.8,-0.07,0.2,0.17]] [0.97] -- the previous flattenLayer is basically an inputLayer from this thing's pov
    flattenLayer 2 2
    poolLayer 2 2
    convLayer [[0.0,0.0,1.0],[0.0,1.0,0.0],[1.0,0.0,0.0]] (-2.0)
    inputLayer

instance Show Tensor where
    show (Vec v) = show v
    show (Mat w) = show w

convTestInput :: Tensor = Mat (map (map fromIntegral) [[0,0,1,1,0,0],[0,1,0,0,1,0],[1,0,0,0,0,1],[1,0,0,0,0,1],[0,1,0,0,1,0],[0,0,1,1,0,0]])


convNetworkAfterOneBackprop :: Free ConvNetwork a
convNetworkAfterOneBackprop = do
    denseLayer [[-0.6989949,-0.07,0.2,0.2710051]] [1.0710051] -- the previous flattenLayer is basically an inputLayer from this thing's pov
    flattenLayer 2 2
    poolLayer 2 2
    convLayer [[0.0,0.0,1.0],[0.0,1.0,0.0],[1.0,0.0,0.0]] (-1.7979898)
    inputLayer

-- debugging would be easier if we had a function to traverse through network
-- and directly view the weights and biases at each layer. Going to make such
-- a function.
pattern InputLayer' <- (prj -> Just InputLayer)
pattern DenseLayer' ws bs nn' <- (prj -> Just (DenseLayer ws bs nn'))
pattern PoolLayer' sext sde nn' <- (prj -> Just (PoolLayer sext sde nn'))
pattern ConvLayer' filter bias nn' <- (prj -> Just (ConvLayer filter bias nn'))
pattern FlattenLayer' h w nn' <- (prj -> Just (FlattenLayer h w nn'))
-- we need these patterns to avoid nesting cases. this is because prj
-- seems to only be able to resolve the :<: functors (f, g, etc.) when we're 
-- once (i.e. after the first case it thinks it's got it figured out). So 
-- instead each pattern does it separately. 

printNetworkCNN :: Free ConvNetwork a -> IO ()
printNetworkCNN (Op cnn) = do
    case cnn of
        InputLayer' -> do 
            putStrLn "inputlayer"                                           
        DenseLayer' ws bs nn' -> do 
            putStrLn "dense layer: "
            putStrLn $ "ws: " ++ show ws
            putStrLn $ "bs: " ++ show bs
            putStrLn "--------------"
            printNetworkCNN nn'
        PoolLayer' sext sde nn' -> do
            putStrLn "max pool layer: "
            putStrLn $ "spatial extent: " ++ show sext
            putStrLn $ "stride: " ++ show sde
            putStrLn "--------------"
            printNetworkCNN nn'
        ConvLayer' filter bias nn' -> do
            putStrLn "convolution layer: "
            putStrLn $ "filter: " ++ show filter
            putStrLn $ "bias: " ++ show bias
            putStrLn "--------------"
            printNetworkCNN nn'
        FlattenLayer' h w nn' -> do
            putStrLn "flatten layer: "
            putStrLn $ "height: " ++ show h ++ ", width: " ++ show w
            putStrLn "--------------"
            printNetworkCNN nn'



