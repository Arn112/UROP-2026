import CnnCode
import BothFolds
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


