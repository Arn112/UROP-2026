-- contains random initialisation functions for weights and biases. 
module NNets.Training.Initialise where 
 
import System.Random
import NNets.Common

-- Creates a vector of a given length initialised with random numbers from -0.5 to 0.5
-- useful for initialising biases. The fact that this uses listArray may contribute to
-- some performance issues but it isn't too major since it's only during initialisation,
-- which occurs once. 
randBias :: Int -> IO Vector
randBias len = 
    return $ toVector $ replicate len 0.0

-- Creates a matrix of a given size initialised with random numbers from -0.5 to 0.5
-- useful for initialising weights. Performance here isn't a big deal.
-- We use Xavier weight initialisation.
randWeight :: Int -> Int -> IO Matrix
randWeight n m =
    do  gen <- newStdGen
        return $ go n gen []
    where
        go 0 _ !xs  =   toMatrix xs
        go !n g !xs =   let (g1, g2) = splitGen g
                            xavierRange = (1.0 / sqrt (fromIntegral m))
                            row = take m $ uniformRs (-xavierRange, xavierRange) g1
                        in  go (n-1) g2 (row : xs) -- right associates.
