module Main where

import Prelude hiding (head)

import NNets
import System.Random
import System.IO
import Data.List.NonEmpty (head)
import Data.Array.Unboxed
import Data.Array.Base
import Control.DeepSeq (force, NFData, rnf)
import Control.Exception (evaluate)
import Control.Concurrent (threadDelay)
import Control.Monad (when)

import Test.Tasty.Bench (bench, bgroup, defaultMain, env, nf, whnf, nfIO, whnfIO)

-- we need these instances here because for the benchmarks to work
instance NFData Vector where
    rnf arr = arr `seq` ()

instance NFData Matrix where
    rnf arr = arr `seq` ()

-- Creates a vector of a given length initialised with random numbers from -0.5 to 0.5
-- useful for initialising biases. The fact that this uses listArray may contribute to
-- some performance issues but it isn't too major since it's only during initialisation,
-- which occurs once. 
randBias :: Int -> IO Vector
randBias len = 
    return $ toVector $ replicate len 0.0

-- Creates a matrix of a given size initialised with random numbers from -0.5 to 0.5
-- useful for initialising weights. The fact that this uses listArray may contribute to
-- some performance issues but it isn't too major since it's only during initialisation,
-- which occurs once. 
-- We use Xavier weight initialisation.
randWeight :: Int -> Int -> IO Matrix
randWeight n m =
    do  gen <- newStdGen
        return $ go n gen []
    where
        go 0 _ xs = toMatrix xs
        go n g xs = let (g1, g2) = splitGen g
                        xavierRange = (1.0 / sqrt (fromIntegral m))
                        row = take m $ uniformRs (-xavierRange, xavierRange) g1
                    in  go (n-1) g2 (row : xs) -- right associates.


mse :: [Vector] -> [Vector] -> Double
mse expected actual = sum (map (vfoldl' (+) 0.0 . vmap (\x -> x*x)) $ zipWith (^-^) expected actual) / fromIntegral (length expected)
-------------------------------- Training -------------------------------------
type FullyConnectedNetwork = (InputLayer :+: DenseLayer) 

miniBatchSize :: Int = 800
numberOfIterations :: Int = 100

sineTestInputs :: Vector
sineTestInputs = let xs = [-pi, -pi + 0.01 .. pi] in toVector xs

-- need a better API for the vector type
sineTestOutputs :: Vector
sineTestOutputs = vmap (\x -> (sin x + 1) / 2) sineTestInputs

fcNetworkPair :: IO (Free FullyConnectedNetwork a)
fcNetworkPair = do
    w1 <- randWeight 12 1
    b1 <- randBias 12 -- these were swapped, cause of the old bug. 
    w2 <- randWeight 1 12
    b2 <- randBias 1  
    return $ do denseLayer w2 b2 -- the layers also should have been this way around
                denseLayer w1 b1
                inputLayer

generateEpoch :: Int -> IO [(Vector, Vector)]
generateEpoch n_samples = do
    g <- newStdGen
    let initialInputs = take n_samples (uniformRs (-pi, pi) g) -- integrates better with list fusion apparently
        desiredOutputs = map (\x -> (sin x + 1) / 2) initialInputs -- keeps between 0 and 1
        trainingData = zipWith (\x y -> (toVector [x], toVector [y])) initialInputs desiredOutputs
    return trainingData

runEpoch network = do
    trainingData <- generateEpoch miniBatchSize
    let nn = trainMany trainingData network
    return nn

runAllEpochs numEpochs = do
    network0 <- fcNetworkPair
    go numEpochs network0
  where
    go 0 net = pure net
    go n net = do
      net' <- runEpoch net
      when (n `mod` 10 == 0) $ printNetwork net'
      go (n - 1) net'

trainSineForBenchOutput = do
    (nn :: Free FullyConnectedNetwork a) <- runAllEpochs numberOfIterations
    let testOutputs = vmap (vhead . head . forwardProp nn . toVector . (:[])) sineTestInputs
    print testOutputs
    return testOutputs

main = do
    testOutputs <- trainSineForBenchOutput
    -- print (concatMap sum testOutputs)
    -- print testOutputs
    -- !_ <- evaluate testOutputs'
    let !testOutputs' = force testOutputs
    return ()

-- to test:
-- cabal run ann-warray-bench -- +RTS -hy -l-agu
-- eventlog2html ann-warray-bench.eventlog
-- open -a 'Brave Browser' ann-warray-bench.eventlog.html