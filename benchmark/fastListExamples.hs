-- Contains a test neural network to train sine on, and the required functions to interface with
-- the network training functions. These should and will be refactored to another nicer API later.
-- It's just for testing for now.

{-# LANGUAGE BangPatterns #-}

module Main where

import Prelude hiding (head)

import NNets
import System.Random
import System.IO
import Data.List.NonEmpty (head)
import Data.List (singleton)
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

-- mean squared error. useful for outputting loss over iterations. 
mse :: [Vector] -> [Vector] -> Double
mse expected actual = sum (map (vfoldl' (+) 0.0 . vmap (\x -> x*x)) $ zipWith (^-^) expected actual) / fromIntegral (length expected)
-------------------------------- Training -------------------------------------
type FullyConnectedNetwork = (InputLayer :+: DenseLayer) 

-- these parameters are somewhat misleading, because I mistakenly thought it was mini-batch 
-- learning but it was all just online learning, but I'd already written the functions
-- so just kept this layout. Number of individual samples is their product anyways.  
miniBatchSize :: Int = 800
numberOfIterations :: Int = 200

sineTestInputs :: Vector
sineTestInputs = let xs = [-pi, -pi + 0.01 .. pi] in toVector xs

sineTestOutputs :: Vector
sineTestOutputs = vmap (\x -> (sin x + 1) / 2) sineTestInputs

fcNetworkPair :: IO (Free FullyConnectedNetwork a)
fcNetworkPair = do
    w1 <- randWeight 12 1
    b1 <- randBias 12 
    w2 <- randWeight 1 12
    b2 <- randBias 1  
    return $ do denseLayer w2 b2 
                denseLayer w1 b1
                inputLayer

twoHiddenNetwork :: IO (Free FullyConnectedNetwork a)
twoHiddenNetwork = do
    w1 <- randWeight 3 1
    b1 <- randBias 3
    w2 <- randWeight 3 3
    b2 <- randBias 3
    w3 <- randWeight 1 3
    b3 <- randBias 1
    w4 <- randWeight 3 3
    b4 <- randBias 3
    return $ do denseLayer w3 b3
                denseLayer w4 b4
                denseLayer w2 b2
                denseLayer w1 b1
                inputLayer

generateEpoch :: Int -> IO [(Vector, Vector)]
generateEpoch n_samples = do
    g <- newStdGen
    let initialInputs = take n_samples (uniformRs (-pi, pi) g) -- uniformRs integrates better with list fusion apparently
        desiredOutputs = map (\x -> (sin x + 1) / 2) initialInputs -- keeps between 0 and 1
        trainingData = zipWith (\x y -> (toVector [x], toVector [y])) initialInputs desiredOutputs
    return trainingData

runEpoch network = do
    trainingData <- generateEpoch miniBatchSize
    let !nn = trainMany trainingData network    
    -- trainMany is just a foldl', and ! just needs the outermost constructor
    -- of the result. However, the result could be Op or Pure, we don't know until we evaluate the full
    -- forward and backprop (i.e. full train) over all the examples, i.e. compute the entire foldl'. 
    -- So (I think) that's why that single bang pattern suffices.
    return nn

runAllEpochs numEpochs = do
    nn <- fcNetworkPair
    go numEpochs nn
  where
    go 0 net = pure net
    go n net = do
      net' <- runEpoch net
    --   when (n `mod` 10 == 0) $ printNetwork net'
      go (n - 1) net'

-- trains the network (which is hardcoded for now in the previous functions' parameters)
-- on the sine function, then applies the output to the tests and returns a bunch of y coordinates.
trainSineForBenchOutput = do
    (nn :: Free FullyConnectedNetwork a) <- runAllEpochs numberOfIterations
    let testOutputs = vmap (vhead . head . forwardProp nn . toVector . singleton) sineTestInputs
    print testOutputs
    return testOutputs

main = do
    testOutputs <- trainSineForBenchOutput
    let !_ = force testOutputs -- lazy list, need to force all elements.
    return ()