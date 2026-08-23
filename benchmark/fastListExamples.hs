{-# LANGUAGE BangPatterns #-}

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


mse :: [Vector] -> [Vector] -> Double
mse expected actual = sum (map (vfoldl' (+) 0.0 . vmap (\x -> x*x)) $ zipWith (^-^) expected actual) / fromIntegral (length expected)
-------------------------------- Training -------------------------------------
type FullyConnectedNetwork = (InputLayer :+: DenseLayer) 

miniBatchSize :: Int = 800
numberOfIterations :: Int = 200

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
    let initialInputs = take n_samples (uniformRs (-pi, pi) g) -- integrates better with list fusion apparently
        desiredOutputs = map (\x -> (sin x + 1) / 2) initialInputs -- keeps between 0 and 1
        trainingData = zipWith (\x y -> (toVector [x], toVector [y])) initialInputs desiredOutputs
    return trainingData

runEpoch network = do
    trainingData <- generateEpoch miniBatchSize
    let !nn = trainMany trainingData network -- this fixed the strictness issue in this file. 
    {-
    Honestly I don't know exactly why this bang pattern fixed it. I just had a hunch. 
    I think it's because, trainMany is just a foldl', and ! just needs the outermost constructor
    of the result. However, the result could be Op or Pure, we don't know until we evaluate the full
    forward and backprop (i.e. full train) over all the examples, i.e. compute the entire foldl'. 
    I don't even think it can skip that last train completely because each train requies a function
    composition which is only applied right at the end, so maybe it would do backprop until the first
    DenseLayer and then stop? But then that's so incomplete and it would have to finish before the next
    train anyways. 

    Altho this doesn't explain why the old version before the bang took up so much memory. There's so
    much ARR_WORDS on that one but the arrays are strict and there's not that many of them, so the 
    only thing I can think of is those are thunks which have a result of ARR_WORDS?

    Maybe the chain of thunks contains space for each trainingData and weights and biases which is preallocated?
    and then can only be GC'd? But then why a slow decay and a cliff at the end?

    apparently sawtooth pattern is generational GC behaviour (claude says it is but idk what's causing it here?)
    -}
    return nn

runAllEpochs numEpochs = do
    network0 <- fcNetworkPair
    go numEpochs network0
  where
    go 0 net = pure net
    go n net = do
      net' <- runEpoch net
    --   when (n `mod` 10 == 0) $ printNetwork net'
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