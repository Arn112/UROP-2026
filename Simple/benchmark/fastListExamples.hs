module Main where

import Helpers
import FastLists
import FastListTraining
import System.Random
import System.IO
import Data.List
import Data.Array.Unboxed
import Control.DeepSeq (force, NFData, rnf)
import Control.Exception (evaluate)

import Test.Tasty.Bench (bench, bgroup, defaultMain, env, nf, whnf, nfIO, whnfIO)

instance NFData (UArray Idx2 Double) where
    rnf arr = arr `seq` ()
-------------------------------- Training -------------------------------------

fcNetworkPair :: RandomGen g => g -> (Free FullyConnectedNetwork a, g)
fcNetworkPair g0 =
    let (w1, g1) = randVectors (Idx2 1 1) (Idx2 1 12) g0   -- output layer: 12 -> 1
        (b1, g2) = randVectors   (Idx2 1 1) (Idx2 1 1)   g1
        (w2, g3) = randVectors (Idx2 1 1) (Idx2 12 1) g2   -- hidden layer: 1 -> 12
        (b2, g4) = randVectors   (Idx2 1 1) (Idx2 1 12) g3
    in 
        (do denseLayer w1 b1
            denseLayer w2 b2
            inputLayer
        , g4
        )

generateEpoch n_samples = do
    g <- newStdGen
    let initialInputs = take n_samples (randomRs (-pi, pi) g)
        desiredOutputs = map (\x -> (sin x + 1) / 2) initialInputs -- keeps between 0 and 1
        trainingData = zipWith (\x y -> (singletonUA x, singletonUA y)) initialInputs desiredOutputs
    return trainingData

runEpoch network = do
    trainingData <- generateEpoch miniBatchSize
    let nn = trainMany trainingData network
        actualOutputs = map (head . forwardProp nn . fst) trainingData
        avgError = mse (map snd trainingData) actualOutputs
    return nn

runAllEpochs numEpochs = do
    let (network0, _) = fcNetworkPair (mkStdGen 42)
    go numEpochs network0
  where
    go 0 net = pure net
    go n net = do
      net' <- runEpoch net
      go (n - 1) net'

-- trainSineForBench :: IO [Vectors]
-- trainSineForBench = do 
--     (nn :: Free FullyConnectedNetwork a) <- runAllEpochs numberOfIterations
--     let testOutputs = map (forwardProp nn . singletonUA) (elems sineTestInputs) 
--     return testOutputs

trainSineForBenchOutput = do
    (nn :: Free FullyConnectedNetwork a) <- runAllEpochs numberOfIterations
    let testOutputs = map ((! Idx2 1 1) . head . forwardProp nn . singletonUA) (elems sineTestInputs)
    print testOutputs
    return testOutputs

main :: IO ()
-- main = return ()
-- main = defaultMain
--     [
--         bgroup "sine" [bench "training" $ nfIO runSineTraining]
--     ]
main = do
    testOutputs <- trainSineForBenchOutput
    -- print (concatMap sum testOutputs)
    -- print testOutputs
    -- let !_ = force testOutputs
    !_ <- evaluate (force testOutputs)
    return ()

-- to test:
-- cabal run ann-warray-bench -- +RTS -hy -l-agu
-- eventlog2html ann-warray-bench.eventlog
-- open -a 'Brave Browser' ann-warray-bench.eventlog.html