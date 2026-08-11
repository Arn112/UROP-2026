module Main where

import NNets
import System.Random
import System.IO
import Data.List
import Data.Array.Unboxed
import Data.Array.Base
import Control.DeepSeq (force, NFData, rnf)
import Control.Exception (evaluate)
import Control.Concurrent (threadDelay)

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
randWeight w h =
    do  gen <- newStdGen
        return $ go w h gen [[]]
    where
        go 0 0 g xs             = toMatrix xs
        go 0 h' g xs            = go w (h'-1) g ([]:xs)
        go w' h' g (row : rows) = let   xavierRange = (1.0 / sqrt (fromIntegral w))
                                        (x, g') = randomR (-xavierRange, xavierRange) g
                                  in    go (w'-1) h' g' ((x : row) : rows)

mse :: [Vector] -> [Vector] -> Double
mse expected actual = sum (map (foldlArray' (+) 0.0 . amap (\x -> x*x)) $ zipWith (^-^) expected actual) / fromIntegral (length expected)
-------------------------------- Training -------------------------------------
type FullyConnectedNetwork = (InputLayer :+: DenseLayer) 

miniBatchSize :: Int = 800
numberOfIterations :: Int = 400

sineTestInputs :: Vector
sineTestInputs = let xs = [-pi, -pi + 0.01 .. pi] in toVector xs

-- need a better API for the vector type
sineTestOutputs :: Vector
sineTestOutputs = amap (\x -> (sin x + 1) / 2) sineTestInputs

fcNetworkPair :: IO (Free FullyConnectedNetwork a)
fcNetworkPair = do
    w1 <- randWeight 12 1
    b1 <- randBias 1
    w2 <- randWeight 1 12
    b2 <- randBias 12  
    return $ do denseLayer w1 b1
                denseLayer w2 b2
                inputLayer

generateEpoch :: Int -> IO [(Vector, Vector)]
generateEpoch n_samples = do
    g <- newStdGen
    let initialInputs = take n_samples (randomRs (-pi, pi) g)
        desiredOutputs = map (\x -> (sin x + 1) / 2) initialInputs -- keeps between 0 and 1
        trainingData = zipWith (\x y -> (toVector [x], toVector [y])) initialInputs desiredOutputs
    return trainingData

runEpoch network = do
    trainingData <- generateEpoch miniBatchSize
    let nn = trainMany trainingData network
        actualOutputs = map (head . forwardProp nn . fst) trainingData
        avgError = mse (map snd trainingData) actualOutputs
    return nn

runAllEpochs numEpochs = do
    network0 <- fcNetworkPair
    go numEpochs network0
  where
    go 0 net = pure net
    go n net = do
      net' <- runEpoch net
      go (n - 1) net'

trainSineForBenchOutput = do
    (nn :: Free FullyConnectedNetwork a) <- runAllEpochs numberOfIterations
    let testOutputs = amap (flip unsafeAt 0 . head . forwardProp nn . toVector . (:[])) sineTestInputs
    -- print testOutputs
    return testOutputs

main :: IO ()
-- main = defaultMain
--     [
--         bgroup "sine" [bench "training" $ nfIO trainSineForBenchOutput]
--     ]
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