module FastListTests where

import Helpers
import FastLists
import System.Random
import System.IO
import Data.List
import Data.Array.Unboxed
----------------------------- IO, hyperparameters -----------------------------

sineTestInputs :: Vectors
sineTestInputs = let xs = [-pi, -pi + 0.01 .. pi] 
                in listArray (Idx2 1 1, Idx2 1 (length xs)) xs

sineTestOutputs :: Vectors
sineTestOutputs = amap (\x -> (sin x + 1) / 2) sineTestInputs

miniBatchSize = 80
numberOfIterations = 50

-- ----------------------------------- Helpers -----------------------------------
mse :: [Vectors] -> [Vectors] -> Double
mse expected actual = sum (map (foldlArray' (+) 0.0 . amap (\x -> x*x)) $ zipWith (^-^) expected actual) / fromIntegral (length expected)

randVectors :: RandomGen g => Idx2 -> Idx2 -> g -> (Vectors, g)
randVectors lo hi gen = go (range (lo, hi)) gen []
  where
    go [] g xs     = (listArray (lo, hi) xs, g)
    go (_:is) g xs = let (x, g') = randomR (-0.5, 0.5) g
                     in go is g' (x : xs)

singletonUA :: Double -> UArray Idx2 Double
singletonUA x = listArray (Idx2 1 1, Idx2 1 1) [x]

type FullyConnectedNetwork = (InputLayer :+: DenseLayer) 

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

-- runSineTraining :: IO (Free FullyConnectedNetwork a)
runSineTraining = do
    nn <- runAllEpochs numberOfIterations
    -- return nn
    -- let testOutputs = map (head . head . forwardProp nn . singleton) sineTestInputs
    -- let testOutputs = map (forwardProp nn . singleton) sineTestInputs
    -- return testOutputs
    -- return nn

    putStrLn "finished training the network with samples. Testing outputs: "
    print ("sin (1.0) + 1 / 2: " ++ show (head $ forwardProp nn (singletonUA 1.0)))
    print ("sin (0.5) + 1 / 2 : " ++ show (head $ forwardProp nn (singletonUA 0.5)))
    print ("sin (pi/6) + 1 / 2 : " ++ show (head $ forwardProp nn (singletonUA (pi/6))))
    -- print $ show (map (head . head . forwardProp nn . singleton) sineTestInputs)




