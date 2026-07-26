import Helpers
import BothFolds
import System.Random
import System.IO
import Data.List

----------------------------- IO, hyperparameters -----------------------------

sineTestInputs :: Values
sineTestInputs = [-pi, -pi + 0.01 .. pi]

sineTestOutputs :: Values
sineTestOutputs = map (\x -> (sin x + 1) / 2) sineTestInputs

miniBatchSize = 800
numberOfIterations = 1000

----------------------------------- Helpers -----------------------------------
mse :: [Values] -> [Values] -> Double
mse expected actual = sum (map (sum . map (\x -> x*x)) $ zipWith (^-^) expected actual) / fromIntegral (length expected)

randVec :: RandomGen g => Int -> g -> (Values, g)
randVec n gen = go n gen [] where
    go 0 gen xs = (xs, gen)
    go n gen xs = go (n-1) g' (x:xs)
        where (x, g') = randomR (-0.5, 0.5) gen

randMat2D :: RandomGen g => Int -> Int -> g -> ([Values], g)
randMat2D m n gen = go m gen [] where
    go 0 gen xs = (xs, gen)
    go m gen xs = go (m-1) g' (x:xs)
        where (x, g') = randVec n gen

type FullyConnectedNetwork = (InputLayer :+: DenseLayer) 

trainMany :: (InputLayer :<: f, AlgFwd f, AlgBwd f f) 
          => [(Values, Values)] -> Free f a -> Free f a
trainMany dataSet nn = foldr train nn dataSet

-------------------------------- Training -------------------------------------

fcNetworkPair :: RandomGen g => g -> (Free FullyConnectedNetwork a, g)
fcNetworkPair g0 =
--   let (w1, g1) = randMat2D 1 3 g0
--       (b1, g2) = randVec   1   g1
--       (w2, g3) = randMat2D 3 3 g2
--       (b2, g4) = randVec   3   g3
--       (w3, g5) = randMat2D 3 3 g4
--       (b3, g6) = randVec   3   g5
--       (w4, g7) = randMat2D 3 1 g6
--       (b4, g8) = randVec   3   g7
--   in
--     ( do denseLayer w1 b1
--          denseLayer w2 b2
--          denseLayer w3 b3
--          denseLayer w4 b4
--          inputLayer
--     , g8
--     )

-- fcNetworkPair g0 = 
    let (w1, g1) = randMat2D 1 12 g0   -- output layer: 12 -> 1
        (b1, g2) = randVec   1   g1
        (w2, g3) = randMat2D 12 1 g2   -- hidden layer: 1 -> 12
        (b2, g4) = randVec   12  g3
    in 
        (do denseLayer w1 b1
            denseLayer w2 b2
            inputLayer
        , g4
        )

generateEpoch n_samples = do
    g <- newStdGen
    let initialInputs = take n_samples (randomRs (-pi, pi) g)
        -- initialInputs = take n_samples (randomRs (-pi, pi) g) ++ take n_samples (randomRs (1, pi-1) g)
        desiredOutputs = map (\x -> (sin x + 1) / 2) initialInputs -- keeps between 0 and 1
        trainingData = zipWith (\x y -> ([x], [y])) initialInputs desiredOutputs
        -- trainingData = zipWith (\x y -> ([x], [y])) sineTestInputs sineTestOutputs
    --putStrLn ("training data: " ++ show trainingData)
    return trainingData

runEpoch network = do
    trainingData <- generateEpoch miniBatchSize
    let nn = trainMany trainingData network
        actualOutputs = map (head . forwardProp nn . fst) trainingData
        avgError = mse (map snd trainingData) actualOutputs
        -- actualOutputs = map (head . forwardProp nn . singleton) sineTestInputs
        -- avgError = mse (map singleton sineTestOutputs) actualOutputs

    -- putStrLn ("actualOutputs: " ++ show actualOutputs ++ ", desired outputs: " ++ show (map snd trainingData))
    -- putStrLn ("finished another epoch. avg loss: " ++ show avgError)
    return nn

runAllEpochs numEpochs = do
    let (network0, _) = fcNetworkPair (mkStdGen 42)
    go numEpochs network0
  where
    go 0 net = pure net
    go n net = do
      net' <- runEpoch net
      go (n - 1) net'

main = do
    nn <- runAllEpochs numberOfIterations

    putStrLn "finished training the network with samples. Testing outputs: "
    -- print ("sin (1.0) + 1 / 2: " ++ show (head $ forwardProp nn [1.0]))
    -- print ("sin (0.5) + 1 / 2 : " ++ show (head $ forwardProp nn [0.5]))
    -- print ("sin (pi/6) + 1 / 2 : " ++ show (head $ forwardProp nn [pi/6]))
    print $ show (map (head . head . forwardProp nn . singleton) sineTestInputs)