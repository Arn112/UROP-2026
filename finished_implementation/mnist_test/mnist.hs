module Mnist where

-- internet archive for MNist download
-- also like all the code here is basically from this one guy: 
-- https://crypto.stanford.edu/~blynn/haskell/brain.html
import CnnCode
import CnnTest
import Helpers

import Codec.Compression.GZip (decompress)
import qualified Data.ByteString.Lazy as BS
import Data.Functor
import System.Random
import Data.Array.IO
import Control.Monad

------------- HYPERPARAMETERS --------------
-- learningrate is in the cnnCode file, dk if I can move here
numEpochs :: Int = 50
trainingBatchSize :: Int = 5000

type Image = [[Int]]

getImage s n = fromIntegral . BS.index s . (n*28^2 + 16 +) <$> [0..28^2 - 1]
getX     s n = (/ 256) <$> getImage s n
getLabel s n = fromIntegral $ BS.index s (n + 8)
getY     s n = fromIntegral . fromEnum . (getLabel s n ==) <$> [0..9]

-- to sample from a normal distribution:
gauss :: Float -> IO Float
gauss stdev = do
    x1 <- randomIO
    x2 <- randomIO
    return $ stdev * sqrt (-2 * log x1) * cos (2 * pi * x2)

-- from: https://wiki.haskell.org/Random_shuffle
-- | Randomly shuffle a list
--   /O(N)/
shuffle :: [a] -> IO [a]
shuffle xs = do
        ar <- newArray n xs
        forM [1..n] $ \i -> do
            j <- randomRIO (i,n)
            vi <- readArray ar i
            vj <- readArray ar j
            writeArray ar j vi
            return vj
  where
    n = length xs
    newArray :: Int -> [a] -> IO (IOArray Int a)
    newArray n xs =  newListArray (1,n) xs

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

-- take a random subset of size N of the images
-- mnistSubsetN :: Int -> [Image] -> [Int] -> [(Image,Int)]
mnistSubsetN n mnist labels = do
    shuffled <- shuffle [1..n]
    return $ map (\x -> (getImage mnist (fromIntegral x), 
                        getLabel labels (fromIntegral x))) shuffled
    

-- takes an integer and puts 1.0 in that position in a list, 0 everywhere else
mnistDesiredOut :: Int -> [Double]
mnistDesiredOut i = map (\x -> if x == i then 1.0 else 0.0) [1..10]

-- train nn on a certain sized dataset
trainNImages nn xs = do
    dataset <- shuffle xs
    return $ foldl' (flip trainCNN) nn dataset

-- runs through a full epoch of training (i.e. full dataset) N times
-- foldM used here because trainNImages returns an IO wrapped cnn
-- (because shuffle uses IO?)
trainNEpochs nn xs n = foldM (\currNN x -> trainNImages currNN xs) (pure nn) [1..n]

-- no clue what the difference between forM and forM_ is
testCNN testData cnn = do
    forM_ (take 10 testData) $ \(inp, desOut) -> do
        let (Vec cnnOutput) = head $ forwardPropConv cnn inp -- this might be a Mat, not sure
        putStrLn $ "expected: " ++ show desOut
        putStrLn $ "actual: " ++ show cnnOutput
        putStrLn "---------"


untrainedCNN :: RandomGen g => g -> Free ConvNetwork a
untrainedCNN g0 = 
    let (w1, g1) = randMat2D 10 32 g0 -- output layer: 32 -> 10
        (b1, g2) = randVec 10 g1 
        (w2, g3) = randMat2D 32 144 g2 -- hidden layer: 144 -> 32
        (b2, g4) = randVec 32 g3
        (k1, g5) = randMat2D 5 5 g4
        bk = 0.0
    
    in 
        (do denseLayer w1 b1
            denseLayer w2 b2
            flattenLayer 12 12
            poolLayer 2 2
            convLayer k1 bk
            inputLayer
        )
    

main = do
    [trainI, trainL, testI, testL] <- mapM ((decompress  <$>) . BS.readFile)
        [ "dataset/train-images-idx3-ubyte.gz"
        , "dataset/train-labels-idx1-ubyte.gz"
        ,  "dataset/t10k-images-idx3-ubyte.gz"
        ,  "dataset/t10k-labels-idx1-ubyte.gz"
        ]

    -- using a small subset of MNIST to test training:
    mnistSub <- mnistSubsetN trainingBatchSize trainI trainL
    let trainingData = map (\(inp, x) -> (Mat inp, mnistDesiredOut x)) mnistSub
        -- output label 1 means desOut is [1,0,0,0,...,0] etc.
    trainedCNN <- trainNEpochs (untrainedCNN (mkStdGen 67)) trainingData numEpochs

    -- testing accuracy on test images (a few for now):
    let testImageLabelPairs = map (\x -> (getImage testI x, getLabel testL x)) [1..100]
        testData = map (\(inp, x) -> (Mat inp, mnistDesiredOut x)) testImageLabelPairs
    testCNN testData trainedCNN