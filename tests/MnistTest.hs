import Codec.Compression.GZip (decompress)
import qualified Data.ByteString.Lazy as BS
import Data.Functor
import System.Random
import Data.List (maximumBy)
import Data.Ord (comparing)
import NNets
import qualified Data.List.NonEmpty as NE

-- Helper functions. Copied from https://crypto.stanford.edu/~blynn/haskell/brain.html
render n = let s = " .:oO@" in s !! (fromIntegral n * length s `div` 256)

getImage s n = fromIntegral . BS.index s . (n*28^2 + 16 +) <$> [0..28^2 - 1]
getX     s n = (/ 256) <$> getImage s n
getLabel s n = fromIntegral $ BS.index s (n + 8)
getY     s n = fromIntegral . fromEnum . (getLabel s n ==) <$> [0..9]

trainingData :: BS.ByteString -> BS.ByteString -> [(Vector, Vector)]
trainingData trainI trainL =
    [(toVector $ getX trainI n, toVector $ getY trainL n) | n <- [0..59999]]

predictDigit :: [Double] -> Free ANN a -> Int
predictDigit inp nn = 
    let output = (NE.head . forwardProp nn . toVector) inp
    in (fst . maximumBy (comparing snd) . zip [0..]) (fromVector output)

checkAccuracy :: BS.ByteString -> BS.ByteString -> Free ANN a -> Double
checkAccuracy testI testL nn = 
    let correct = length
            [
                n | n <- [0..9999], getLabel testL n == predictDigit (getX testI n) nn
            ]
    in  (fromIntegral correct / 10000) * 100

-- Network definition:
type ANN = InputLayer :+: DenseLayer

mnistANN :: IO (Free ANN a)
mnistANN = do
    w1 <- randWeight 128 784 -- 784 -> 128 (first layer)
    b1 <- randBias 128
    w2 <- randWeight 64 128 -- 128 -> 64 (second layer)
    b2 <- randBias 64
    w3 <- randWeight 10 64 -- 64 -> 10 (third layer)
    b3 <- randBias 10
    return $ do denseLayer w3 b3
                denseLayer w2 b2
                denseLayer w1 b1
                inputLayer -- a vector of length 784

-- Training:
main = do
    [trainI, trainL, testI, testL] <- mapM ((decompress  <$>) . BS.readFile)
        [ "train-images-idx3-ubyte.gz"
        , "train-labels-idx1-ubyte.gz"
        ,  "t10k-images-idx3-ubyte.gz"
        ,  "t10k-labels-idx1-ubyte.gz"
        ]
    let d = trainingData trainI trainL
    initialNetwork <- mnistANN -- probably can improve the IO wrapper here
    putStrLn "Starting training"
    let !finalNetwork = trainMany d initialNetwork
    putStrLn "Finished training. checking accuracy now... "
    -- Verify accuracy: 
    let accuracy = checkAccuracy testI testL finalNetwork

    putStrLn $ "final accuracy: " ++ show accuracy
