import Codec.Compression.GZip (decompress)
import Data.ByteString qualified as B
import qualified Data.ByteString.Lazy as BS
import Control.Exception (evaluate)
import Data.ByteString.Unsafe qualified as BU
import Data.Functor
import System.Random
import Data.List (maximumBy)
import Data.Ord (comparing)
import Data.Map qualified as Map
import NNets
import qualified Data.List.NonEmpty as NE

-- Helper functions. Inspired from https://crypto.stanford.edu/~blynn/haskell/brain.html
render n = let s = " .:oO@" in s !! (fromIntegral n * length s `div` 256)

-- here, WHNF = NF, so evaluate forces the full thing
loadGz :: FilePath -> IO B.ByteString
loadGz p = BS.readFile p >>= evaluate . BS.toStrict . decompress

-- unsafe indexing is pointer arithmetic compared to normal lazy indexing which
-- was linked list traversal
getX :: B.ByteString -> Int -> Matrix
getX s n = (toMatrix . chunksOf 28) [fromIntegral (BU.unsafeIndex s (n * 784 + 16 + i)) / 256 | i <- [0 .. 783]]

getLabel :: B.ByteString -> Int -> Int
getLabel s n = fromIntegral (BU.unsafeIndex s (n + 8))

getY :: B.ByteString -> Int -> Vector
getY     s n = toVector $ fromIntegral . fromEnum . (getLabel s n ==) <$> [0..9]

trainingData :: B.ByteString -> B.ByteString -> [(Tensor, Vector)]
trainingData trainI trainL =
    [(TMat $ getX trainI n, getY trainL n) | n <- [0..9999]]

predictDigit :: Matrix -> Free CNN a -> Int
predictDigit inp nn = 
    let (TVec output) = (head . forwardPropConv nn . TMat) inp
    in (fst . maximumBy (comparing snd) . zip [0..]) (fromVector output)

checkAccuracy :: B.ByteString -> B.ByteString -> Free CNN a -> Double
checkAccuracy testI testL nn = 
    let correct = length
            [
                n | n <- [0..9999], getLabel testL n == predictDigit (getX testI n) nn
            ]
    in  (fromIntegral correct / 10000) * 100

predictionHistogram :: B.ByteString -> Free CNN a -> Map.Map Int Int
predictionHistogram testI nn = foldr f (Map.empty) [0..9999]
    where
        f i mp = Map.insertWith (\ x y -> y + 1) (predictDigit (getX testI i) nn) 0 mp

-- Network definition:
type CNN = (InputLayer :+: DenseLayer :+: ConvLayer :+: PoolLayer :+: FlattenLayer)

mnistCNN :: IO (Free CNN a)
mnistCNN = do
    f1 <- randWeight 5 5 -- 28 x 28 x 1 conv layer
    b1 <- randBias 1
    w2 <- randWeight 64 144 -- 144 -> 64 (first dense layer)
    b2 <- randBias 64
    w3 <- randWeight 10 64 -- 64 -> 10 (second dense layer)
    b3 <- randBias 10
    -- return $ do denseLayer w3 b3
    --             denseLayer w2 b2
    --             flattenLayer 12 12
    --             poolLayer 2 2
    --             convLayer f1 (head $ fromVector $ b1)
    --             inputLayer -- a matrix of size 28 x 28
    return $ do denseLayer w3 b3
                denseLayer w2 b2
                flattenLayer 12 12
                poolLayer 2 2
                convLayer ((toMatrix . chunksOf 5) (replicate 25 0.04)) 0.0
                inputLayer -- a matrix of size 28 x 28

-- Training:
main = do
    [trainI, trainL, testI, testL] <- mapM loadGz
        [ "train-images-idx3-ubyte.gz"
        , "train-labels-idx1-ubyte.gz"
        ,  "t10k-images-idx3-ubyte.gz"
        ,  "t10k-labels-idx1-ubyte.gz"
        ]
    setStdGen (mkStdGen 42)

    let d = trainingData trainI trainL
    initialNetwork <- mnistCNN -- probably can improve the IO wrapper here

    putStrLn "Printing initial network:"
    printNetwork initialNetwork

    putStrLn "Starting training"
    let !finalNetwork = trainManyCNN d initialNetwork
    putStrLn "Finished training. checking accuracy now... "
    -- Verify accuracy: 
    let accuracy = checkAccuracy testI testL finalNetwork

    putStrLn $ "final accuracy: " ++ show accuracy

    -- Print network to make sure it's not dead:
    printNetwork finalNetwork

    -- Show the prediction histogram:
    putStrLn $ "our outputs: " ++ show (predictionHistogram testI finalNetwork)
