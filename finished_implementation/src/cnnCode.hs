-- Implementing the CNN example in the paper. There is not much detail, so need
-- to derive the algebras myself.
{-# LANGUAGE UndecidableInstances #-}

module CnnCode where

import Helpers
import BothFolds
import Data.List
import Data.List.Utils

import Debug.Trace

type Filter = [[Double]]
type SpatExt = Int -- spatial extent
type Stride = Int

-------------- Operations ------------------
-- Disclaimer: These operations work but to say they are inefficient is an
-- understatement. A full redesign will probably be needed for conv and pool

relu :: Values -> Values
relu = map (max 0)

-- each 'window' can be described as the spatExt x spatExt 'screenshot' at i,j
-- we take all 'screenshots' then extract the maximum inside it
pool :: [Values] -> Int -> Int -> [Values] -- assuming square matrix vals
pool vals spatExt stride = 
    let n = length vals 
        nums = [0, 0+stride .. (n-spatExt)] 
        windows = map (\x -> map (\y -> map (take spatExt . drop y) ((take spatExt . drop x) vals)) nums) nums
    in map (map (maximum . map maximum)) windows

    
-- test vals spatExt stride = 
--     let n = length vals 
--         nums = [0, 0+stride .. (n-spatExt)] 
--         windows = map (\x -> map (\y -> map (take spatExt . drop y) ((take spatExt . drop x) vals)) nums) nums
--     in map (map (maximum . map maximum)) windows

-- pretending for now that we only have one filter and hence one bias value
-- do the same window thing as before, then take dot product of matrices
-- and sum. apply bias and ReLu before returning.
convolution :: [Values] -> Filter -> Double -> [Values]
convolution vals filter b = 
    let n = length vals 
        m = length filter
        nums = [0 .. (n-m)] 
        windows = map (\x -> map (\y -> map (take m . drop y) ((take m . drop x) vals)) nums) nums
        featureMap = map (map (sum . map sum . zipWith (^*^) filter)) windows
    in map (relu . map (b +)) featureMap
-- side note: this operation is actually called cross correlation but whatever
-- https://glassboxmedicine.com/2019/07/26/convolution-vs-cross-correlation/


-- convTestVals = [[0,0,1,1,0,0],[0,1,0,0,1,0],[1,0,0,0,0,1],[1,0,0,0,0,1],[0,1,0,0,1,0],[0,0,1,1,0,0]]
-- convTestFilter = [[0,0,1],[0,1,0],[1,0,0]]

-- (^**^) :: [Integer] -> [Integer] -> [Integer]
-- v1 ^**^ v2 = zipWith (*) v1 v2

-- testConv vals filter b = 
--     let n = length vals 
--         m = length filter
--         nums = [0 .. (n-m)] 
--         windows = map (\x -> map (\y -> map (take m . drop y) ((take m . drop x) vals)) nums) nums
--         featureMap = map (map (sum . map sum . zipWith (^**^) filter)) windows
--     in map ((map (max 0)) . map (b +)) featureMap

-------------- Instances ------------------

-- first we need the layers definitions and smart constructors:
data ConvLayer a = ConvLayer Filter Double a deriving Functor
data PoolLayer a = PoolLayer SpatExt Stride a deriving Functor

-- the need for a flatten layer will be seen when describing the algebra
-- we have two parameters for height and width, corresponding to the result
-- of the previous pooling / conv layer. this will need to be given in 
-- manually for now. 
data FlattenLayer a = FlattenLayer Int Int a deriving Functor 

convLayer :: (ConvLayer :<: f) => Filter -> Double -> Free f ()
convLayer filter biases = Op (inj $ ConvLayer filter biases (Pure ()))

poolLayer :: (PoolLayer :<: f) => SpatExt -> Stride -> Free f ()
poolLayer spatialExtent stride = Op (inj $ PoolLayer spatialExtent stride (Pure ()))

flattenLayer :: (FlattenLayer :<: f) => Int -> Int -> Free f ()
flattenLayer height width = Op (inj $ FlattenLayer height width (Pure ()))

-- need instances of AlgFwd for all new layers. The problem is that conv layers
-- take in an image as an input but then turn that into a vector near the end
-- for the denselayer step. So halfway through we switch from a Values -> [Values]
-- function to a [Values] -> [Values] function. We can unify these simply with a 
-- tensor type. This isn't really type safe I don't think (it will just throw type
-- mismatch at runtime) but I'd rather get it working
data Tensor = Vec Values | Mat [Values]

-------------------- Forward Propagation ----------------------------
-- Here we have created a new typeclass but I can definitely merge this with 
-- the old one later on - change old one to use tensors.

class Functor f => AlgFwdNew f where
    algFwdNew :: f (Tensor -> [Tensor]) -> (Tensor -> [Tensor])

instance AlgFwdNew InputLayer where
    algFwdNew InputLayer = (: []) -- simplified lambda

instance AlgFwdNew DenseLayer where
    algFwdNew (DenseLayer wl bl forwardPass) = (\((Vec vals) : vs) -> 
        let cvals = Vec (sigmoid ((wl #> vals) ^+^ bl)) in (cvals : Vec vals : vs)) 
        . forwardPass
    -- should change to use ReLU and cross entropy loss later.
        
instance AlgFwdNew FlattenLayer where
    algFwdNew (FlattenLayer height width forwardPass) =
        (\((Mat poolLayerOutput) : vs) -> Vec (concat poolLayerOutput) : Mat poolLayerOutput : vs)
        . forwardPass
    -- we turn the 2D output of the prev pool (or conv) layer into a vector
    -- then add it to the activations of each layer so the next denselayer
    -- sees the right thing

instance AlgFwdNew PoolLayer where
    algFwdNew (PoolLayer spatialExtent stride forwardPass) = 
        (\(Mat vals : vs) -> let pooledVals = pool vals spatialExtent stride in (Mat pooledVals : Mat vals : vs))
        . forwardPass

instance AlgFwdNew ConvLayer where
    algFwdNew (ConvLayer filter bias forwardPass) = 
        (\(Mat vals : vs) -> let convolvedVals = convolution vals filter bias in (Mat convolvedVals : Mat vals : vs))
        . forwardPass

instance (AlgFwdNew f, AlgFwdNew g) => AlgFwdNew (f :+: g) where
    algFwdNew (L ft) = algFwdNew ft
    algFwdNew (R gt) = algFwdNew gt


genFwdNew :: a -> (Tensor -> [Tensor])
genFwdNew = const (: [])

forwardPropConv :: AlgFwdNew f => Free f a -> (Tensor -> [Tensor])
forwardPropConv = eval genFwdNew algFwdNew

------------------- Back Propagation ----------------------------
-- https://www.youtube.com/watch?v=vbUozbkMhI0&list=PLuhqtP7jdD8CD6rOWy20INGM44kULvrHu&index=9
-- this guy is the goat

-- we create a unified backprop representation because the backwarwd algebra 
-- has type:  g (BackProp -> Free f a) -> (BackProp -> Free f a) so we can't 
-- change BackProp exactly :( There is probably a fix though

-- data BackPropCNN = BackPropCNN { _as :: [Tensor], -- all (in/out)puts from forwardProp
--                           _ws' :: Weights,
--                           _ds' :: Deltas, 
--                           _desiredOutput :: Values, 
--                           _dsCNN' :: [Values] } -- ∂L/∂Z, where Z is the output

data BackPropNew = BackPropCNN {cnnAs :: [Tensor], cnnDs' :: Tensor} 
                | BackPropMLP {mlpAs :: [Tensor], mlpWs' :: Weights, mlpDs' :: Deltas, mlpDesiredOutput :: Values}

-- need instances of AlgBwd for all new layers. New class needs creating:
class AlgBwdNew g f where
    algBwdNew :: g (BackPropNew -> Free f a) -> (BackPropNew -> Free f a)

instance (AlgBwdNew g f, AlgBwdNew h f) => AlgBwdNew (g :+: h) f where
    algBwdNew (L fx) = algBwdNew fx
    algBwdNew (R gx) = algBwdNew gx


-------- InputLayer: ----------

instance (InputLayer :<: f) => AlgBwdNew InputLayer f where
    algBwdNew InputLayer = const (Op (inj InputLayer))

-------- DenseLayer: ----------
-- currently using sigmoid and L2 loss but should change to cross entropy
backwardMLP :: Weights -> Biases -> BackPropNew -> (Weights, Biases, Deltas)
backwardMLP ws bs (BackPropMLP (Vec al : Vec alPrev : as) ws' ds' desiredOutput) =
    let dlNew = case ds' of
            [] -> (al ^-^ desiredOutput) ^*^ sigmoid' al
            -- delta_j(L) = (a_j(L) - y_j) * sigma'(z_j(L))
            _  -> (transpose ws' #> ds')  ^*^ sigmoid' al
            -- delta_l = ((w_l+1)^T delta_l+1) ^*^ sigma'(z_l)
        wlNew = ws -#- map (map (*learningRate)) (dlNew >< alPrev)
        -- ∂C/∂w_jk = a_k(l-1) * delta_j(l)
        blNew = bs ^-^ map (*learningRate) dlNew
        -- ∂C/∂b_j = delta_j(l)
    in trace ("dlNew: " ++ show dlNew) (wlNew, blNew, dlNew)

instance (DenseLayer :<: f) => AlgBwdNew DenseLayer f where
    algBwdNew (DenseLayer ws bs nextLayersFunc) backPropForCurr = 
        let (wlNew, blNew, dlNew) = backwardMLP ws bs backPropForCurr
            backPropForNext = BackPropMLP {mlpWs' = ws, -- next layer's weights now this layer's
                                mlpDs' = dlNew, -- same for next layer's errors
                                mlpAs = tail (mlpAs backPropForCurr), -- same for outputs
                                mlpDesiredOutput = mlpDesiredOutput backPropForCurr} -- same desired output

        in Op (inj (DenseLayer wlNew blNew (nextLayersFunc backPropForNext)))

------------ ConvLayer: --------------

-- Source of the below function - https://stackoverflow.com/a/8700618
-- takes a vector and turns it into a square matrix of size n
reshape :: Int -> [a] -> [[a]]
reshape n = takeWhile (not.null) . map (take n) . iterate (drop n)

-- -- pads with a single layer of zeroes enveloping the entire matrix.
-- padMatrix :: [Values] -> [Values]
-- padMatrix mat = [emptyRow] ++ mat' ++ [emptyRow] where
--     emptyRow = replicate (length (head mat) + 2) 0.0
--     mat' = map (\x -> [0.0] ++ x ++ [0.0]) mat

-- pads with n layers of zeroes enveloping the entire matrix.
padMatrixN :: [Values] -> Int -> [Values]
padMatrixN mat n = nEmptyRows ++ mat' ++ nEmptyRows where
    emptyRow = replicate (length (head mat) + 2*n) 0.0
    nEmptyRows = replicate n emptyRow
    mat' = map (\x -> let rp = replicate n 0.0 in rp ++ x ++ rp) mat

-- rotates the matrix by 180 degrees (i.e. pi radians)
rotatePi :: [Values] -> [Values]
rotatePi mat = reverse $ map reverse mat

backwardConvLayer :: Filter -> Double -> BackPropNew -> (Filter, Double, Tensor)
-- again the head of the list is the output of the current layer
-- so the output of the conv. layer is ReLU(Z)
backwardConvLayer filter bias (BackPropCNN ((Mat reluZ) : (Mat x) : ts) (Mat dLdC)) = 
        -- ∂L/∂Z = ∂L/∂(Re(Z)) * ∂(Re(Z))/∂Z, ∂(Re(Z_mn))/∂Z_mn = 1 if Z_mn > 0, 0 otherwise
        -- keep in mind the * is pointwise multiplication because we did everything
        -- one index at a time then just made the matrix notation to keep it cleaner
    let dCdZ = map (map (\x -> if x > 0.0 then 1.0 else 0.0)) reluZ
        dLdZ = zipWith (^*^) dLdC dCdZ -- matrix dot product
        -- ∂L/∂K = conv (x, ∂L/∂Z), can be derived
        delK = convolution x dLdZ 0.0
        -- ∂L/∂B = ∑ ∂L/∂Z
        delB = sum $ concat dLdZ
        -- icba writing it out but this is also derivable
        delX = convolution (padMatrixN dLdZ (length filter - 1)) (rotatePi filter) 0.0
    in trace ("dldz: " ++ show dLdZ ++ "dcdz: " ++ show dCdZ ++ "dLdC: " ++ show dLdC) (delK, delB, Mat delX)

instance (ConvLayer :<: f) => AlgBwdNew ConvLayer f where
    algBwdNew (ConvLayer filter bias nextLayersFunc) backPropForCurr = 
        let (delK, delB, delX) = backwardConvLayer filter bias backPropForCurr
            updFilter = filter -#- delK
            updBias = bias - delB
            backPropForNext = BackPropCNN {cnnAs = tail (cnnAs backPropForCurr),
                                           cnnDs' = delX}
        in Op (inj (ConvLayer updFilter updBias (nextLayersFunc backPropForNext)))
-- NEED TO ACCOUNT FOR THE RELU!!!!! (i think i have?)

------------ PoolLayer: --------------
windows :: [Values] -> Int -> Int -> [[[Values]]]
windows vals spatExt stride = 
    let n = length vals 
        nums = [0, 0+stride .. (n-spatExt)] 
    in map (\x -> map (\y -> map (take spatExt . drop y) ((take spatExt . drop x) vals)) nums) nums

-- again for simplicity we assume square matrices for now
reversePool :: Tensor -> Tensor -> Tensor -> Int -> Int -> [Values]
reversePool (Mat inp) (Mat out) (Mat dLdZ) spe str = 
    let n = length inp
        wns = concat $ windows inp spe str -- create the windows again
        scaledDLdZ = zipWith (\wdw repl -> repl / fromIntegral (countElem repl (concat wdw))) wns (concat dLdZ)
        -- scaledDLdZ = zipWith (\wdw repl -> repl / (max (fromIntegral (countElem repl (concat wdw))) 1.0)) wns (concat dLdZ)
        xs = zip (concat out) (concat dLdZ) -- result of pool and what to sub.
        -- xs = zip (concat out) scaledDLdZ -- result of pool and what to sub.
        ys = zipWith (\zs (seen,repl) -> map (map (\x -> if x == seen then repl else 0)) zs) wns xs
        -- replace window positions with either derivative or 0
    in trace ("inp: " ++ show inp ++ ", out: " ++ show out ++ ", dLdZ: " ++ show dLdZ) concatMap (map concat . transpose) (reshape spe ys) 
        -- turn windows format back into matrix format

-- remember how the backprop for pooling works: if any element in the input 
-- (to the pool layer) was equal to the maximum in that window, it is replaced
-- with the derivative value of that pool's position. Otherwise, it is 0.-
-- (this is for calculating ∂L/∂X btw). Small caveat: for repeat entries either
-- split equally or just give to one. There is some mathematical reason but idk
-- here I find it is easier to just split equally. 

-- it's inconsistent because I'm pattern matching on the backPropForCurr ik
-- will fix it later i swear, will have to rewrite this anyways
instance (PoolLayer :<: f) => AlgBwdNew PoolLayer f where
    algBwdNew (PoolLayer spatialExtent stride nextLayersFunc) (BackPropCNN (out : inp : rest) dLdZ) =
        let dLdX = reversePool inp out dLdZ spatialExtent stride
            backPropForNext = BackPropCNN {cnnDs' = Mat dLdX, cnnAs = inp : rest}
        in Op (inj (PoolLayer spatialExtent stride (nextLayersFunc backPropForNext)))
-- nothing needs updating, just need to find ∂L/∂X, X being the input into pool
-- to pass backwards. 

------------ FlattenLayer: --------------

instance (FlattenLayer :<: f) => AlgBwdNew FlattenLayer f where
    -- algBwdNew (FlattenLayer height width nextLayersFunc) backPropForCurr = 
    algBwdNew (FlattenLayer height width nextLayersFunc) backPropForCurr = 
        let Vec al = head $ mlpAs backPropForCurr 
            flattened = (transpose (mlpWs' backPropForCurr) #> (mlpDs' backPropForCurr))
            -- flattened = mlpDs' backPropForCurr
            reshapedMat = reshape width flattened -- reshape the vector of errors into a matrix
            -- ignore the element we added to the AS during forward prop - repeated result
            backPropForNext = BackPropCNN {cnnDs' = Mat reshapedMat, cnnAs = tail (mlpAs backPropForCurr)}
        in trace ("mlpWs': " ++ (show $ mlpWs' backPropForCurr) ++ ", mlpDs': " ++ (show $ mlpDs' backPropForCurr) ) Op (inj (FlattenLayer height width (nextLayersFunc backPropForNext)))
-- ok this needs a bit more explanation than I initially thought. 
-- The thing we need to reshape is actually the error of the INPUT layer, but
-- previous I was calculating the error of the layer after that. So you just do
-- the same thing as the MLP backprop for the dlNew case, but without the 
-- sigmoid because there was no calculation even done in this layer hence
-- no activation function was used. 

--------------- The actual final backprop function ----------------------------
genBwdNew :: (InputLayer :<: f) => a -> (BackPropNew -> Free f a) 
genBwdNew _ = const (Op (inj InputLayer))

backPropagateCNN :: forall f a. (InputLayer :<: f, AlgBwdNew f f) => Free f a -> (BackPropNew -> Free f a)
backPropagateCNN = eval genBwdNew algBwdNew

trainCNN :: (InputLayer :<: f, AlgFwdNew f, AlgBwdNew f f) => (Tensor, Values) -> Free f a -> Free f a
trainCNN (inp, desout) nn = res 
    where
        vals = forwardPropConv nn inp
        res = backPropagateCNN nn (BackPropMLP vals [] [] desout)