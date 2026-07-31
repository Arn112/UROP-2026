{-# LANGUAGE DeriveAnyClass, UndecidableInstances #-}

module FastLists where

import Helpers
import Data.Array.Unboxed

-- unpack removes the boxing, which takes away the overhead of the type
-- Under the hood it's functionally equivalent to Idx2 Int# Int#, just avoids
-- us have to deal with the # methods. !Int forces the args
data Idx2 = Idx2 {-# UNPACK #-} !Int {-# UNPACK #-} !Int 
    deriving (Eq, Ord, Ix) --, Show)
-- deriving Ix gives it the required methods to be used as an index in an array
-- i.e. our index can now be of type Idx2

-- UArray is a flat, strict version which stores in contiguous memory locations
-- (hence strict since thunk sizes are not constant / known). Allows for fast
-- access, but mutation is still O(n) (returns a new copy)

type Vectors = UArray Idx2 Double
-- vector operations now should compound from the speed. I'm not sure whether
-- defining like this or saying Vector = UArray Idx Double, Matrix = 
-- UArray Idx Vector is better?

v1, v2 :: Vectors
-- exValues = array (Idx2 1 1, Idx2 1 4) [(Idx2 1 i, fromIntegral(i*j)) | i <- [1..5], j <- [1..5]]
v1 = listArray (Idx2 1 1, Idx2 1 2) [1,2]
v2 = listArray (Idx2 1 1, Idx2 1 2) [4,5]

m1, m2 :: Vectors
m1 = listArray (Idx2 1 1, Idx2 2 2) [1,2,3,4]
m2 = listArray (Idx2 1 1, Idx2 2 2) [1,2,3,4]

-- this still constructs intermediate lists. May have to use ST mutations to 
-- fix that, might be the only way if this doesn't work. Or just use the vector
-- package that has all this already (but only 1D)
zipWithA :: (IArray a c, IArray a e1, IArray a e2, Ix i) => 
            (e1 -> e2 -> c) ->
            a i e1 -> a i e2 -> a i c
zipWithA f xs ys 
    | rangeSize (bounds xs) == rangeSize (bounds ys) 
        = listArray (bounds xs) [f (xs ! i) (ys ! i) | i <- range (bounds xs)]
    | otherwise              = error "incompatible bounds"


(^+^) :: Vectors -> Vectors -> Vectors
xs ^+^ ys = zipWithA (+) xs ys

(^-^) :: Vectors -> Vectors -> Vectors
v1 ^-^ v2 = zipWithA (-) v1 v2

(^*^) :: Vectors -> Vectors -> Vectors
v1 ^*^ v2 = zipWithA (*) v1 v2

(-#-) :: Vectors -> Vectors -> Vectors
matx -#- maty = zipWithA (-) matx maty

(><) :: Vectors -> Vectors -> Vectors
v1 >< v2 = listArray resultBounds [(v1 ! i) * (v2 ! j) | i <- range(bounds v1), j <- range(bounds v2)]
    where
        (start, Idx2 _ n) = bounds v1
        (_, Idx2 _ m) = bounds v2
        resultBounds = (start, Idx2 n m)

-- there is no way this is efficient with the intermediate list, also consider
-- STUArray
(#>) :: Vectors -> Vectors -> Vectors
mat #> v = listArray resultBounds [
    let row :: Vectors = listArray (bounds v) 
                            [(mat ! Idx2 i j) * (v ! Idx2 1 j) | j <- [1..m]]
    in foldlArray' (+) 0.0 row | i <- [1..n] ]

    where
        (start, Idx2 n m) = bounds mat 
        (_, Idx2 n' m') = bounds v
        resultBounds 
            | n' == 1 && m == m' = (Idx2 1 1, Idx2 1 n)
            | otherwise          = error "dimensions for matrix-vector mult. not correct"

transposeA :: Vectors -> Vectors
transposeA mat = listArray resultBounds [ mat ! Idx2 i j | j <- [1..m], i <- [1..n] ]
    where 
        (start, Idx2 n m) = bounds mat
        resultBounds = (start, Idx2 m n)

sigmoid :: Vectors -> Vectors 
sigmoid = amap (\x -> 1 / (1 + exp (-x))) 

sigmoid' :: Vectors -> Vectors
sigmoid' = amap (\x -> x * (1 - x))

type Weights = Vectors
type Biases = Vectors
type Deltas = Vectors
learningRate :: Double = 0.4

-- this time we define the layers to be different types:
data InputLayer a = InputLayer deriving Functor
data DenseLayer a = DenseLayer Weights Biases a deriving Functor

-- now we can define smart constructors:
inputLayer :: (InputLayer :<: f) => Free f a
inputLayer = Op (inj InputLayer)

denseLayer :: (DenseLayer :<: f) => Weights -> Biases -> Free f ()
denseLayer ws bs = Op (inj $ DenseLayer ws bs (Pure ()))

-- there will be some required changes for this, basically we can't have a
-- [Values] anymore, so we need a different type to store this stuff. 
-- note Idx2 doesn't work because diff num of outputs every layer
-- and we can only use UArrays for primitives
-- type Activations = Array Int Vectors

class Functor f => AlgFwd f where
    algFwd :: f (Vectors -> [Vectors]) -> (Vectors -> [Vectors])

instance AlgFwd InputLayer where
    algFwd InputLayer = (: []) -- simplified lambda

instance AlgFwd DenseLayer where
    algFwd (DenseLayer wl bl forwardPass) = (\(vals : vs) -> 
        let cvals = sigmoid ((wl #> vals) ^+^ bl) in (cvals : vals : vs)) 
        . forwardPass

instance (AlgFwd f, AlgFwd g) => AlgFwd (f :+: g) where
    algFwd (L ft) = algFwd ft
    algFwd (R gt) = algFwd gt
-- if f is large, then this rolls out to a bunch of Ls,Rs. Could be a potential
-- optimisation

genFwd :: a -> (Vectors -> [Vectors])
genFwd = const (: [])

forwardProp :: AlgFwd f => Free f a -> (Vectors -> [Vectors])
forwardProp = eval genFwd algFwd

-------- back propagation ( i really don't want to do this ughhhh )
data BackProp = BackProp {as :: [Vectors], -- all (in/out)puts from forwardProp
                          ws' :: Weights, -- weights_l+1
                          ds' :: Deltas, -- delta_l+1
                          desiredOutput :: Vectors, -- this layer new vals
                          layerIndex :: Int} -- output layer is index 0, it's weird

backward :: Weights -> Biases -> BackProp -> (Weights, Biases, Deltas)
backward ws bs (BackProp (al : alPrev : as) ws' ds' desiredOutput layerIndex) =
    let dlNew = case layerIndex of
            0 -> (al ^-^ desiredOutput) ^*^ sigmoid' al
            -- delta_j(L) = (a_j(L) - y_j) * sigma'(z_j(L))
            _  -> (transposeA ws' #> ds')  ^*^ sigmoid' al
            -- delta_l = ((w_l+1)^T delta_l+1) ^*^ sigma'(z_l)
        wlNew = ws -#- amap (*learningRate) (dlNew >< alPrev)
        -- ∂C/∂w_jk = a_k(l-1) * delta_j(l)
        blNew = bs ^-^ amap (*learningRate) dlNew
        -- ∂C/∂b_j = delta_j(l)
    in (wlNew, blNew, dlNew)

class AlgBwd g f where
    algBwd :: g (BackProp -> Free f a) -> (BackProp -> Free f a)

instance (InputLayer :<: f) => AlgBwd InputLayer f where
    algBwd InputLayer = const (Op (inj InputLayer))

instance (DenseLayer :<: f) => AlgBwd DenseLayer f where
    algBwd (DenseLayer ws bs nextLayersFunc) backPropForCurr = 
        let (wlNew, blNew, dlNew) = backward ws bs backPropForCurr
            backPropForNext = BackProp {ws' = ws, -- next layer's weights now this layer's
                                ds' = dlNew, -- same for next layer's errors
                                as = tail (as backPropForCurr), -- same for outputs
                                desiredOutput = desiredOutput backPropForCurr, -- same desired output
                                layerIndex = layerIndex backPropForCurr + 1} -- increment index

        in Op (inj (DenseLayer wlNew blNew (nextLayersFunc backPropForNext)))

instance (AlgBwd g f, AlgBwd h f) => AlgBwd (g :+: h) f where
    algBwd (L fx) = algBwd fx
    algBwd (R gx) = algBwd gx

genBwd :: (InputLayer :<: f) => a -> (BackProp -> Free f a) 
genBwd _ = const (Op (inj InputLayer))

backPropagate :: forall f a. (InputLayer :<: f, AlgBwd f f) => Free f a -> (BackProp -> Free f a)
backPropagate = eval genBwd algBwd

------------------------ training functions: --------------------------

train :: (InputLayer :<: f, AlgFwd f, AlgBwd f f) => (Vectors, Vectors) -> Free f a -> Free f a
train (inp, desOut) nn = 
    let emptyArray :: Vectors = listArray (Idx2 1 1, Idx2 1 1) [0]
        -- just a placeholder, we never use the ws' or ds' on output layer

        h :: [Vectors] -> BackProp
        h vals = BackProp vals emptyArray emptyArray desOut 0

    in (backPropagate nn . h . forwardProp nn) inp


trainMany :: (InputLayer :<: f, AlgFwd f, AlgBwd f f) 
          => [(Vectors, Vectors)] -> Free f a -> Free f a
trainMany dataSet nn = foldr train nn dataSet
