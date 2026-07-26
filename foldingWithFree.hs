{-# LANGUAGE PatternSynonyms #-}
{-# LANGUAGE ViewPatterns #-}

module FoldingWithFree where

import Helpers
import Data.List

-- cata :: Functor f => (f a -> a) -> Free f a -> a
-- cata alg = alg . fmap (cata alg) . out

-- ana :: Functor f => (b -> f b) -> b -> Fix f
-- ana coalg = In . fmap (ana coalg) . coalg

----------------------  defining types ----------------------------------------

type Values = [Double]
type Biases = [Double]
type Weights = [[Double]]
type Deltas = [Double]


-- vector addition:
(^+^) :: [Double] -> [Double] -> [Double]
v1 ^+^ v2 = zipWith (+) v1 v2

-- vector subtraction:
(^-^) :: [Double] -> [Double] -> [Double]
v1 ^-^ v2 = zipWith (-) v1 v2

-- vector multiplication:
(^*^) :: [Double] -> [Double] -> [Double]
v1 ^*^ v2 = zipWith (*) v1 v2

-- matrix subtraction:
(-#-) :: [[Double]] -> [[Double]] -> [[Double]]
matx -#- maty = zipWith (^-^) matx maty

-- outer product: (u >< v = u . v^T)
(><) :: [Double] -> [Double] -> [[Double]]
v1 >< v2 = map (\x -> map (x *) v2) v1

-- matrix-vector multiplication:
(#>) :: [[Double]] -> [Double] -> [Double]
mat #> v = map (sum . zipWith (*) v) mat

-- sigmoid function:
sigmoid :: [Double] -> [Double]
sigmoid = map (\x -> 1 / (1 + exp (-x))) 

-- inverse then differential sigmoid function:
sigmoid' :: [Double] -> [Double]
sigmoid' = map (\x -> let y = log (x / (1-x)) in y * (1-y))

-- this time we define the layers to be different types:
data InputLayer a = InputLayer deriving Functor
data DenseLayer a = DenseLayer Weights Biases a deriving Functor

-- now we can define smart constructors:
inputLayer :: (InputLayer :<: f) => Free f a
inputLayer = Op (inj InputLayer)

denseLayer :: (DenseLayer :<: f) => Weights -> Biases -> Free f ()
denseLayer ws bs = Op (inj $ DenseLayer ws bs (Pure ()))

--------------------- forward propagation: ------------------------------------

-- promoting modularity (and simplifying the eval) by writing separately
class Functor f => AlgFwd f where
    algFwd :: f (Values -> [Values]) -> (Values -> [Values])

instance AlgFwd InputLayer where
    algFwd InputLayer = (: []) -- simplified lambda

instance AlgFwd DenseLayer where
    algFwd (DenseLayer wl bl forwardPass) = (\(vals : vs) -> 
        let cvals = sigmoid (wl #> vals ^+^ bl) in (cvals : vals : vs)) 
        . forwardPass

instance (AlgFwd f, AlgFwd g) => AlgFwd (f :+: g) where
    algFwd (L ft) = algFwd ft
    algFwd (R gt) = algFwd gt

forwardProp :: AlgFwd f => Free f a -> (Values -> [Values])
forwardProp = eval gen algFwd
    where
        gen :: a -> (Values -> [Values])
        gen = const (: [])


------------------------- back propagation: -----------------------------------

-- type to store backprop information:
data BackProp = BackProp {as :: [Values], -- all (in/out)puts from forwardProp
                          ws' :: Weights, -- weights_l+1
                          ds' :: Deltas, -- delta_l+1
                          desiredOutput :: Values} -- this layer new vals

-- patterns for easier pattern matching on the layers:
pattern InputLayer' <- (prj -> Just InputLayer)
pattern DenseLayer' ws bs prevLayer <- (prj -> Just (DenseLayer ws bs prevLayer))

-- backward computes the errors of all neurons in the l-th layer, as well as
-- the new weights and biases for the l-th layer. the coalg then uses this to
-- construct a new layer and then pass remaining information for the next one.
backward :: Weights -> Biases -> BackProp -> (Weights, Biases, Deltas)
backward ws bs (BackProp (al : alPrev : as) ws' ds' desiredOutput) =  
    let dlNew = case ds' of
            [] -> (al ^-^ desiredOutput) ^*^ sigmoid' alPrev 
            -- i.e. ∂C/∂a_jl * sigma'(z_jl) , which is why sigmoid' inverses
            _ -> (transpose ws' #> ds') ^*^ sigmoid' alPrev
        wlNew = ws -#- (alPrev >< dlNew)
        blNew = bs ^-^ dlNew
    in (wlNew, blNew, dlNew)
    -- these match the backprop equations, check notes

-- again creating a class for backwards coalgebra:
class Functor f => CoalgBwd f where
    coalgBwd :: (Free f a, BackProp) -> f (Free f a, BackProp)

-- instances for coalg are for the entire network, not just one layer
-- because we are constructing the entire network so f must be of that one type
instance CoalgBwd (InputLayer :+: DenseLayer) where
    coalgBwd (Op InputLayer', _) = inj InputLayer
    coalgBwd (Op (DenseLayer' ws bs prevLayer), backProp) = 
        let (wlNew, blNew, dlNew) = backward ws bs backProp
            backProp' = BackProp {ws' = ws, ds' = dlNew, as = tail (as backProp)}
        -- passing the current layers weights back; it becomes the next layer
        in inj (DenseLayer wlNew blNew (prevLayer, backProp'))

-- the reason a and b occur here is because there will never be a leaf of type
-- a to ever contradict the type signature. It is polymorphic in all b. The 
-- InputLayer stops the infinite tree without adding a value at the leaf 
backPropagate :: CoalgBwd f => (Free f a, BackProp) -> Free f b
backPropagate = build coalgBwd

--------------------------- training ------------------------------------------

train :: (AlgFwd f, CoalgBwd f) => (Values, Values) -> Free f a -> Free f a
train (inp, desOut) nn = (backPropagate . h . forwardProp) nn 
    where
        -- h is intermediate between forward and back prop
        -- helps transform the inputs basically
        h :: (Value -> [Values]) -> (Free f a, BackProp)
        h fwd = (nn, BackProp (fwd inp) [] [] desOut)

-- during backpropagation, it thinks it's building something completely new,
-- but it is just unwrapping the existing one from the pair, peeling layers 
-- and putting them into a new nn

