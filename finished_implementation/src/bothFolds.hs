{-# LANGUAGE PatternSynonyms #-}
{-# LANGUAGE ViewPatterns #-}
{-# LANGUAGE UndecidableInstances #-}

module BothFolds where

import Helpers
import Data.List

----------------------  defining types ----------------------------------------
learningRate = 1.0 :: Double -- best results were with 0.6, 1 - 12 - 1 setup

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

-- vector multiplication (i.e. hadamard product):
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
-- sigmoid' = map (\x -> let y = log (x / (1-x)) in y * (1-y))
sigmoid' = map (\x -> x * (1 - x)) -- this is indeed correct for inverse then derivative

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
        let cvals = sigmoid ((wl #> vals) ^+^ bl) in (cvals : vals : vs)) 
        . forwardPass
        -- note that the newest outputs are added to the FRONT of the list
        -- i.e. activations of the last layer will be at the front, etc.
        

instance (AlgFwd f, AlgFwd g) => AlgFwd (f :+: g) where
    algFwd (L ft) = algFwd ft
    algFwd (R gt) = algFwd gt
-- if f is large, then this rolls out to a bunch of Ls,Rs. Could be a potential
-- optimisation

genFwd :: a -> (Values -> [Values])
genFwd = const (: [])

forwardProp :: AlgFwd f => Free f a -> (Values -> [Values])
forwardProp = eval genFwd algFwd
        
{- Explanation on forward prop:
remember what eval is doing: applying algebra to current layer after evaluating
all the nested layers. The end goal is a function which takes in the input 
values and evaluates them as the network would have. I.e. function composition.

each DenseLayer forward pass evaluates the previous layers forwardpass, 
pretends it's taking in the inputs and returns the output (pretend because it's
a lambda) of the current layer.
-}

------------------------- back propagation: -----------------------------------

-- type to store backprop information:
data BackProp = BackProp {as :: [Values], -- all (in/out)puts from forwardProp
                          ws' :: Weights, -- weights_l+1
                          ds' :: Deltas, -- delta_l+1
                          desiredOutput :: Values} -- this layer new vals

-- backward computes the errors of all neurons in the l-th layer, as well as
-- the new weights and biases for the l-th layer. the coalg then uses this to
-- construct a new layer and then pass remaining information for the next one.
-- backward :: Weights -> Biases -> BackProp -> (Weights, Biases, Deltas)
-- backward ws bs (BackProp (al : alPrev : as) ws' ds' desiredOutput) =  
--     let dlNew = case ds' of
--             [] -> (al ^-^ desiredOutput) ^*^ sigmoid' alPrev
--             -- i.e. ∂C/∂a_jl * sigma'(z_jl) , which is why sigmoid' inverses
--             _ -> (transpose ws' #> ds') ^*^ sigmoid' alPrev
--         wlNew = ws -#- (alPrev >< dlNew)
--         blNew = bs ^-^ dlNew
--     in (wlNew, blNew, dlNew)
--     -- these match the backprop equations, check notes

-- this for some reason seems correct. still try with the previous version. 
backward :: Weights -> Biases -> BackProp -> (Weights, Biases, Deltas)
backward ws bs (BackProp (al : alPrev : as) ws' ds' desiredOutput) =
    let dlNew = case ds' of
            [] -> (al ^-^ desiredOutput) ^*^ sigmoid' al
            -- delta_j(L) = (a_j(L) - y_j) * sigma'(z_j(L))
            _  -> (transpose ws' #> ds')  ^*^ sigmoid' al
            -- delta_l = ((w_l+1)^T delta_l+1) ^*^ sigma'(z_l)
        wlNew = ws -#- map (map (*learningRate)) (dlNew >< alPrev)
        -- ∂C/∂w_jk = a_k(l-1) * delta_j(l)
        blNew = bs ^-^ map (*learningRate) dlNew
        -- ∂C/∂b_j = delta_j(l)
    in (wlNew, blNew, dlNew)


class AlgBwd g f where
    algBwd :: g (BackProp -> Free f a) -> (BackProp -> Free f a)
    -- g is the current layer backpropagated over 
    -- (the BackProp -> Free f a function is what will be stored in the 'hole' of the functor)

    -- f is the type of the final network

instance (InputLayer :<: f) => AlgBwd InputLayer f where
    algBwd InputLayer = const (Op (inj InputLayer))
    -- inj because we dk what f looks like, Op because needs to be Free f a

instance (DenseLayer :<: f) => AlgBwd DenseLayer f where
    -- note that backProp :: BackProp -> Free f a
    algBwd (DenseLayer ws bs nextLayersFunc) backPropForCurr = 
        let (wlNew, blNew, dlNew) = backward ws bs backPropForCurr
            backPropForNext = BackProp {ws' = ws, -- next layer's weights now this layer's
                                ds' = dlNew, -- same for next layer's errors
                                as = tail (as backPropForCurr), -- same for outputs
                                desiredOutput = desiredOutput backPropForCurr} -- same desired output

        in Op (inj (DenseLayer wlNew blNew (nextLayersFunc backPropForNext)))
    -- eval will give us the function for backpropagating over the rest of
    -- the deeper layers (I.e. towards the input layer). We just need to 
    -- incorporate the current layer's changes into it.

{- A deeper explanation of this algBwd:

Lets take a denselayer for example: DenseLayer ws bs (rest :: Free f a)
now lets do backpropagate, i.e. eval:

eval genBwd algBwd (DenseLayer ws bs rest) = 
algBwd (DenseLayer ws bs (eval genBwd algBwd rest))

now to evaluate the outermost algBwd, we'd have to give a BackProp type to it
it will take the outermost DenseLayer (i.e. the rightmost layer), apply the
changes required to the weights and biases of that layer, and then construct
the new layer with the Op (inj ) part. 

Crucially, the function sitting in the 'hole' of the DenseLayer can now be
evaluated because we give it the updated BackProp object. So the evaluation
of the outermost layer finally triggered the evaluation of the nested layer, 
i.e. the layer behind it in the network. I (Think?) this is laziness? 

We're building a function which waits for both the output of forward prop and
the result of backprop on the layers to the right (i.e. outer nested layers),
so there is no need to worry about purity or anything.

-}

-- this following instance wasn't in the paper but need it to typecheck, just 
-- like for the AlgFwd case:
instance (AlgBwd g f, AlgBwd h f) => AlgBwd (g :+: h) f where
    algBwd (L fx) = algBwd fx
    algBwd (R gx) = algBwd gx

-- I don't even think this gen case is ever attainable because there is
-- no pure values inside our tree
genBwd :: (InputLayer :<: f) => a -> (BackProp -> Free f a) 
genBwd _ = const (Op (inj InputLayer))

backPropagate :: forall f a. (InputLayer :<: f, AlgBwd f f) => Free f a -> (BackProp -> Free f a)
backPropagate = eval genBwd algBwd
        
--------------------------- training ------------------------------------------

-- banana split property of folds: any pair of folds can be combined to a 
-- single fold which generates a pair:

pairGen :: (a -> b) -> (a -> c) -> a -> (b, c)
pairGen f g x = (f x, g x)

pairAlg :: Functor f => (f b -> b) -> (f c -> c) -> f (b, c) -> (b, c)
pairAlg algB algC fbc = (algB (fmap fst fbc), algC (fmap snd fbc))

train :: (InputLayer :<: f, AlgFwd f, AlgBwd f f) => (Values, Values) -> Free f a -> Free f a
train (inp, desOut) nn = 
    let algTrain = pairAlg algFwd algBwd
        genTrain = pairGen genFwd genBwd

        -- you don't need to run forwardPass for backwardPass to exist anymore
        -- they are just functions composed independently, then chained
        (forwardPass, backwardPass) = eval genTrain algTrain nn

        h :: [Values] -> BackProp
        h vals = BackProp vals [] [] desOut

    in (backwardPass . h . forwardPass) inp
