-- definition of forward and backward propagation functions.
-- required to define training functions. 

module NNets.Training.Propagations where

import NNets.Common
import NNets.Layers
-- import NNets.PrintNets
import Data.Array.Unboxed
import Data.List.NonEmpty (NonEmpty((:|)), singleton)

-- There are no pure values inside the tree, so this is never reachable.
genFwd :: a -> (Vector -> NonEmpty Vector)
genFwd = error "genFwd is invoked, but there are no pure values in the Free tree."

-- There are no pure values inside our tree, so this is never reachable.
genBwd :: (InputLayer :<: f) => a -> (BackProp -> Free f a) 
genBwd = error "genBwd is invoked, but there are no pure values in the free tree"

forwardProp :: AlgFwd f => Free f a -> (Vector -> NonEmpty Vector)
forwardProp = eval genFwd algFwd
-- I'm not sure why eval works fast in this case. I would assume that building the function
-- creates huge thunks, but it doesn't seem to show much in the benchmark. Maybe it's because
-- we only do forwardProp nn on the input, so then the laziness helps it perhaps avoid building
-- the lambda as it threads through the outputs of each step?

backPropagate :: forall f a. (InputLayer :<: f, AlgBwd f f) => Free f a -> (BackProp -> Free f a)
backPropagate = eval genBwd algBwd
