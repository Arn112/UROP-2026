-- definition of forward and backward propagation functions. required to define training functions. 

module NNets.Training.Propagations where

import NNets.Common
import NNets.Layers
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
-- Hard to know how much work is actually being done when this function is called
-- It's definitely lazy, but I'm not sure whether it's deallocating the intermediate
-- results because it doesn't need them for the next layer.

backPropagate :: forall f a. (InputLayer :<: f, AlgBwd f f) => Free f a -> (BackProp -> Free f a)
backPropagate = eval genBwd algBwd