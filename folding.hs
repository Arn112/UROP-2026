-- An implementation of N. Wu and M. Nguyen's paper 'Folding over Neural Networks'
-- Pretty rough so will refactor later. Just a quick prototype
import Data.Kind

-- some boilerplate:
newtype Fix (f :: Type -> Type) = In (f (Fix f))

out :: Functor f => Fix f -> f (Fix f)
out (In x) = x -- only works if there is at least one layer of structure

cata :: Functor f => (f a -> a) -> Fix f -> a
cata alg = alg . fmap (cata alg) . out

ana :: Functor f => (b -> f b) -> b -> Fix f
ana coalg = In . fmap (ana coalg) . coalg

-- defining types for weights & biases:
type Values = [Double]
type Biases = [Double]
type Weights = [[Double]]

-- vector addition:
(^+^) :: [Double] -> [Double] -> [Double]
v1 ^+^ v2 = zipWith (+) v1 v2

-- matrix-vector multiplication:
(#>) :: [[Double]] -> [Double] -> [Double]
mat #> v = map (sum . zipWith (*) v) mat

-- sigmoid function:
sigmoid :: [Double] -> [Double]
sigmoid = map (\x -> 1 / (1 + exp (-x))) 



-- The actual network:

data Layer k = InputLayer | DenseLayer Weights Biases k deriving Functor
-- k describes the previous connected layer

-- an algebra for forward propagation. [Values] is the outputs of neurons at
-- all previous layers (lines up with the outermost DenseLayer being the 
-- last layer)
algFwd :: Layer (Values -> [Values]) -> (Values -> [Values])
algFwd InputLayer = (: []) -- lambda which takes in values and puts into a list
algFws (DenseLayer wl bl forwardPass) = (\(vals : vs) -> 
    let cvals = sigmoid (wl #> vals ^+^ bl) in (cvals : vals : vs)) 
    . forwardPass

