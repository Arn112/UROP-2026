{-# LANGUAGE QuantifiedConstraints #-}
{-# LANGUAGE FlexibleContexts #-}
{-
Module: NNets.Common.Numeric.Vectors
Description: Defines a type to efficiently store and process vectors and matrices

This contains the definition of Vector and Matrix types, as well as common operations
with these types required for neural networks.
-}

module NNets.Common.Numeric.Vectors (
    toVector, toMatrix, -- user functions for creating vectors and matrices
    Vector, Matrix, 
    (^+^), (^-^), (^*^), (-#-), (><), (#>), transposeA,
    sigmoid, sigmoidInv'
) where

import Data.Array.Unboxed
import Data.Array.ST
import Data.Array.MArray
import Control.Monad.ST
import Control.Monad
import Data.Array.Base
{-
Custom index types strict in its arguments to avoid laziness.

Unpack means the ints are stored directly with constructors (not as ptrs). 
Under the hood, functionally equv. to Idx2 Int# Int#, but saves hassle of dealing with # methods.

For Idx1, when it was a data Idx1 = Idx1 {-# UNPACK #-} !Int, took like 1.5 MB memory. Replaced with
newtype, and now it's gone (total memory also decreased). Strictness isn't even allowed and the Unpack
makes no difference either. 

If unpacked only acts on constructors, then it makes sense that it does nothing for newtypes, because
there is no constructor. However, could it not just inline in the use site (for a strict type, precond.)

https://github.com/tibbe/talks/blob/master/zurihac-2015/slides.md, apparently small strict types are 
unpacked automatically, so maybe that's why removing it doesn't matter? However, it's only for strict types, 
and Int itself is not strict? Is GHC in some way optimising the Int here to be strict depending on how
we're using it, or does it just not matter because we never do any crazy calculations with the indexes anyways?
-}
newtype Idx1 = Idx1 Int
    deriving (Eq, Ord, Ix, Show)

data Idx2 = Idx2 {-# UNPACK #-} !Int {-# UNPACK #-} !Int 
    deriving (Eq, Ord, Ix, Show)

type Vector = UArray Idx1 Double
type Matrix = UArray Idx2 Double

toVector :: [Double] -> Vector
toVector v = listArray (Idx1 0, Idx1 (length v - 1)) v

toMatrix :: [[Double]] -> Matrix
toMatrix v = listArray (Idx2 0 0, Idx2 (length v - 1) (length (head v) - 1)) (concat v)

{-
A generic zipWith function for Vector and Matrix types. Corresponds to elementwise operations.
f is the function to zip the elements of each with.

Since Vector and Matrix have different index types, already prevents mixing. 
-}
{-# INLINE zipWithA #-}
zipWithA :: (Ix i, IArray a e1, IArray a e2, forall s. (MArray (STUArray s) c (ST s))) 
                => (e1 -> e2 -> c) 
                -> a i e1 
                -> a i e2 
                -> UArray i c
zipWithA f xs ys = runSTUArray $ do 
    let upperInd = rangeSize (bounds xs) - 1 
    zipped <- newArray_ (bounds xs) 
    forM_ [0..upperInd] $ \i -> do 
        let x = unsafeAt xs i
            y = unsafeAt ys i
        unsafeWrite zipped i (f x y)
    return zipped 

-- vector addition
(^+^) :: Vector -> Vector -> Vector
v1 ^+^ v2 = zipWithA (+) v1 v2

-- vector subtraction
(^-^) :: Vector -> Vector -> Vector
v1 ^-^ v2 = zipWithA (-) v1 v2

-- hadamard product
(^*^) :: Vector -> Vector -> Vector
v1 ^*^ v2 = zipWithA (*) v1 v2

-- matrix subtraction
(-#-) :: Matrix -> Matrix -> Matrix
matx -#- maty = zipWithA (-) matx maty

-- outer product (i.e. u >< v = u x v^T)
(><) :: Vector -> Vector -> Matrix
v1 >< v2 = runSTUArray $ do 
    let (_, Idx1 n) = bounds v1
        (_, Idx1 m) = bounds v2
    res <- newArray_ (Idx2 0 0, Idx2 n m)
    forM_ [0..n] $ \i -> do
        forM_ [0..m] $ \j -> do
            let x = unsafeAt v1 i
                y = unsafeAt v2 j
            unsafeWrite res ((i*m) + j) (x * y)
    return res

-- matrix-vector multiplication:
(#>) :: Matrix -> Vector -> Vector
mat #> v = runSTUArray $ do
    let (_, Idx2 n m) = bounds mat -- pre: m == length of v
    res <- newArray_ (Idx1 0, Idx1 n) -- res :: STUArray
    forM_ [0..n] $ \i -> do
        let loop j !acc
                | j > m = acc
                | otherwise = loop (j+1) (acc + unsafeAt v j * unsafeAt mat ((m+1)*i+j))
        unsafeWrite res i (loop 0 0.0)
    return res

transposeA :: Matrix -> Matrix
transposeA mat = runSTUArray $ do
    let (_, Idx2 n m) = bounds mat
    res <- newArray_ (Idx2 0 0, Idx2 m n)
    forM_ [0..n] $ \i -> do
        forM_ [0..m] $ \j -> do
            unsafeWrite res ((n+1)*j+i) (unsafeAt mat ((m+1)*i+j))
    return res

-- sigmoid function. maps sigmoid over a vector
sigmoid :: Vector -> Vector
sigmoid = amap (\x -> 1 / (1 + exp (-x))) 

-- sigmoidInv' x = sigma' (sigma_inv(x)), i.e. derivative at inv. point.
-- (this is because input will usually be activation a, not raw z)
sigmoidInv' :: Vector -> Vector
sigmoidInv' = amap (\x -> x * (1 - x))

