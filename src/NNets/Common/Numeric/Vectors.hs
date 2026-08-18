{-# LANGUAGE QuantifiedConstraints #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE DerivingStrategies #-}
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
    sigmoid, sigmoidInv',
    vmap, mmap, vfoldl', vhead, clipVec
) where

import Data.Array.Unboxed
import Data.Array.ST
import Data.Array.MArray
import Control.Monad.ST
import Control.Monad
import Data.Array.Base
import Numeric (showFFloat)
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
    deriving newtype (Eq, Ord, Ix, Show)

data Idx2 = Idx2 {-# UNPACK #-} !Int {-# UNPACK #-} !Int 
    deriving (Eq, Ord, Ix, Show)

newtype Vector = Vector (UArray Idx1 Double) deriving newtype (Eq)
newtype Matrix = Matrix (UArray Idx2 Double) deriving newtype (Eq)

toVector :: [Double] -> Vector
toVector v = Vector (listArray (Idx1 0, Idx1 (length v - 1)) v)

toMatrix :: [[Double]] -> Matrix
toMatrix [[]] = Matrix (listArray (Idx2 0 0, Idx2 0 (-1)) []) -- head warning go byebye
toMatrix vs@(v:_) = Matrix (listArray (Idx2 0 0, Idx2 (length vs - 1) (length v - 1)) (concat vs))

------- pretty printing functions: ----------
roundTo2dp :: [Double] -> [Double]
roundTo2dp xs = [fromInteger (round (x * 100)) / 100.0 | x <- xs]

chunksOf :: Int -> [Double] -> [[Double]]
chunksOf _ [] = []
chunksOf n xs = let (chunk, rest) = splitAt n xs
                in chunk : chunksOf n rest    

instance Show Vector where
    -- show (Vector v) = show (roundTo2dp (elems v))
    show (Vector v) = show (elems v)

instance Show Matrix where 
    -- show (Matrix m) = let (_, Idx2 _ width) = bounds m
    --                     in unlines (map (show . roundTo2dp) (chunksOf (width+1) (elems m)))
    show (Matrix m) = show (elems m)
---------------------------------------------

-- A generic zipWith function for Vector and Matrix types. Corresponds to elementwise operations.
-- f is the function to zip the elements of each with.
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
Vector v1 ^+^ Vector v2 = Vector (zipWithA (+) v1 v2)

-- vector subtraction
(^-^) :: Vector -> Vector -> Vector
Vector v1 ^-^ Vector v2 = Vector (zipWithA (-) v1 v2)

-- hadamard product
(^*^) :: Vector -> Vector -> Vector
Vector v1 ^*^ Vector v2 = Vector (zipWithA (*) v1 v2)

-- matrix subtraction
(-#-) :: Matrix -> Matrix -> Matrix
Matrix matx -#- Matrix maty = Matrix (zipWithA (-) matx maty)

-- outer product (i.e. u >< v = u x v^T)
(><) :: Vector -> Vector -> Matrix
Vector v1 >< Vector v2 = Matrix $ runSTUArray $ do 
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
Matrix mat #> Vector v = Vector $ runSTUArray $ do
    let (_, Idx2 n m) = bounds mat -- pre: m == length of v
    res <- newArray_ (Idx1 0, Idx1 n) -- res :: STUArray
    forM_ [0..n] $ \i -> do
        let loop j !acc
              | j > m = acc
              | otherwise = loop (j+1) (acc + unsafeAt v j * unsafeAt mat ((m+1)*i+j))
        unsafeWrite res i (loop 0 0.0)
    return res

transposeA :: Matrix -> Matrix
transposeA (Matrix mat) = Matrix $ runSTUArray $ do
    let (_, Idx2 n m) = bounds mat
    res <- newArray_ (Idx2 0 0, Idx2 m n)
    forM_ [0..n] $ \i -> do
        forM_ [0..m] $ \j -> do
            unsafeWrite res ((n+1)*j+i) (unsafeAt mat ((m+1)*i+j))
    return res

-- generic mapping functions for vectors and matrices. Just a wrapper around amap
vmap :: (Double -> Double) -> Vector -> Vector
vmap f (Vector v) = Vector (amap f v)

mmap :: (Double -> Double) -> Matrix -> Matrix
mmap f (Matrix v) = Matrix (amap f v)

-- sigmoid function. maps sigmoid over a vector
sigmoid :: Vector -> Vector
-- sigmoid = vmap (\x -> 1 / (1 + exp (-x)))
sigmoid = vmap (\x -> if x >= 0 then 1 / (1 + exp (-x)) else exp x / (1 + exp x))

-- sigmoidInv' x = sigma' (sigma_inv(x)), i.e. derivative at inv. point.
-- (this is because input will usually be activation a, not raw z)
sigmoidInv' :: Vector -> Vector
sigmoidInv' = vmap (\x -> x * (1 - x))

vfoldl' :: (Double -> Double -> Double) -> Double -> Vector -> Double
vfoldl' f k (Vector v) = foldlArray' f k v

-- function is unsafe, we don't really care, could make nonempty guarantees about this one
-- as well but it's mostly a use site thing for our particular example.
vhead :: Vector -> Double
vhead (Vector v) = unsafeAt v 0

-- clips the vectors' values so that for each element, |element| < x , (x > 0)
clipVec :: Double -> Vector -> Vector
clipVec x = vmap (\y -> if y > 0 then min x y else max (-x) y) 