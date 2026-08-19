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
    vmap, mmap, vfoldl', vhead
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

-- does what you think it does
-- it's just a helper to print out numbers more compactly
roundTo2dp :: [Double] -> [Double]
roundTo2dp xs = [fromInteger (round (x * 10000)) / 10000.0 | x <- xs]

-- also just a helper to print out matrices more nicely. It was easier to write this
-- than figure out why I couldn't import Data.List.Split
chunksOf :: Int -> [Double] -> [[Double]]
chunksOf _ [] = []
chunksOf n xs = let (chunk, rest) = splitAt n xs
                in chunk : chunksOf n rest    

instance Show Vector where
    show (Vector v) = show (roundTo2dp (elems v))

instance Show Matrix where 
    show (Matrix m) = let (_, Idx2 _ width) = bounds m
                        in unlines (map (show . roundTo2dp) (chunksOf (width+1) (elems m)))
                        -- unlines :: [String] -> String, adds newlines between each



-- slight rant: how the hell do you do an if then in haskell. I don't want the else here.
-- and using maybe in the below is so overkill like I just want to use `pseq`

-- checkRangeSizes :: (Ix i, IArray a e1, IArray a e2) => a i e1 -> a i e2 -> String -> Maybe String
-- checkRangeSizes xs ys funcName = if rxs == rys then Nothing else Just errmsg
--     where
--         rxs = rangeSize (bounds xs)
--         rys = rangeSize (bounds ys)
--         errmsg = "range sizes on " ++ funcName ++ " not equal: " 
--                 ++ show rxs ++ " vs " ++ show rys



-- A generic zipWith function for Vector and Matrix types. Corresponds to elementwise operations.
-- f is the function to zip the elements of each with. We don't deal with error checking in here,
-- the service this provides is an extremely fast no fluff function. All error checks are for the 
-- callers (keep in mind it's just for debugging anyways)
{-# INLINE zipWithA #-}
zipWithA :: (Ix i, IArray a e1, IArray a e2, forall s. (MArray (STUArray s) c (ST s))) 
                => (e1 -> e2 -> c) 
                -> a i e1 
                -> a i e2 
                -> UArray i c
zipWithA f xs ys = runSTUArray $ do 
    let bds = bounds xs
        upperInd = rangeSize bds - 1 
    zipped <- newArray_ bds
    forM_ [0..upperInd] $ \i -> do 
        let x = unsafeAt xs i
            y = unsafeAt ys i
        unsafeWrite zipped i (f x y)
    return zipped 

-- vector addition
(^+^) :: Vector -> Vector -> Vector
Vector v1 ^+^ Vector v2 
    | rangeSize (bounds v1) /= rangeSize (bounds v2) = error "range sizes on zipWithA are not equal on vec add"
    | otherwise = Vector (zipWithA (+) v1 v2)

-- vector subtraction
(^-^) :: Vector -> Vector -> Vector
Vector v1 ^-^ Vector v2
    | rangeSize (bounds v1) /= rangeSize (bounds v2) = error "range sizes on zipWithA are not equal on vec subtr"
    | otherwise = Vector (zipWithA (-) v1 v2)

-- hadamard product
(^*^) :: Vector -> Vector -> Vector
Vector v1 ^*^ Vector v2 
    | rangeSize (bounds v1) /= rangeSize (bounds v2) = error "range sizes on zipWithA are not equal on hadamard prod"
    | otherwise = Vector (zipWithA (*) v1 v2)

-- matrix subtraction
(-#-) :: Matrix -> Matrix -> Matrix
Matrix matx -#- Matrix maty 
    | rangeSize (bounds matx) /= rangeSize (bounds maty) = error "range sizes on zipWithA are not equal on mat subtr"
    | otherwise = Matrix (zipWithA (-) matx maty)

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
            unsafeWrite res ((i*(m+1)) + j) (x * y)
    return res

-- matrix-vector multiplication:
-- the bounds check per matrix is not really a huge performance hit; keep it
-- until end of project then remove because this was catching bugs
(#>) :: Matrix -> Vector -> Vector
Matrix mat #> Vector v = Vector $ runSTUArray $ do
    let (_, Idx2 n m) = bounds mat -- pre: m == length of v
    when (m + 1 /= rangeSize (bounds v)) $ 
        error $ "dimension error in matrix multiplication. Matrix has dimension " 
                ++ show (n+1) ++ ", " ++ show (m+1) ++ ", vector has " ++ show (rangeSize (bounds v))
    res <- newArray_ (Idx1 0, Idx1 n) -- res :: STUArray
    forM_ [0..n] $ \i -> do
        let loop j !acc
              | j > m = acc
              | otherwise = loop (j+1) (acc + unsafeAt v j * unsafeAt mat ((m+1)*i + j))
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

vfoldl' :: (Double -> Double -> Double) -> Double -> Vector -> Double
vfoldl' f k (Vector v) = foldlArray' f k v

-- function is unsafe, we don't really care, could make nonempty guarantees about this one
-- as well but it's mostly a use site thing for our particular example.
vhead :: Vector -> Double
vhead (Vector v) = unsafeAt v 0
