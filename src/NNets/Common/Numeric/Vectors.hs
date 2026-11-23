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
    toVector, toMatrix, fromVector, fromMatrix, -- user functions for creating vectors and matrices
    Vector, Matrix, Tensor(..), Filter, SpatExt, Stride,
    (^+^), (^-^), (^*^), (-#-), (><), (#>), transposeA,
    vmap, mmap, vfoldl', vhead, matToVec, mheight, mwidth, vlength,
    convolution, pool, reshape, padMatrixN, rotatePi, reversePool, (^**^),
    chunksOf
) where

import Data.Array.Unboxed
import Data.Array.ST
import Data.Array.MArray
import Control.Monad.ST
import Control.Monad
import Data.STRef
import Data.Array.Base

import Debug.Trace (trace)

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

{-
We have to go down the tensor route. No tensor would been changing between Vector and Matrix or writing all the nice matrix ops
to deal with vectors too. Tensors are just more extensible and easier than nothing.
-}
data Tensor = TVec Vector | TMat Matrix

toVector :: [Double] -> Vector
toVector v = Vector (listArray (Idx1 0, Idx1 (length v - 1)) v)

toMatrix :: [[Double]] -> Matrix
toMatrix [[]] = Matrix (listArray (Idx2 0 0, Idx2 0 (-1)) []) -- head warning go byebye
toMatrix vs@(v:_) = Matrix (listArray (Idx2 0 0, Idx2 (length vs - 1) (length v - 1)) (concat vs))

fromVector :: Vector -> [Double]
fromVector (Vector v) = elems v

fromMatrix :: Matrix -> [[Double]]
fromMatrix (Matrix m) =
  let (_, Idx2 _ width) = bounds m
   in chunksOf (width + 1) (elems m)

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

instance Show Tensor where
    show (TVec v) = show v
    show (TMat m) = show m

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

-- matrix dot product
(^**^) :: Matrix -> Matrix -> Matrix
Matrix m1 ^**^ Matrix m2 
    | bounds m1 /= bounds m2 = error "matrix bounds on zipWithA are not equal on matrix dot prod"
    | otherwise = Matrix (zipWithA (*) m1 m2)

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
    res <- newArray_ (Idx1 0, Idx1 n) -- res :: STUArray, uninitialised
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

-- clips the vectors' values so that for each element, |element| < x , (x > 0)
clipVec :: Double -> Vector -> Vector
clipVec x = vmap (\y -> if y > 0 then min x y else max (-x) y) 



------------------ New functions for CNN integration: --------------------
type Filter = Matrix
type SpatExt = Int
type Stride = Int

-- the implementation of a concat function but for Matrices to Vectors. Simply a case of 
-- changing the index type since the elements are already unstructured. Ixmap means the 
-- conversion function is called every time when indexing, but since we're processing it
-- once for construction of future layer (and maybe in backward too) then it's not too deep.
matToVec :: Matrix -> Vector
matToVec (Matrix mat) = Vector (ixmap (Idx1 0, Idx1 ((h+1)*(w+1)-1)) conversion mat)
    where
        (_, Idx2 h w) = bounds mat
        conversion :: Idx1 -> Idx2
        conversion (Idx1 i) = Idx2 (i `div` (w+1)) (i `mod` (w+1))

-- convolution operation (technically cross correlation). It is quite an expensive operation, 
-- and for now I'm going to keep it naive. There are faster implementations but for small
-- test networks this should suffice.
-- currently implemented for one filter. applies bias and ReLU before returning.
-- precondition: matrix >= filter in both dimensions
convolution :: Matrix -> Filter -> Double -> Matrix
convolution (Matrix inp) (Matrix filter) bias = Matrix $ runSTUArray $ do
    let (_, Idx2 n m) = bounds inp
        (_, Idx2 h w) = bounds filter
    
    when (n < h || m < w) $ error "filter is larger than matrix in one or both dimensions"

    res <- newArray_ (Idx2 0 0, Idx2 (n-h) (m-w)) -- n-h and m-w are indices remember
    forM_ [0..(n-h)] $ \i_offset -> do 
        forM_ [0..(m-w)] $ \j_offset -> do
            -- for every element in res, compute dot product of filter and inp and sum.
            sum <- newSTRef 0
            forM_ [0..h] $ \i -> do
                forM_ [0..w] $ \j -> do
                    let x = i_offset + i
                        y = j_offset + j
                        -- dot product:
                        value = unsafeAt inp (x * (m+1) + y) * unsafeAt filter (i * (w+1) + j)
                    modifySTRef sum (+ value) -- sum
            sum' <- readSTRef sum -- final sum
            unsafeWrite res (i_offset * (m-w+1) + j_offset) (max 0.0 (sum' + bias))
            -- applied relu and bias before returning
    -- return $ trace "finished conv successfully" res
    return res

-- max pool operation. takes the maximum in each 'window' of size spatExt x spatExt
-- preconditions: stride <= spatial extent, both dimension spacing divisible by stride
-- there is probably some prefix maximum / minimum trick here, but for now keeping it simple.
-- time complexity: O(nm)
pool :: Matrix -> SpatExt -> Stride -> Matrix
pool (Matrix inp) spatExt stride = Matrix $ runSTUArray $ do
    let (_, Idx2 n m) = bounds inp
    
    when (stride > spatExt || (n+1-spatExt) `mod` stride /= 0 || (m+1-spatExt) `mod` stride /= 0) $ 
        error $ unlines ["pooling operation invalid. ensure stride <= spatExt and both n,m \
            \ divisible by stride. Relevant values are: n, m, spatExt, stride: ", show n, 
            show m, show spatExt, show stride]
    
    let n' = (n+1 - spatExt) `div` stride -- result height index (actual height is this + 1)
        m' = (m+1 - spatExt) `div` stride -- result width index (actual width is this + 1)
    res <- newArray_ (Idx2 0 0, Idx2 n' m')
    forM_ [0..n'] $ \i -> do
        forM_ [0..m'] $ \j -> do
            -- for every single result point, find the spatext x spatext 
            -- window corresponding to that result and then take the maximum over it.
            let start_i = stride * i
                start_j = stride * j
            maxInWindow <- newSTRef $ unsafeAt inp (start_i * (m+1) + start_j)
            forM_ [0..(spatExt-1)] $ \offset_i -> do
                forM_ [0..(spatExt-1)] $ \offset_j -> do
                    let elm = unsafeAt inp ((start_i + offset_i) * (m+1) + (start_j + offset_j))
                    modifySTRef maxInWindow (`max` elm)
            windowMax <- readSTRef maxInWindow
            unsafeWrite res (i * (m' + 1) + j) windowMax
    -- return $ trace "finished pool successfully" res
    return res

-- reshape (i.e. vec to mat) takes a vector and integers n,m and turns it into 
-- an n x m matrix (not necessarily a square). Again, we aren't scared to use 
-- Ixmap, because number of accesses is small.
reshape :: Int -> Int -> Vector -> Matrix
reshape n m (Vector vec) 
    | n * m /= (k+1) = error "vector cannot be reshaped into given size"
    | otherwise = Matrix (ixmap (Idx2 0 0, Idx2 (n-1) (m-1)) conversion vec) 
    where
        (_, Idx1 k) = bounds vec
        conversion :: Idx2 -> Idx1
        conversion (Idx2 i j) = Idx1 (i * m + j)

-- pads a matrix with x layers of zeroes enveloping the entire matrix. 
-- constructing a new copy is inevitable since bounds are baked in and now we need
-- them to change. 
padMatrixN :: Int -> Matrix -> Matrix
padMatrixN x (Matrix mat) = Matrix $ runSTUArray $ do
    when (x < 0) $ error "invalid padding number. must be non-negative"
    let (_, Idx2 n m) = bounds mat
        n' = n + 2 * x
        m' = m + 2 * x
    res <- newArray (Idx2 0 0, Idx2 n' m') 0.0 -- initialises all to 0
    forM_ [0..n] $ \i -> do
        forM_ [0..m] $ \j -> do
            -- writes at (i+x, j+x) the value at (i,j)
            unsafeWrite res ((i + x) * (m' + 1) + (j + x)) (unsafeAt mat (i * (m+1) + j))
    return res


-- rotates the matrix by 180 degrees, i.e. pi radians
rotatePi :: Matrix -> Matrix
rotatePi (Matrix mat) = Matrix $ ixmap bs conversion mat
    where
        bs@(_, Idx2 n m) = bounds mat
        conversion :: Idx2 -> Idx2
        conversion (Idx2 i j) = Idx2 (n-i) (m-j)

-- height and width functions for accessing height and width properties of matrix
-- also a length property for vectors
mheight :: Matrix -> Int
mheight (Matrix mat) = let (_, Idx2 n m) = bounds mat in (n+1)

mwidth :: Matrix -> Int
mwidth (Matrix mat) = let (_, Idx2 n m) = bounds mat in (m+1)

vlength :: Vector -> Int
vlength (Vector vec) = let (_, Idx1 n) = bounds vec in (n+1)

-- backprop for max pooling: for every window, distribute the incoming gradient
-- (dLdZ) back onto whichever input position(s) achieved the max in that window
-- (everywhere else gets 0). If a window has several elements tied for the max,
-- the gradient is split equally between them. Windows can overlap so in this case,
-- we do sum instead of just replacing value, so that they can add up.
-- precondition: 'out' and 'dLdZ' have the same shape, and are consistent with
-- what (pool inp spatExt stride) would have produced.
reversePool :: Matrix -> Matrix -> Matrix -> SpatExt -> Stride -> Matrix
reversePool (Matrix inp) (Matrix out) (Matrix dLdZ) spatExt stride
    | bounds out /= bounds dLdZ = error "reversePool: 'out' and 'dLdZ' must have the same shape"
    | otherwise = Matrix $ runSTUArray $ do
        let (_, Idx2 n m)   = bounds inp
            (_, Idx2 n' m') = bounds out

        res <- newArray (Idx2 0 0, Idx2 n m) 0.0 -- ∂L/∂X, starts all zero

        forM_ [0..n'] $ \i -> do
            forM_ [0..m'] $ \j -> do
                let start_i = stride * i
                    start_j = stride * j
                    target  = unsafeAt out  (i * (m'+1) + j) -- max value this window produced
                    grad    = unsafeAt dLdZ (i * (m'+1) + j) -- gradient flowing back into it

                -- first pass to find how many positions tie for maximum and split gradient
                -- accordingly
                countRef <- newSTRef (0 :: Int)
                forM_ [0..(spatExt-1)] $ \oi ->
                    forM_ [0..(spatExt-1)] $ \oj -> do
                        let v = unsafeAt inp ((start_i+oi) * (m+1) + (start_j+oj))
                        when (v == target) $ modifySTRef' countRef (+1)
                cnt <- readSTRef countRef
                when (cnt == 0) $ error "cnt was 0, how is this even possible"
                let share = grad / fromIntegral cnt

                -- second pass, add the split gradient onto every tied position in ∂L/∂X
                forM_ [0..(spatExt-1)] $ \oi ->
                    forM_ [0..(spatExt-1)] $ \oj -> do
                        let idx = (start_i+oi) * (m+1) + (start_j+oj)
                        when (unsafeAt inp idx == target) $ do
                            v <- unsafeRead res idx
                            unsafeWrite res idx (v + share)
                            -- adding on the share rather than setting it, += behaviour
        return res