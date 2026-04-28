## Definition
**Block diagonal decomposition** is a way to rewrite a matrix into **smaller independent blocks along the diagonal**, with **zeros everywhere else** so each block can be studied separately.

A **block diagonal matrix** looks like this:
$$  
\begin{bmatrix}  
A_1 & 0 & 0 \\ 
0 & A_2 & 0 \\ 
0 & 0 & A_3  
\end{bmatrix}  
$$ where:
- $A_1, A_2, A_3$ are **smaller matrices (blocks)**.
- All **off-diagonal blocks are zero matrices**.

The idea of **block diagonal decomposition** is:
$$  
P^{-1} A P =  
\begin{bmatrix}  
B_1 & 0 & 0 \\ 
0 & B_2 & 0 \\ 
0 & 0 & B_3  
\end{bmatrix}  
$$ 
where:
- $A$ = original matrix
- $P$ = change-of-basis matrix
- $B_i$ = smaller independent blocks
## Example
Suppose we have the matrix
$$  
A =  
\begin{bmatrix}  
2 & 1 & 0 & 0 \\ 
0 & 2 & 0 & 0 \\ 
0 & 0 & 3 & 4 \\ 
0 & 0 & 0 & 3  
\end{bmatrix}  
$$
We can see it naturally splits into two blocks:
$$  
A =  
\begin{bmatrix}  
\boxed{\begin{matrix}2 & 1 \\0 & 2\end{matrix}} & 0 \\ 
0 & \boxed{\begin{matrix}3 & 4 \\0 & 3\end{matrix}}  
\end{bmatrix}  
$$
So the blocks are:

$$  
B_1 =  
\begin{bmatrix}  
2 & 1 \\ 
0 & 2  
\end{bmatrix}, 
\quad  
B_2 =  
\begin{bmatrix}  
3 & 4 \\ 
0 & 3  
\end{bmatrix}  
$$

Thus the matrix is already in **block diagonal form**:

$$  
A = \text{diag}(B_1, B_2)  
$$