## Definition
A **change of [basis](Basis.md)** is converting the coordinates of vectors (or matrices representing linear maps) from one set of basis vectors to another using a **change-of-basis matrix**.

For some basis $b_0, b_1, ..., b_n$, the change of basis matrix will be in the form:
$$
B = 
\begin{bmatrix}  
\mid & \mid & & \mid \\ 
b_0 & b_1 & ... & b_n \\
\mid & \mid & & \mid \\ 
\end{bmatrix}
$$ such that:
$$
x_{\text{old}} = Bx_{\text{new}}, \quad x_{\text{new}} = B^{-1}x_{\text{old}}
$$
That is, $B$ converts from new to old and $B^{-1}$ converts from old to new.
### Example
Recall the previous [example](Basis.md):

$$  
v =  
\begin{bmatrix} 3 \\ 2 \end{bmatrix} =  
3e_1 + 2e_2  
$$

Define some basis vectors:

$$  
b_0 = \begin{bmatrix} 1 \\ 1 \end{bmatrix},  
\quad  
b_1 = \begin{bmatrix} 1 \\ -1 \end{bmatrix}  
$$

We observe that the change of basis matrix (B) is:

$$  
B =  
\begin{bmatrix}  
1 & 1 \\  
1 & -1 \\  
\end{bmatrix}  
$$

since its columns are the new basis vectors written in the standard basis.

Thus for coordinates $v'$ in the basis $B$,

$$  
v = Bc
$$

To find the coordinates of $v$ in the new basis we compute

$$  
v' = B^{-1}v  
$$

First compute the inverse of $B$:
$$  
B^{-1} =  
\frac{1}{-2}  
\begin{bmatrix}  
-1 & -1 \\  
-1 & 1  
\end{bmatrix} =
\begin{bmatrix}  
\frac{1}{2} & \frac{1}{2} \\  
\frac{1}{2} & -\frac{1}{2}  
\end{bmatrix}  
$$

Now compute the coordinates:
$$  
v' =  
\begin{bmatrix}  
\frac12 & \frac12 \\  
\frac12 & -\frac12  
\end{bmatrix}  
\begin{bmatrix}  
3 \\  
2  
\end{bmatrix} = 
\begin{bmatrix}  
\frac{5}{2} \\  
\frac{1}{2}  
\end{bmatrix}  
$$

Therefore the vector can be written in the new basis as

$$  
v =  
\frac{5}{2} b_0 +  
\frac{1}{2} b_1  
$$

and its coordinate representation in the basis ${b_0, b_1}$ is

$$  
[v]_B =  
\begin{bmatrix}  
\frac{5}{2} \\  
\frac{1}{2}  
\end{bmatrix}  
$$