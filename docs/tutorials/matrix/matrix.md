## 行列式

$$
|A^T|=|A|
$$

$$
|kA|=k^n|A|
$$

$$
|AB|=|A||B|
$$

$$
|A^n|=|A|^n
$$

$$
|A^*|=|A|^{n-1}
$$

$$
|A^{-1}|=|A|^{-1}
$$

* 如果$\lambda_i$是$A$的特征值，则
  $$
  |A|=\prod_{i=1}^n\lambda_i
  $$
  
* 如果矩阵$A$相似于矩阵$B$，即$A\sim{B}$，那么
  $$
  |A|=|B|
  $$

* **行列式$|A|$是否为零的判定**

行列式$|A|=0$ $\Leftrightarrow$ 矩阵$A$不可逆

​                          $\Leftrightarrow$ 秩 $r(A)<n$

​                          $\Leftrightarrow$ $Ax=0$有非零解

​                          $\Leftrightarrow$ $0$是矩阵$A$的特征值

​                          $\Leftrightarrow$ $A$的列(行)向量线性相关



## 转置

$$
(A+B)^T=A^T+B^T
$$

$$
(kA)^T=kA^T
$$

$$
(AB)^T=B^TA^T
$$



## 伴随矩阵

$$
AA^*=A^*A=|A|E
$$

$$
A^*=A^{-1}|A|
$$

$$
(A^*)^{-1}=(A^{-1})^*=\frac{A}{|A|}
$$

$$
\begin{equation}
r(A^*)=
\begin{cases}
n,&\mbox{若 $r(A)=n$}\\
1,&\mbox{若 $r(A)=n-1$}\\
0,&\mbox{若 $r(A)<n-1$}
\end{cases}
\end{equation}
$$



## 逆矩阵

$$
AA^{-1}=A^{-1}A=E
$$

$$
A^{-1}=\frac{A^*}{|A|}
$$

$$
(kA)^{-1}=\frac{1}{k}A^{-1}
$$

$$
(AB)^{-1}=B^{-1}A^{-1}
$$

$$
(A^n)^{-1}=(A^{-1})^n
$$

$$
(A^T)^{-1}=(A^{-1})^T
$$



## 秩

* 经初等变换，矩阵的秩不变

* 如果$A$可逆，则
  $$
  r(AB)=r(B), \quad r(BA)=r(B)
  $$

* 

$$
r(A+B)\leq r(A)+r(B)
$$

$$
r(AB)\leq min(r(A),r(B))
$$



## 对角矩阵

$$
diag[a_1,a_2,...,a_n]^n=diag[a_1^n,a_2^n,...,a_n^n]
$$

$$
diag[a_1,a_2,...,a_n]^{-1}=diag[\frac{1}{a_1},\frac{1}{a_2},...,\frac{1}{a_n}]
$$



## 线性相关

* $n$维向量组$\alpha_1,\alpha_2,...,\alpha_s$线性相关

$\Leftrightarrow$ 齐次方程组$(\alpha_1,\alpha_2,...,\alpha_s)$$\begin{bmatrix}x_1\\x_2\\.\\.\\.\\x_n\end{bmatrix}$$=0$有非零解

$\Leftrightarrow$ 秩$r(\alpha_1,\alpha_2,...,\alpha_s)<s$

* $n$个$n$维向量$\alpha_1,\alpha_2,...,\alpha_n$线性相关 $\Leftrightarrow$ 行列式$|\alpha_1,\alpha_2,...,\alpha_n|=0$

* $n+1$个$n$维向量必线性相关
* $n$维向量$\beta$可由$\alpha_1,\alpha_2,...,\alpha_m$线性表出

$\Leftrightarrow$ 非齐次方程组$x_1\alpha_1+x_2\alpha_2+...+x_m\alpha_m=\beta$有解

$\Leftrightarrow$ 秩$r(\alpha_1,\alpha_2,...,\alpha_m)=r(\alpha_1,\alpha_2,...,\alpha_m,\beta)$

* 向量组$\alpha_1,\alpha_2,...,\alpha_s$线性无关，而向量组$\alpha_1,\alpha_2,...,\alpha_s,\beta$线性相关，则向量$\beta$可以由$\alpha_1,\alpha_2,...,\alpha_s$线性表出，且表示法唯一



## 反对称矩阵

* 满足$A^T=-A$的矩阵称为反对称矩阵

* 若$A$是奇数阶反对称阵，则$|A|=0$

* 设单位长度的向量$\boldsymbol n=[x,y,z]^T$，则

$$
\boldsymbol n^{\wedge}=\begin{bmatrix}0&-z&y\\z&0&-x\\-y&x&0\end{bmatrix}
$$

$$
\boldsymbol n\boldsymbol n^T=\begin{bmatrix}x^2&xy&xz\\xy&y^2&yz\\xz&yz&z^2\end{bmatrix}
$$

$$
\boldsymbol n^{\wedge}\boldsymbol n\boldsymbol n^T=\boldsymbol 0
$$


$$
\boldsymbol n^{\wedge}\boldsymbol n^{\wedge}=\begin{bmatrix}-y^2-z^2&xy&xz\\xy&-x^2-z^2&yz\\xz&yz&-x^2-y^2\end{bmatrix}
$$

$$
\boldsymbol n^{\wedge}\boldsymbol n^{\wedge}=\boldsymbol n\boldsymbol n^T-\boldsymbol I
$$





* 若向量$\boldsymbol a$和$\boldsymbol b$维数相同，则

$$
\boldsymbol a^{\wedge}\boldsymbol b=-\boldsymbol b^{\wedge}\boldsymbol a
$$



## 正交矩阵

### 施密特正交化(正交规范化)

设向量组$\alpha_1,\alpha_2,\alpha_3$线性无关，其正交规范化方法步骤如下：

令
$$
\beta_1=\alpha_1\\
\beta_2=\alpha_2-\frac{(\alpha_2,\beta_1)}{(\beta_1,\beta_1)}\beta_1\\
\beta_3=\alpha_3-\frac{(\alpha_3,\beta_1)}{(\beta_1,\beta_1)}\beta_1-\frac{(\alpha_3,\beta_2)}{(\beta_2,\beta_2)}\beta_2
$$
则$\beta_1,\beta_2,\beta_3$两两正交



再将$\beta_1,\beta_2,\beta_3$单位化，取
$$
\gamma_1=\frac{\beta_1}{|\beta_1|},\quad\gamma_2=\frac{\beta_2}{|\beta_2|},\quad \gamma_3=\frac{\beta_3}{|\beta_3|},
$$
则$\gamma_1,\gamma_2,\gamma_3$是正交规范向量组

### 正交矩阵

* 设$A$是$n$阶矩阵，满足$AA^T=A^TA=E$，则$A$是正交矩阵

* $A$是正交矩阵 $\Leftrightarrow$ $A^T=A^{-1}$

  ​                        $\Leftrightarrow$ $A$的列(行)向量组是正交规范向量组

* 若$A$是正交矩阵，则$|A|=1$或$-1$



## 线性方程组

* 初等行变换不改变线性方程组的解



### 齐次线性方程组的解

设$A_{m\times n}x=0$

* $r(A)+$线性无关解的个数$=n$
* 若$\xi_1,\xi_2,...,\xi_{n-r}$是$Ax=0$的基础解系，则
$$
k_1\xi_1+k_2\xi_2+...+k_{n-r}\xi_{n-r}
$$
是$Ax=0$的通解



### 非齐次线性方程组的解

$A_{m\times n}x=b$的有解条件



## 特征值、特征向量

$A$是$n$阶方阵，如果对于数$\lambda$，存在非零向量$\alpha$，使得
$$
A\alpha=\lambda\alpha\quad(\alpha\neq0)
$$
成立，则称$\lambda$是$A$的特征值，$\alpha$是$A$的对应于$\lambda$的特征向量



* $(\lambda E-A)\alpha=0$

* 特征值的性质
$$
\sum_{i=1}^n\lambda_i=\sum_{i=1}^na_{ii}\\
\prod_{i=1}^n\lambda_i=|A|
$$

* 对角阵、上下三角阵的特征值，是主对角线元素



