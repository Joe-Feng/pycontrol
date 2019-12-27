## 一阶方程及解法

#### 可分离变量的方程

$$
\frac{\mathrm{d}y}{\mathrm{d}x}=f(x)\mathrm{g}(y)
$$

将原方程改写为

$$
\frac{\mathrm{d}y}{\mathrm{g}(y)}=f(x)\mathrm{d}x
$$
然后两端积分
$$
\int\frac{\mathrm{d}y}{\mathrm{g}(y)}=\int f(x)\mathrm{d}x
$$
求得原方程通解



#### 齐次方程

$$
\frac{\mathrm{d}y}{\mathrm{d}x}=f(\frac{y}{x})
$$

求解该方程的方法是作变量代换$\frac{y}{x}=u$，则$y=xu$，$\frac{\mathrm{d}y}{\mathrm{d}x}=u+x\frac{\mathrm{d}u}{\mathrm{d}x}$，代入原方程化为可分离变量的方程
$$
\frac{\mathrm{d}u}{f(u)-u}=\frac{\mathrm{d}x}{x}
$$
然后求解



#### 线性方程

$$
\frac{\mathrm{d}y}{\mathrm{d}x}+P(x)y=Q(x)
$$

线性方程的通解是
$$
y=\mathrm{e}^{-\int P(x)\mathrm{d}x}(\int Q(x)\mathrm{e}^{\int P(x)\mathrm{d}x}\mathrm{d}x+C)
$$



## 可降阶的高阶微分方程

#### $y^{(n)}=f(x)$型的微分方程

原方程两端反复对$x$积分，便可求得原方程的解



#### $y^{''}=f(x,y^{'})$型的微分方程(不显含$y$)

令$y^{'}=p$，则$y^{''}=\frac{\mathrm{d}p}{\mathrm{d}x}$，代入原方程得以下一阶方程
$$
\frac{\mathrm{d}p}{\mathrm{d}x}=f(x,p)
$$


#### $y^{''}=f(y,y^{'})$型的微分方程(不显含$x$)

令$y^{'}=p$，则$y^{''}=\frac{\mathrm{d}p}{\mathrm{d}y}\frac{\mathrm{d}y}{\mathrm{d}x}=p\frac{\mathrm{d}p}{\mathrm{d}y}$，代入原方程得以下一阶方程
$$
p\frac{\mathrm{d}p}{\mathrm{d}y}=f(y,p)
$$



## 高阶线性方程

#### 线性常系数微分方程求解

1. ##### **线性常系数齐次方程求解**

* 二阶线性常系数齐次方程求解

  二阶线性常系数齐次方程
  $$
  y^{''}+py^{'}+qy=0
  $$
  的通解为

  | 特征方程$r^2+pr+q=0$的两个根$r_1$,$r_2$         | 通解                                                     |
  | :---------------------------------------------- | -------------------------------------------------------- |
  | 两个不相等的实数根$r_1$,$r_2$                   | $y=C_1\mathrm{e}^{r_1x}+C_2\mathrm{e}^{r_2x}$            |
  | 两个相等的实数根$r_1$=$r_2$                     | $y=(C_1+C_2x)\mathrm{e}^{r_1x}$                          |
  | 一对共轭复根$r_{1,2}=\alpha\pm \mathrm{i}\beta$ | $y=(C_1\cos\beta x+C_2\sin\beta x)\mathrm{e}^{\alpha x}$ |

* $n$阶线性常系数齐次方程求解

  $n$阶线性常系数齐次方程
  $$
  y^{(n)}+p_1y^{(n-1)}+p_2y^{(n-2)}+...+p_{n-1}y^{'}+p_ny=0
  $$
  的通解为

  | 特征方程的根                                     | 通解                                                         |
  | :----------------------------------------------- | ------------------------------------------------------------ |
  | 单实根$r$                                        | 对应一项 $C\mathrm{e}^{rx}$                                  |
  | $k$重实根$r$                                     | 对应$k$项 $(C_1+C_2x+...+C_kx^{k-1})\mathrm{e}^{rx}$         |
  | 一对单复根$r_{1,2}=\alpha\pm \mathrm{i}\beta$    | 对应两项 $(C_1\cos\beta x+C_2\sin\beta x)\mathrm{e}^{\alpha x}$ |
  | 一对$k$重复根$r_{1,2}=\alpha\pm \mathrm{i}\beta$ | 对应$2k$项 $[(C_1+C_2x+...+C_kx^{k-1})\cos\beta x+(D_1+D_2x+...+D_kx^{k-1})\sin\beta x]\mathrm{e}^{\alpha x}$ |



2. **线性常系数非齐次方程求解**

   二阶线性常系数非齐次方程的一般形式是
   $$
   y^{''}+py^{'}+qy=f(x)
   $$
   求其特解$y*$

   

* $f(x)=\mathrm{e}^{\lambda x}P_m(x)$型 ($\lambda$为已知常数，$P_m(x)$为$x$的$m$次多项式)

  其待定特解为
  $$
  y^*=x^k\mathrm{e}^{\lambda x}Q_m(x)
  $$
  其中$k$是特征方程根$\lambda$的重数，$Q_m(x)$是系数待定的$x$的$m$次多项式



* $f(x)=\mathrm{e}^{\lambda x}[P_{l}^{(1)}(x)\cos \omega x+P_{n}^{(2)}(x)\sin \omega x]$型 ($\lambda$为已知常数，$P_{l}^{(1)}(x)$与$P_{n}^{(2)}(x)$分别为$x$的$l$次、$m$次多项式)

  其待定特解为

$$
y^*=x^k\mathrm{e}^{\lambda x}[Q_{m}^{(1)}(x)\cos \omega x+Q_{m}^{(2)}(x)\sin \omega x]
$$

​      其中$k$是特征方程根$\lambda+i\omega$(或$\lambda-i\omega$)的重数，$Q_{m}^{(1)}(x)$与$Q_{m}^{(2)}(x)$为系数待定的$x$的$m$次多项式，$m=max\{l,n\}$



