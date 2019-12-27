## 李代数

#### 李代数so(3)

$SO(3)$对应的李代数记作$\phi$，则：
$$
\phi^{\wedge}=\begin{bmatrix}0&-\phi_3&\phi_2\\\phi_3&0&-\phi_1\\-\phi_2&\phi_1&0\end{bmatrix}
$$
在此定义下，两个向量$\phi_1$和$\phi_2$的李括号为：
$$
[\phi_1,\phi_2]=(\phi_1\phi_2-\phi_2\phi_1)^{\vee}
$$
$SO(3)$与$so(3)$映射关系：
$$
R=\exp(\phi^{\wedge})
$$


#### $SO(3)$上的指数映射

设$\phi$是三维向量，定义它的模长和方向，分别记作$\theta$和$\boldsymbol a$，于是有$\phi=\theta\boldsymbol a$。这里$\boldsymbol a$是长度为1的方向向量，即$||a||=1$

指数映射：
$$
\exp(\theta\boldsymbol a^{\wedge})=\cos{\theta}\boldsymbol I+(1-cos{\theta})\boldsymbol a\boldsymbol a^T+\sin{\theta}\boldsymbol a^{\wedge}
$$


## 李代数求导和扰动模型

#### BCH公式与近似形式

BCH公式：
$$
\ln(\exp(A)\exp(B))=A+B+\frac{1}{2}[A,B]+\frac{1}{3!}[A,[A,B]]-\frac{1}{3!}[B,[A,B]]+...
$$
其中$[\quad]$为李括号



考虑$SO(3)$上的李代数$\ln(\exp(\phi_1^{\wedge})\exp(\phi_2^{\wedge}))^{\vee}$，当$\phi_1$或$\phi_2$为小量时，小量二次项以上的项都可以忽略。此时BCH拥有线性近似表达：
$$
\ln(\exp(\phi_1^{\wedge})\exp(\phi_2^{\wedge}))^{\vee}\approx 
\begin{cases}
J_l(\phi_2)^{-1}\phi_1+\phi_2 &当\phi_1为小量\\
J_r(\phi_1)^{-1}\phi_2+\phi_1 &当\phi_2为小量
\end{cases}
$$
以第一个近似为例。当对一个旋转矩阵$R_2$(李代数为$\phi_2$)左乘一个微小旋转矩阵$R_1$(李代数为$\phi_1$)时，可近似地看作，在原有的李代数$\phi_2$上加了一项$J_l(\phi_2)^{-1}\phi_1$。同理，第二个近似描述了右乘一个微小位移的情况

其中：
$$
J_l=\frac{\sin{\theta}}{\theta}\boldsymbol I+(1-\frac{\sin{\theta}}{\theta})\boldsymbol a\boldsymbol a^T+\frac{1-\cos{\theta}}{\theta}\boldsymbol a^{\wedge}
$$
它的逆为：
$$
J_l^{-1}=\frac{\theta}{2}\cot{\frac{\theta}{2}}\boldsymbol I+(1-\frac{\theta}{2}\cot{\frac{\theta}{2}})\boldsymbol a\boldsymbol a^T-\frac{\theta}{2}\boldsymbol a^{\wedge}
$$
右乘雅可比：
$$
J_r(\phi)=J_l(-\phi)
$$


李群上做乘法$\Delta R\cdot R$，近似为李代数上做加法：
$$
\exp(\Delta\phi^{\wedge})\exp(\phi^{\wedge})=\exp((\phi+J_l^{-1}(\phi)\Delta \phi)^{\wedge})
$$
在李代数上做加法，近似为李群上的乘法：
$$
\exp((\phi+\Delta \phi)^{\wedge})=\exp((J_l\Delta \phi)^{\wedge})\exp(\phi^{\wedge})=\exp(\phi^{\wedge})\exp((J_r\Delta \phi)^{\wedge})
$$


#### $SO(3)$上李代数求导

假设我们对一个空间点$p$进行旋转，得到了$Rp$，现在，要计算旋转之后点的坐标相对于旋转的导数：
$$
\frac{\partial(Rp)}{\partial R}=\frac{\partial(\exp(\phi^{\wedge})p)}{\partial\phi}=(-Rp)^{\wedge}J_l
$$


#### $SO(3)$上扰动模型（左乘）

设左扰动$\Delta R$对应的李代数为$\psi$，$\Delta R=\exp(\psi^{\wedge})$，对$\psi$求导：
$$
\frac{\partial(Rp)}{\partial \psi}=-(Rp)^{\wedge}
$$


#### $SE(3)$上扰动模型（左乘）

设左扰动$\Delta T=\exp(\delta \xi^{\wedge})$，扰动项的李代数为$\delta \xi=[\delta \rho,\delta \phi]^T$，那么：
$$
\frac{\partial(Tp)}{\partial \delta \xi}=
\begin{bmatrix}
\boldsymbol I&-(Rp+t)^{\wedge}\\
\boldsymbol 0^T&\boldsymbol 0^T
\end{bmatrix}
$$
