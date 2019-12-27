## 旋转矩阵和旋转向量

#### 旋转向量转换为旋转矩阵

设旋转轴是单位长度的向量$\boldsymbol n=[x,y,z]^T$，$(\sqrt{x^2+y^2+z^2}=1)$，角度为$\theta$，则转换公式为
$$
R=\cos{\theta}\boldsymbol I+(1-\cos{\theta})\boldsymbol n\boldsymbol n^T+\sin{\theta}\boldsymbol n^{\wedge} \quad(罗德里格斯公式)
$$
其中
$$
\boldsymbol n^{\wedge}=\begin{bmatrix}0&-z&y\\z&0&-x\\-y&x&0\end{bmatrix}
$$

#### 旋转矩阵转换为旋转向量

对罗德里格斯公式两边求迹
$$
tr(R)=\cos{\theta}tr(\boldsymbol I)+(1-\cos{\theta})tr(\boldsymbol n\boldsymbol n^T)+\sin{\theta}tr(\boldsymbol n^{\wedge})\\
=3\cos{\theta}+(1-\cos{\theta})\\
=1+2\cos{\theta}
$$
所以
$$
\theta=\arccos{\frac{tr(R)-1}{2}}
$$
对罗德里格斯公式两端求转置，得
$$
R^T=\cos{\theta}\boldsymbol I+(1-\cos{\theta})\boldsymbol n\boldsymbol n^T+\sin{\theta}\boldsymbol (\boldsymbol n^{\wedge})^{T}
$$
两式相减，得
$$
R-R^T=\sin{\theta}(\boldsymbol n^{\wedge}-(\boldsymbol n^{\wedge})^{T})\\
R-R^T=2\sin{\theta}\boldsymbol n^{\wedge}\\
\boldsymbol n^{\wedge}=\frac{R-R^T}{2\sin{\theta}}
$$
