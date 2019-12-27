## 常见函数的导数

$$
(\arctan{x})^{'}=\frac{1}{1+x^2}
$$

证明：

令$y=\tan{x}$，则$\frac{\mathrm{d}y}{\mathrm{d}x}=\frac{1}{\cos^2x}$

其反函数为$x=\arctan{y}$，则
$$
\frac{\mathrm{d}x}{\mathrm{d}y}=\cos^2x=\frac{\cos^2x}{\cos^2x+\sin^2x}=\frac{1}{1+\tan^2x}=\frac{1}{1+y^2}
$$
所以$y=\arctan{x}$的导数为
$$
\frac{\mathrm{d}y}{\mathrm{d}x}=\frac{1}{1+x^2}
$$


## 佩亚诺余项泰勒公式

#### 一元函数泰勒公式

设$f(x)$在$x=x_0$处存在$n$阶导数，则有公式
$$
f(x)=f(x_0)+f^{'}(x_0)(x-x_0)+\frac{1}{2!}f^{''}(x_0)(x-x_0)^2+...+\frac{1}{n!}f^{(n)}(x_0)(x-x_0)^n+\omicron((x-x_0)^n)
$$
其中
$$
\lim_{x\to x_0}\frac{\omicron((x-x_0)^n)}{(x-x_0)^n}=0
$$
上述公式称为在$x=x_0$处展开的具有佩亚诺余项的$n$阶泰勒公式，
$$
R_n(x)=\omicron((x-x_0)^n)
$$
称为佩亚诺余项



#### 常见函数的泰勒公式

$$
e^x=1+x+\frac{1}{2!}x^2+...+\frac{1}{n!}x^n+\omicron(x^n)\\
\sin{x}=x-\frac{1}{3!}x^3+...+\frac{(-1)^n}{(2n+1)!}x^{2n+1}+\omicron(x^{2n+2})\\
\cos{x}=1-\frac{1}{2!}x^2+...+\frac{(-1)^n}{(2n)!}x^{2n}+\omicron(x^{2n+1})
$$

#### 二元函数泰勒公式



