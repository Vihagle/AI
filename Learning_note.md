## **核心课课堂笔记**

#### 1.Framework机器学习框架

<img src="C:\Users\10024\AppData\Roaming\Typora\typora-user-images\image-20210302140317736.png" alt="image-20210302140317736" style="zoom:25%;" />

#### 2.将生活中的问题抽象成函数的过程称之为建模

#### 3.奥卡姆剃刀原理：若无必要，勿增实体

#### 4.线性回归：用一条直线去拟合数据，将其称之为线性关系

$$
y_i = \omega_1\cdot x_1+\omega_2\cdot x_2+\omega_3\cdot x_3+...+\omega_n\cdot x_i+b
=\vec\omega^T\cdot \vec x+b
$$

#### 5.线性回归损失函数：

- L1-distance:

$$
loss(\theta)=\frac{1}{2}\sum{|f_\theta(x^i)-y^i|}=\frac{1}{2}\sum{|\theta^T-y^i|}
$$

- L2-distance:

$$
loss(\theta)=\frac{1}{2}\sum{(f_\theta(x^i)-y^i)^2}=\frac{1}{2}\sum{(\theta^T-y^i)^2}
$$

#### 6.Logistic Regression的输出虽然是实数，但它表示的是分类

#### 7.逻辑回归：

$$
y = \frac{1}{1+e^-z}=\frac{1}{1+e^-(\vec{\omega}^T\cdot \vec{x}+b)};y\epsilon (0,1)
$$

- 其中参数w会引起输出y的变化，通过降低损失，使得输出y更接近真实值

#### 8.逻辑回归损失函数(交叉熵)：

$$
loss=
\begin{cases}
-y_{true}\cdot log(y_{hat}),&\text{if}\quad y_{true}=1,y_{hat}\rightarrow1,loss\rightarrow0\\-(1-y_{true})\cdot log(1-y_{hat}),&\text{if}\quad y_{true}=0,y_{hat}\rightarrow0,loss\rightarrow0
\end{cases}\\==>loss(\theta)=-\sum_{i}(y^i\cdot log(h_\theta(x^i))+(1-y^i)\cdot log(1-h_\theta(x^i)))
$$

#### 9.Q&A

- Q：为什么损失函数不写成：
  $$
  loss(\theta) = -\sum_{i}(y^i\cdot (1-h_\theta(x^i))+(1-y^i)\cdot h_\theta(x^i))
  $$

- A:原损失函数比上述损失函数具有更优的特性。对于原损失函数，当损失值远离0时，原损失函数会更新的更快，使得损失值更快地接近于0；而对于上述损失函数，其损失值更新的速度始终是一样的（线性特点：保持相对稳定）。

<img src="C:\Users\10024\AppData\Roaming\Typora\typora-user-images\image-20210302113022783.png" alt="image-20210302113022783" style="zoom:25%;" />

<img src="C:\Users\10024\AppData\Roaming\Typora\typora-user-images\image-20210302113431997.png" alt="image-20210302113431997" style="zoom:25%;" />

#### 10.香农理论：信息的包含程度和其不可预测性有关

#### 11.softmax相对于逻辑回归是一个更高维度的输出，它表示不同分类发生的概率。如果说逻辑回归是处理二分类问题，那么softmax就是处理多个二分类问题

#### 12.softmax：

$$
\sigma(z)_j = \frac{e^{z_j}}{\sum_{k=1}^{K}e^{z_k}}
$$

<img src="C:\Users\10024\AppData\Roaming\Typora\typora-user-images\image-20210302134454594.png" alt="image-20210302134454594" style="zoom: 25%;" />

- 当求出cross-entropy后，求出w,b参数的偏导，通过梯度下降，更新w,b参数的值，从而更新cross-entropy的值，使其不断接近于0，即损失函数loss接近于0

#### 13.混淆矩阵

![在这里插入图片描述](https://img-blog.csdnimg.cn/20190415235646429.png)

- Positive:指的是我们期望找的目标

$$
Accuracy = \frac{TP+TN}{TP+TN+FP+FN}\\Precision = \frac{TP}{TP+FP}\\Recall = \frac{TP}{TP+FN}\\F1 = \frac{2\cdot (Precision\cdot Recall)}{Precision+Recall}\\F2 = \frac{(1+\beta^2)\cdot (Precision\cdot Recall)}{\beta^2\cdot Precision+Recall}
$$

- F1是一个用来衡量Precision和Recall的一个指标，往往当F1很高时，Precision和Recall也很高。
- AUC是一个衡量模型的指标，当AUC趋近于1，说明该模型的分类效果越好；反之，若AUC趋近于0，说明该模型分类效果越差。
- AUC一般在数据不平衡的情况下使用。（邮件、疾病、推荐）