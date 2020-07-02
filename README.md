# Gray-scale-Hand-Written-Digits-Pytorch
# 手写数字识别Mnist的Pytorch实现
博客：https://blog.csdn.net/qq_36426650/article/details/107095059
### 一、引言（Introduction）
&emsp;&emsp;手写数字识别时经典的图像分类任务，也是经典的有监督学习任务，经常被用于测试图像的特征提取效果、分类器性能度量等方面，本文将通过应用机器学习和深度学习算法实现手写数字识别。
&emsp;&emsp;图像分类任务是指给定一张图像来对其进行分类，常见的图像分类任务有手写数字识别、猫狗分类、物品识别等，图像分类也是计算机视觉基本的分类任务。而对于手写数字识别任务来说，可以当做图像分类问题，也可以当做目标检测与分类。其中图像分类即输入整个图像并预测其类别，例如含有6的数字的图像我们希望最大化预测为6的概率；另外也可以视为目标检测任务，即提取图像中的目标并将目标提取出后进行预测，例如OCR对字符进行识别。因为手写数字是被预处理后的图像，且一张图像中只包含一个数字，因此本文则将手写数字识别视为整个图像的分类。

### 二、任务分析

**2.1 形式化描述**
&emsp;&emsp;给定一个图像数据集，其中图像记做，是一个宽为，高为，通道数为的图像，是图像对应的类标，任务的目标是寻找一个由图像数据到类别的映射。

**2.2 任务分析**
&emsp;&emsp;传统的方法是对图像进行序列化，即使用一组向量来对图像进行表示。例如本文处理的手写数字识别是灰度图像，即通道数为1，宽高均为28像素的图像，因此可以直接将图像的每个像素使用0-255整型数进行表示，并形成784维度的向量，然后使用包括SVM（支持向量机）、LR（逻辑回归）、DT（决策树）等机器学习学习多个超平面将假设空间中的样本正确的分类。另外也可以使用聚类算法，例如KNN、K-means、DBSCAN等算法自动将样本聚到10个类别上。
&emsp;&emsp;另外由于手写数字相同类别之间会存在相关性，因此也有基于图像压缩方法进行特征提取工作。通常使用PCA等降维技术将784维度的图像降维到较低空间，形成潜在的特征向量，且这些向量每一个维度之间是不相关；其次对压缩后的特征向量在使用机器学习算法进行分类，这种方法可以大大提高对重要特征的学习，忽略噪声对分类的影响。
&emsp;&emsp;随着深度学习的发展，基于深度学习神经网络可以自动地对特征进行提取以及分类成为图像分类的主流方法。常规有直接将图像对应的矩阵（或张量）进行展开后直接喂入一个前馈神经网络，或使用卷积神经网络或胶囊网络对特征进行提取，并使用一层前馈神经网络进行分类。基于深度学习的方法通常可以有效的提升分类的性能和精度。
本文主要进行了简单的对比实验，对比方案包括机器学习算法（KNN算法和决策树算法）以及深度学习算法（神经网络、CNN），并进行可视化展示。机器学习算法在实验1和2中有所介绍，因此本节主要介绍CNN网络：
&emsp;&emsp;CNN为两层卷积层以及池化层。第一层卷积层为32个大小为3*3的卷积核，第二层卷积层为64个2*2的卷积核，两个池化层均为2值最大池化，卷积网络则为3136维度的向量，输出层为两层神经网络，网络中使用正则化防止过拟合，输出部分为softmax。

### 三、数据描述
&emsp;&emsp;本次实验使用MNIST数据集进行实验，其中我们用6000张图像作为训练集，1000张图像作为测试集，图像的示例如图所示：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200702231243581.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM2NDI2NjUw,size_16,color_FFFFFF,t_70#pic_center)

&emsp;&emsp;由于数据集已经集成于一些深度学习框架中，因此我们直接使用pytorch的torchvision中的datasets库下载相应数据集。数据集包括如下几个文件，如下所示：

> train-images-idx3-ubyte	训练集图像数据二进制文件
> train-labels-idx1-ubyte	训练集对应类标二进制文件
> t10k-images-idx3-ubyte	测试集10K图像数据二进制文件
> t10k-labels-idx1-ubyte	测试集10K对应类标二进制文件

&emsp;&emsp;数据集是二进制文件，因此我们使用Pytorch读取数据集，并直接转换为张量，其次将每张图像与类标存入JSON数据中，保存为“{'img': img, 'label': label}”格式。另外我们使用min_batch方法进行训练，因此使用Pytorch提供的DataLoader方法自动生成batch。

### 四、实验
&emsp;&emsp;实验中，首先使用Sklearn调用了包括KNN（K近邻）和DT（决策树）两个算法并对训练集进行训练，其次在测试集上进行实验：。其次使用Pytorch实现只有一层隐层的神经网络以及含有多层卷积核池化层的CNN网络进行实验，程序划分为基于机器学习的训练入口函数（ml_main.py），机器学习算法类为Classifier.py；基于深度学习的训练入口函数（dl_main.py）以及模型为Network.py。使用机器学习的算法实验结果如表所示：

|算法|	精确度|
|--|--|
|KNN	|97.50%|
|DT	|75.96%|
|SVM	|97.92%|

&emsp;&emsp;在使用深度学习训练时，相关超参数如表2所示：

|超参数	|取值|
|--|--|
|Epoch|	20|
|Batch_size|	30|
|Learn_rate|	0.01|
|Hidden_size	|196|

基于深度学习的实验结果如表4所示：

|算法	|精确度|
|--|--|
|单隐层神经网络|	93.86%|
|CNN|	98.86%|

&emsp;&emsp;下图展示了在CNN模型下训练和测试过程中的损失与精度的变化曲线，以展示最优的CNN的收敛情况。
![在这里插入图片描述](https://img-blog.csdnimg.cn/2020070223172954.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM2NDI2NjUw,size_16,color_FFFFFF,t_70)

&emsp;&emsp;其中横坐标表示统计的次数，训练集的loss和acc则是每训练20个batch统计一次，测试集的loss和test则是每1000个batch记录一次。
### 五、总结
&emsp;&emsp;通过使用几个简单的机器学习和深度学习算法实现了对手写数字识别数据集MNIST的分类，可以发现机器学习算法中SVM模型表现最优，在深度学习模型中CNN分类效果最优。另外通过对模型训练过程中的收敛情况可知，当训练第3轮时模型以及基本达到收敛，因此可知模型的收敛速度和收敛性得以保证。在今后的拓展实验中，我们还将会对彩色图像以及场景图像进行识别，以提升模型的鲁棒性。


**参考文献**

[1]Simonyan K, Zisserman A. VERY DEEP CONVOLUTIONAL NETWORKS FOR LARGE-SCALE IMAGE RECOGNITION[C]. computer vision and pattern recognition, 2014.
[2]He K, Zhang X, Ren S, et al. Deep Residual Learning for Image Recognition[C]. computer vision and pattern recognition, 2016: 770-778.
[3]张黎;刘争鸣;唐军;;基于BP神经网络的手写数字识别方法的实现[J];自动化与仪器仪表;2015年06期
