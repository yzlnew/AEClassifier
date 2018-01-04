# 声发射信号 MFCC 特征提取及神经网络识别分类

## 使用方法

```bash
pip install -r requirements.txt
python ae_classifier.py
```

## 实现流程

1. 输入波形，进行归一化处理
1. 通过`get_features.py`获得波形的 MFCC 特征并且存入`*.npy`
1. `ae_classifier.py`读入特征并训练神经网络，并对验证集上数据进行验证。

## 原理

### 音频信号 MFCC 特征

梅尔倒谱系数（Mel-scale Frequency Cepstral Coefficients，简称MFCC）是在Mel标度频率域提取出来的倒谱参数，描述了人耳频率的非线性特性。

如断铅和压裂的两类声发射信号的典型 MFCC 特征如图。

![](dq/dq0.csv_mfcc.png)
![](yl/yl0.csv_mfcc.png)

使用`speechpy`提供的`mfcc`函数的默认参数得到了每一帧13个特征向量的均值作为分类的输入。

### 神经网络分类

对特征向量归一化之后，通过`sklearn`的`MLPClassifier`进行训练，参数如下：

- 隐藏层数：15
- 激活函数：tanh
- 最大迭代数： 20000

断铅和压裂信号验证结果（0代表断铅信号，1代表压裂信号）

```
[ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
  0.  0.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.
  1.  1.  1.  1.]
```

由于两个信号差别很大，识别正确率 100%。



