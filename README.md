# Periodogram算法与ILS算法时序相位解缠仿真模拟函数包使用工具书

# 1.运行环境
使用 miniconda 配置虚拟环境
[python:3.11 ,numpy:1.24.2, scipy:1.11.2,joblib:1.3.2]

# 2.使用流程
## 2.1 函数包utilis
utilis里包含两个模块 [peirod_ils模块](./utilis/period_ils.py) 和 [LAMBDA模块](./utilis/LAMBDA.py)。在使用Periodogram算法和ILS、IB算法进行仿真模拟实验时，直接调用[peirod_ils模块](./utilis/period_ils.py)即可。
### 2.2.1 period_ils模块
该模块封装了三个对象：
1.观测相位仿真模拟模块: class phase_data_preparation
2.Periodogram相关模块：class Period_estimator
3.ILS相关模块：class ILS_estimator
一般情况下，仿真模拟的实验步骤是：观测相位仿真-->时序相位解缠-->结果绘制
### 2.2.2 LAMBDA模块
该模块是荷兰代尔夫特理工大学开源的LAMBDA python package 1.0 ,
具体下载网址如下[](https://www.tudelft.nl/citg/over-faculteit/afdelingen/geoscience-remote-sensing/research/lambda/lambda)
其中包含了LAMBDA算法的代码和论文.
## 2.2 使用示例 template
该文件简单就关于时序相位解缠算法定量分析仿真模拟实验,对Periodogram算法和ILS做了两个demo文件,
具体的实验参数设置均在[仿真模拟设置文件](./template/parms_period_ils.json)中
### 2.2.1 parms_period_ils.json文件
    输入的是字典json文件，因此在使用时请删除这里的备注部分
```python
{
    "sentinel-1":{"Bn":150,"revisit_cycle": 12,"H":693000,"incidence_angle": 38,"wavelength":0.056},
    #Sentinel-1/ERS卫星参数：Bn:垂直基线标准差, revisit_cycle:重返周期, incidence_angle:入射角, wavelength:波长
    "ERS":{"Bn":333,"revisit_cycle": 35,"H":780000,"incidence_angle": 23,"wavelength":0.056},
    "Nifg": 30, # 干涉影像数量
    "param_simulation": {"height":10,"velocity":0.01},# 仿真模拟的形变速率v[m/yr],地形高度残差h[m]
    "param_pseudo": {"height":0,"velocity":0},# ILS算法里的伪估计值
    "noise_level": 10, # 相位噪声标准差单位[deg]
    "bounds": {"dh":10,"dv":5},# ILS算法中的估计参数h,v的先验方差的标准差
    "sigma_y": 40,# ILS算法中的观测相位的先验方差的标准差
    "coarse_search": {"h_range":[-10,10],"v_range":[-290,290],"step_size":[2,0.005]},# Periodogram粗搜索的参数范围和搜索步长
    "fine_search": {"h_range":[-20 ,20],"v_range":[-1450,1450],"step_size":[1,0.0001]},# Periodogram精搜索的参数范围和搜索步长
    "h_sum_range":[-20,20],# Linear-Periodogram算法中步骤1的h搜索范围
    "period_est_mode": "grid-period",# 使用哪一种Periodogram时序相位解缠算法[linear-period,grid-peirod,grid-peirod-itr]
    "grid-period":{"mode":"iterative","iterative_times":2},# 传统Peridogram的迭代模式和次数
    "linear-period":{"mode":"iterative","iterative_times":2}# Linear-Periodogram算法的迭代模式和次数
}
```
### 2.2.2 lab_line_period.py
具体使用过程参考[period_实验](./template/lab_line_period.py)
该实验脚本主要展示了使用Periodogram算法，改变模拟观测相位中的形变速率v和h,并对每组(v,h)进行多次重复实验,获得参数估计的结果(成功率、MAE等)。
**使用了并行计算加速实验,使用者可根据自己的实验改变参数(影像数量、搜索参数等)进行合理调整**
需要注意的是，```parms_period_ils.json```文件中的"period_est_mode"决定了使用的Periodogram方法，目前的```class Period_estimator```模块
包含了3种Perirogram算法,**输入对应名称字符改变使用方法**：
1.linear-period 快速的线性搜索下的Periodogram，可以通过控制"linear-period":{"mode":"iterative","iterative_times":2}来选择是否迭代估计
2.grid-period   传统的网格搜索下的Periodogram
3.grid-period-itr 粗搜索+迭代下的网格搜索Periodogram "grid-period":{"mode":"iterative","iterative_times":2},
针对于该实验脚本和并行运算,自主编写了实验数据整理与保存的函数 **period_ils.data_collect,period_ils.dict_collect**,你可以根据自己的实验要求自行编写，不是必要函数。

### 2.2.3 lab_ils.py
具体使用过程参考[period_实验](./template/lab_ils.py)
该实验脚本主要展示了使用ILS算法，改变模拟观测相位中的形变速率v和h,并对每组(v,h)进行多次重复实验,获得参数估计的结果(成功率、MAE等)。
**使用了并行计算加速实验,使用者可根据自己的实验改变参数(影像数量、搜索参数等)进行合理调整**
有关ILS算法的模块被封装成了```class ILS_estimator```类。
需要注意的是，```parms_period_ils.json```文件中有关Periodogram算法模块只有在将Periodogram+ILS结合在一起时，才起作用。
研究ILS算法有关的定量分析，实验代码可以参考```period_ils.lab_ils```函数
研究Periodogram+ILS算法有关的定量分析，实验代码可以参考```period_ils.lab_period_ils```函数
针对于该实验脚本和并行运算,自主编写了实验数据整理与保存的函数 **period_ils.data_collect,period_ils.dict_collect**,你可以根据自己的实验要求自行编写，不是必要函数。

# 3.注意事项
1.joblib并行运算的**n_job**代表了并行数量，和cpu核数有关,请按照之际的cpu核数合理设置。
2.LAMBDA的封装了ILS、IB、IR三种整数估计算法,你可以根据自己的实验调整```lambda_method.maina_float, Q_a, method=1, ncands=2)[0]```中的method值来决定使用哪一种方法。
3.真的使用的核心代码模块只有 ```class phase_data_preparation, class ILS_estimator,class Period_estimator```3个,你可以根据自己的研究内容进行深度定制。

参考文献：
[1] [](https://www.tudelft.nl/citg/over-faculteit/afdelingen/geoscience-remote-sensing/research/lambda/lambda)下载python版本
[2] Ferretti A, Prati C, Rocca F. Permanent scatterers in sar interferometry[J]. IEEE Transactions on geoscience and remote sensing, 2001, 39(1): 8-20.
[3] Van Leijen F J. Persistent scatterer interferometry based on geodetic estimation theory[D]. Delft University of Technology, 2014.
[4] Czikhardt R, Van Der Marel H, Papco J. Gecoris: An open-source toolbox for analyzingtime series of corner reflectors in insar geodesy[J]. Remote Sensing, 2021, 13(5): 926.