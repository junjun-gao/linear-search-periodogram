"""
#################################################################################################################
                                                                                                                
                                  基于 Periodogram 的时序相位解缠方法函数包                                                 
                                                                                                             
#################################################################################################################
#-------------------------------------Python Version 1.0.0------------------------------------------------------#
# Author           : 王家兴 
# Organization     : Aprilab@NWPU
# Date             : 2024-12-11
# Description      : 本函数包包含了基于 Periodogram 算法的InSAR时序相位解缠方法函数包,
#                    包括时序InSAR观测相位仿真模拟、参数估计、实验结果整合等函数。
# --------------------------------------------------------------------------------------------------------------#
#
# 参考文献：
    [1] [](https://www.tudelft.nl/citg/over-faculteit/afdelingen/geoscience-remote-sensing/research/lambda/lambda)下载LAMBDA software package Python implementation,Version 1.0
    [2] Ferretti A, Prati C, Rocca F. Permanent scatterers in sar interferometry[J]. IEEE Transactions on geoscience and remote sensing, 2001, 39(1): 8-20.
    [3] Van Leijen F J. Persistent scatterer interferometry based on geodetic estimation theory[D]. Delft University of Technology, 2014.
    [4] Czikhardt R, Van Der Marel H, Papco J. Gecoris: An open-source toolbox for analyzingtime series of corner reflectors in insar geodesy[J]. Remote Sensing, 2021, 13(5): 926.
#
# 主要函数包括：
#-------------------------------------------------------------------------------------------------------------------------------------------------------------#
# 1.仿真模拟函数包:phase_data_preparation --> 通过人为设置SAR参数、干涉影像数量来模拟相位数据,生成包括随机相位噪声、随机垂直基线、参数到相位的转换系数、每arc弧观测相位等。
# 2.参数估计函数包:Period_estimator       --> 基于谱估计的相位解缠方法,包括Linear-Period和Grid-Period两种方法,可以选择迭代估计。
#-------------------------------------------------------------------------------------------------------------------------------------------------------------#
#
# 注意事项：
# 1.使用本程序需要提前根据实验要求,准备一个json文件,包含有仿真模拟的卫星参数、干涉图像的参数、相位解缠的初始参数设置等。
# 例子如下：
# # {
# #    "sentinel-1":{"Bn":150,"revisit_cycle": 12,"H":693000,"incidence_angle": 38,"wavelength":0.056},
# #    "ERS":{"Bn":333,"revisit_cycle": 35,"H":780000,"incidence_angle": 23,"wavelength":0.056},
# #    "Nifg": 30,
# #    "param_simulation": {"height":10,"velocity":0.01},
# #    "param_pseudo": {"height":0,"velocity":0},
# #    "noise_level": 10,
# #    "bounds": {"dh":10,"dv":5},
# #    "sigma_y": 40,
# #    "coarse_search": {"h_range":[-10,10],"v_range":[-290,290],"step_size":[2,0.005]},
# #    "fine_search": {"h_range":[-20 ,20],"v_range":[-1450,1450],"step_size":[1,0.0001]},
# #    "h_sum_range":[-20,20],
# #    "period_est_mode": "grid-period",
# #    "grid-period":{"mode":"iterative","iterative_times":2},
# #    "linear-period":{"mode":"iterative","iterative_times":2}
# #        }
#
# 2.运行环境,使用miniconda搭建,python 3.11,numpy 1.24.2,scipy1.3.2,joblib 1.3.2

# ---------------------------------------------------------------------------------------------#
# 声明：本程序包是仅供学术研究使用，未经作者允许，不得用于商业用途。
#----------------------------------------------------------------------------------------------#
"""

import numpy as np

class phase_data_preparation:
    """观测相位仿真模拟,包括随机相位噪声、随机垂直基线、参数到相位的转换系数、每arc弧观测相位等,
    采用“线性形变+地形高度残差+随机相位噪声”的模拟方式
    """

    def __init__(self, param_file, check_times, check_num, sat_name="sentinel-1"):
        """仿真模拟参数初始化

        Parameters
        ----------
        param_file : json
            _字典文件,包含有仿真模拟的卫星参数、干涉图像的参数、相位解缠的初始参数设置_
        check_times : _int_
            _仿真模拟次数_
        check_num : _int_
            _模拟参数下标，用于控制随机种子号_
        sat_name : str, optional
            _模拟的卫星名称_, by default "sentinel-1"

        m2ph : _float_
            _形变到相位的转换系数,4pi/lambda_
        incidence_angle : _float_
            _入射角,弧度制_
        R : _float_
            _斜距_
        Bn : _float_
            _垂直基线的标准差_
        revisit_cycle : _float_
            _重访周期_
        """
        self.param_file = param_file
        self.check_times = check_times
        self.check_num = check_num
        self.m2ph = 4 * np.pi / self.param_file[sat_name]["wavelength"]
        self.incidence_angle = param_file[sat_name]["incidence_angle"] * np.pi / 180
        self.R = param_file[sat_name]["H"] / np.cos(self.incidence_angle)
        self.Bn = param_file[sat_name]["Bn"]
        self.revisit_cycle = param_file[sat_name]["revisit_cycle"]

        # print(self.R, self.Bn, self.revisit_cycle, self.incidence_angle)

    def add_radom_noise(self):
        """随即相位噪声
        note:
        通过随机种子生成随机相位噪声,随机噪声符合正态分布,均值为0,标准差为noise_level[deg]
        """
        # 随机种子
        np.random.seed(self.check_num + self.check_times)
        # 随机相位噪声
        self.noise = np.random.normal(0, np.pi * self.param_file["noise_level"] / 180, self.param_file["Nifg"])

    def add_random_baseline(self):
        """垂直基线随机生成
        note:
            通过随机种子生成随机垂直基线,随机垂直基线符合正态分布,均值为0,标准差为Bn[m]
        """
        np.random.seed(self.check_num)
        self.normal_baseline = np.random.normal(0, self.Bn, self.param_file["Nifg"])

    def par2ph(self):
        """_估计参数到相位的转换系数_
        Parameters:
        -----------
        h2ph : _float_
            _高度h到相位的转换系数,和垂直基线有关_
        v2ph : _float_
            _形变速率v到相位的转换系数,和时间基线有关_
        par2ph : _dict_
            _参数到相位的转换系数,便于后续使用_
        note:
            根据观测相位的模型function modeol 来获得参数到相位的转换系数,包括v2ph和h2ph。具体的观测相位模型参考论文第2章节
        """
        self.add_random_baseline()
        par2ph = dict()
        # h2ph
        self.h2ph = self.m2ph * self.normal_baseline / (self.R * np.sin(self.incidence_angle))
        # v2ph
        self.v2ph = self.m2ph * self.revisit_cycle * np.arange(1, self.param_file["Nifg"] + 1, 1) / 365
        par2ph["height"] = self.h2ph
        par2ph["velocity"] = self.v2ph
        self.par2ph = par2ph

    def sim_arc_phase(self):
        """_模拟每arc弧观测相位_

        note:
        通过参数到相位的转换系数和仿真模拟参数v,h生成每arc弧观测相位

        Returns
        -------
        arc_phase : _array_
            _每arc弧观测相位_
        par2ph : _dict_
            _参数到相位的转换系数:字典形式_
        """
        self.add_radom_noise()
        self.par2ph()
        phase_true = self.param_file["param_simulation"]["height"] * self.h2ph + self.param_file["param_simulation"]["velocity"] * self.v2ph + self.noise
        complex_signal = np.exp(1j * phase_true)
        arc_phase = np.angle(complex_signal)
        return arc_phase, self.par2ph

    def sim_arc_phase_a(self):
        # self.add_radom_noise()
        self.par2ph()
        phase_true = self.param_file["param_simulation"]["height"] * self.h2ph + self.param_file["param_simulation"]["velocity"] * self.v2ph
        complex_signal = np.exp(1j * phase_true)
        arc_phase = np.angle(complex_signal)
        a = np.round((arc_phase - phase_true) / (2 * np.pi))
        return arc_phase, self.par2ph, a

class Period_estimator:
    """Periodogram method for temporal phase unwrapping 基于谱估计的相位解缠方法"""

    def __init__(self, param_file, phase_obs, par2ph) -> None:
        """初始化参数

        Parameters
        ----------
        param_file : _dict_
            _字典文件,包含有仿真模拟的卫星参数、干涉图像的参数、相位解缠的初始参数设置_
        phase_obs : _ndarray_
            _仿真模拟的观测相位,被缠绕到了[-pi,pi]_
        par2ph : _dict_
            _参数v,h到相位的转换系数,维度由干涉图影像数量决定__
        fine_search : _dict_
            _细搜索参数设置:搜索范围,搜索步长_
        coarse_search : _dict_
            _粗搜索参数设置:搜索范围,搜索步长_
        h_sum_range : _list_
            _初步h搜索的参数设置:搜索范围,搜索步长_
        """
        self.param_file = param_file
        self.arc_phase = phase_obs
        self.par2ph = par2ph
        self.fine_search = self.param_file["fine_search"]
        self.coarse_search = self.param_file["coarse_search"]
        self.h_sum_range = self.param_file["h_sum_range"]

    def construct_searched_space(self):
        """构建参数搜搜空间

        Parameters
        ----------
        W_h_fine : _ndarray_
            _高度h的细搜索空间_
        W_v_fine : _ndarray_
            _形变速率v的细搜索空间_
        h_sum : _ndarray_
            _初步h搜索的参数空间_
        note:
        根据所选择的Periodogram方法,构建相应的搜索空间
        """
        # Fine search space精细搜索空间
        self.W_h_fine = np.arange(self.fine_search["h_range"][0], self.fine_search["h_range"][1] + 1, 1) * self.fine_search["step_size"][0]
        self.W_v_fine = np.arange(self.fine_search["v_range"][0], self.fine_search["v_range"][1] + 1, 1) * self.fine_search["step_size"][1]
        # 根据所选择的Periodogram方法,构建相应的搜索空间
        if self.param_file["period_est_mode"] == "linear-period":
            self.h_sum = np.arange(self.h_sum_range[0], self.h_sum_range[1] + 1, 1) * self.fine_search["step_size"][0]
        elif self.param_file["period_est_mode"] == "grid-period":
            # Coarse search space粗搜索空间
            self.W_h_coarse = np.arange(self.coarse_search["h_range"][0], self.coarse_search["h_range"][1] + 1, 1) * self.coarse_search["step_size"][0]
            self.W_v_coarse = np.arange(self.coarse_search["v_range"][0], self.coarse_search["v_range"][1] + 1, 1) * self.coarse_search["step_size"][1]
            # print(self.W_h_coarse, self.W_v_coarse)
            # print(self.W_h_fine, self.W_v_fine)

    @staticmethod
    def construct_phase_space(W_h, W_v, h2ph, v2ph):
        """构建搜索相位空间:

        Parameters
        ----------
        W_h : _type_
            _h的搜索空间_
        W_v : _type_
            _v的搜索空间_
        h2ph : _type_
            _h到相位的转换系数_
        v2ph : _type_
            _v到相位的转换系数_

        Returns
        -------
        search_phase_space : _ndarray_
            _搜索相位空间_
        note:
        grid-period方法会使用到该相位搜索空间
        """
        v2ph = v2ph.reshape(-1, 1)
        h2ph = h2ph.reshape(-1, 1)
        search_phase_space = np.kron(W_v * v2ph, np.ones((1, len(W_h)))) + np.kron(np.ones((1, len(W_v))), W_h * h2ph)
        return search_phase_space

    @staticmethod
    def Periodogram_1D_DFT(e_phase, p2ph, W):
        """1D_DFT谱估计

        Parameters
        ----------
        e_phase : _type_
            _观测相位的复数信号形式_
        p2ph : _type_
            _参数到相位的转换系数_
        W : _type_
            _参数搜索空间_

        Returns
        -------
        _type_
            _目标函数值/频谱值_
        note:
        针对于相位模型只包含1个参数的情况,根据参数搜索空间W,计算目标函数值/频谱值
        """
        # print(e_phase.shape)
        sub_e = np.exp(-1j * (p2ph.reshape(-1, 1) * W.reshape(1, -1)))  ##shape:(W_shape,Nifg))
        # print(sub_e.shape)
        xjw_dft_v = np.mean(e_phase.reshape(-1, 1) * sub_e, axis=0)  ##shape:(W_shape)
        # print(xjw_dft_v.shape)
        coh = np.abs(xjw_dft_v)
        # print(coh.shape)
        return coh

    def linear_periodogram(self):
        """Linear-Periodogram方法
        分为两步分:1.模糊估计 2.迭代精细估计
        """
        # Mean Estimation 1D_V
        # 根据h_sum来获得I_sum
        self.Periodogram_1D_h()
        # 基于I_sum进行参数v的1D-Periodogram估计
        DFT_V = self.Periodogram_1D_DFT(self.e_phase_sub_h, self.par2ph["velocity"], self.W_v_fine)
        v_est = self.W_v_fine[np.argmax(DFT_V)]
        # Fine Esitimation 1D_H
        # 根据v_est去除v的影响，获得新的相位信息
        phase_sub_v = self.arc_phase - v_est * self.par2ph["velocity"]
        e_phase_sub_v = np.exp(1j * phase_sub_v)
        # 进行参数h的1D-Periodogram估计
        DFT_H = self.Periodogram_1D_DFT(e_phase_sub_v, self.par2ph["height"], self.W_h_fine)
        h_est = self.W_h_fine[np.argmax(DFT_H)]

        # Iterative estimation
        if self.param_file["linear-period"]["mode"] == "iterative":
            for i in range(self.param_file["linear-period"]["iterative_times"]):
                # Fine Esitimation 1D_V
                # 根据h_est去除h的影响，获得新的相位信息
                phase_sub_h = self.arc_phase - h_est * self.par2ph["height"]
                e_phase_sub_h = np.exp(1j * phase_sub_h)
                # 进行参数v的1D-Periodogram高精度估计
                DFT_V = self.Periodogram_1D_DFT(e_phase_sub_h, self.par2ph["velocity"], self.W_v_fine)
                v_est = self.W_v_fine[np.argmax(DFT_V)]
                # Fine Esitimation 1D_H
                # 根据v_est去除v的影响，获得新的相位信息
                phase_sub_v = self.arc_phase - v_est * self.par2ph["velocity"]
                e_phase_sub_v = np.exp(1j * phase_sub_v)
                # 进行参数h的1D-Periodogram高精度估计
                DFT_H = self.Periodogram_1D_DFT(e_phase_sub_v, self.par2ph["height"], self.W_h_fine)
                h_est = self.W_h_fine[np.argmax(DFT_H)]
        self.param = {"height": h_est, "velocity": v_est}

    def Periodogram_1D_h(self):
        """降低h再观测相位中的影响
        note:
        根据h_sum来获得I_sum

        """
        # 从观测相位中减去h_sum对应的相位
        self.sub_phase_h = self.arc_phase - self.h_sum.reshape(-1, 1) * self.par2ph["height"].reshape(1, -1)
        # print(self.sub_phase_h)
        # 计算减去h_sum后的复数信号并在h维度上求和
        self.e_phase_sub_h = np.sum(np.exp(1j * self.sub_phase_h), axis=0)
        # print(self.arc_phase)
        # print(np.angle(self.e_phase_sub_h))
        # print(self.e_phase_sub_h.shape)

    def grid_periodogram(self, W_v, W_h):
        """Grid-Periodogram方法,基于网格搜索的Periodogram相位解缠方法

        Parameters
        ----------
        W_v : _type_
            _参数v搜索空间_
        W_h : _type_
            _参数h搜索空间_
        """
        # 根据搜索空间构建搜索相位空间
        search_phase_space = self.construct_phase_space(W_h, W_v, self.par2ph["height"], self.par2ph["velocity"])
        # 观测相位与搜索相位空间的差值
        sub_phase = self.arc_phase.reshape(-1, 1) - search_phase_space
        # 计算sub_phase并在影像数量维度上求和，获得目标函数值
        gamma_temporal = np.sum(np.exp(1j * sub_phase), axis=0, keepdims=True) / self.param_file["Nifg"]
        # 获得目标函数值最大的索引
        max_index = np.argmax(np.abs(gamma_temporal))
        # print(max_index)
        # 根据索引获得估计的参数v,h
        param_index = np.unravel_index(max_index, [len(W_v), len(W_h)], order="C")
        h_est = W_h[param_index[1]]
        v_est = W_v[param_index[0]]
        self.param = {"height": h_est, "velocity": v_est}

    def grid_periodogram_itr(self):
        """基于迭代估计的Grid-Periodogram方法
        note:
        1.粗搜索 2.细搜索 3.迭代估计
        """
        # coarse search
        self.grid_periodogram(self.W_v_coarse, self.W_h_coarse)
        h_est = self.param["height"]
        v_est = self.param["velocity"]
        # iterative fine search
        if self.param_file["grid-period"]["mode"] == "iterative":
            for i in range(self.param_file["grid-period"]["iterative_times"]):
                # Fine Esitimation 1D_V
                phase_sub_h = self.arc_phase - h_est * self.par2ph["height"]
                e_phase_sub_h = np.exp(1j * phase_sub_h)
                DFT_V = self.Periodogram_1D_DFT(e_phase_sub_h, self.par2ph["velocity"], self.W_v_fine)
                v_est = self.W_v_fine[np.argmax(DFT_V)]
                # Fine Esitimation 1D_H
                phase_sub_v = self.arc_phase - v_est * self.par2ph["velocity"]
                e_phase_sub_v = np.exp(1j * phase_sub_v)
                DFT_H = self.Periodogram_1D_DFT(e_phase_sub_v, self.par2ph["height"], self.W_h_fine)
                h_est = self.W_h_fine[np.argmax(DFT_H)]
        self.param = {"height": h_est, "velocity": v_est}

    def periodogram_estimation(self):
        self.construct_searched_space()
        if self.param_file["period_est_mode"] == "linear-period":
            self.linear_periodogram()
        elif self.param_file["period_est_mode"] == "grid-period":
            self.grid_periodogram(self.W_v_fine, self.W_h_fine)
        elif self.param_file["period_est_mode"] == "grid-period-itr":
            self.grid_periodogram_itr()
        return self.param["height"], self.param["velocity"]


def data_collect(data, changed_param_length, V_orig_length, check_times, nifg):
    """_并行实验数据整理与收集_
    note:
    本函数只是针对使用joblib进行并行实验的数据收集和整理,不是必要函数。
    你可以根据自己的需求编写自己的数据收集函数。

    Parameters
    ----------
    data : _dict_
        _joblib并行处理后返回的数据_
    changed_param_length : _int_
        _改变的参数总长度_
    V_orig_length : _int_
        _改变的v的总长度_
    check_times : _int_
        _重复试验次数_
    nifg : _int_
        _干涉图影像数量_

    Returns
    -------
    success_rate : _ndarray_
        _相位解缠参数估计成功率_
    v_est_data : _ndarray_
        _v的估计值_
    h_est_data : _ndarray_
        _h的估计值_
    a_est_data : _ndarray_
        _整数模糊度解_
    mae_v : _ndarray_
        _v的平均绝对估计误差_
    mae_h : _ndarray_
        _h的平均绝对估计误差_
    """
    success_rate = np.zeros((changed_param_length, V_orig_length))
    mae_v = np.zeros((changed_param_length, V_orig_length))
    mae_h = np.zeros((changed_param_length, V_orig_length))
    v_est_data = []
    h_est_data = []
    a_est_data = []
    for k in range(changed_param_length):
        for i in range(V_orig_length):
            success_rate[k][i] = data[k][i]["success_rate"]
            mae_v[k][i] = data[k][i]["mae_v"]
            mae_h[k][i] = data[k][i]["mae_h"]
            v_est_data.append(data[k][i]["est_data_v"].reshape(1, -1))
            h_est_data.append(data[k][i]["est_data_h"].reshape(1, -1))
            a_est_data.append(data[k][i]["a_data_1st"])
    v_est_data = np.concatenate(v_est_data, axis=0)
    h_est_data = np.concatenate(h_est_data, axis=0)
    a_est_data = np.concatenate(a_est_data, axis=0).reshape(changed_param_length, V_orig_length, check_times, nifg)
    return success_rate, v_est_data, h_est_data, a_est_data, mae_v, mae_h


def dict_collect(data_list):
    """_将多个字典合并为一个字典_

    Parameters
    ----------
    data_list : _type_
        _joblib并行实验后,返回的数据列表_

    Returns
    -------
    data : _dict_
        _合并后的数据_
    """
    # 将列中的字典合并
    data = {}
    for i in range(len(data_list)):
        data.update(data_list[i])
    return data


def lab_period(param_file, check_times, change_name, value, data_id, sat_name="sentinel-1"):
    """_实验室实验:Periodogram相位解缠方法_

    Parameters
    ----------
    param_file : _json_
        _仿真模拟的卫星参数、干涉图像参数、相位解缠算法参数设置_
    check_times : _int_
        _重复试验次数_
    change_name : _str_
        _改变的参数名称_
    value : _type_
        _改变参数对应的值_
    data_id : _int_
        _改变数据对应的索引,用于随机种子_
    sat_name : str, optional
        _模拟的SAR卫星_, by default "sentinel-1"

    Returns
    -------
    data : _dict_
        _实验数据_
    """
    # 初始化数据空间
    est_data_h = np.zeros(check_times)
    est_data_v = np.zeros(check_times)
    a_data_1st = np.zeros((check_times, param_file["Nifg"]))
    # 改变参数
    if change_name in ["velocity", "height"]:
        param_file["param_simulation"][change_name] = value
    else:
        param_file[change_name] = value
    print(param_file["param_simulation"])
    success_time = 0
    data = {}
    # 重复试验
    for i in range(check_times):
        # 生成仿真观测相位数据
        arc_phase, par2ph = phase_data_preparation(param_file, check_times, i, sat_name).sim_arc_phase()
        # periodgram 估计v,h
        h_est, v_est = Period_estimator(param_file, arc_phase, par2ph).periodogram_estimation()
        est_data_h[i] = h_est
        est_data_v[i] = v_est
        # 计算整数模糊度解
        a_data_1st[i, :] = np.round((arc_phase - v_est * par2ph["velocity"] - h_est * par2ph["height"]) / (2 * np.pi))
        # print(f"check_times={i}, h_est={h_est}, v_est={v_est}")
        # 计算成功率,成功率判定标准为估计的v,h与真实值的误差小于一定阈值
        if abs(h_est - param_file["param_simulation"]["height"]) < 0.5 and abs(v_est - param_file["param_simulation"]["velocity"]) < 0.0005:
            success_time += 1
    success_rate = success_time / check_times
    # 计算v,h的平均绝对估计误差
    mae_v = np.mean(np.abs(est_data_v - param_file["param_simulation"]["velocity"]))
    mae_h = np.mean(np.abs(est_data_h - param_file["param_simulation"]["height"]))
    data[data_id] = {"success_rate": success_rate, "est_data_h": est_data_h, "est_data_v": est_data_v, "a_data_1st": a_data_1st, "mae_v": mae_v, "mae_h": mae_h}
    return data
