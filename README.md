# AIsearcher_SICISP
A project using intelligent algorithms to find the optimal location for observation stations.


AIsearcher_SICISP 是一个面向武汉大学遥感院SICISP课程实习任务一搭建的空间智能优化程序。该项目旨在解决在指定区域中寻找5个最优点位建立观测站，以最大化对甲烷通量的观测程度。

## 项目结构
- [interpolation.py](https://github.com/AaronChou313/AIsearcher_SICISP/interpolation.py): 基于GDAL库实现GeoTIFF数据插值，将实验提供的0.1°空间分辨率的甲烷通量场tif文件插值为0.01°分辨率，用于后续适应度计算。
- [find.py](https://github.com/AaronChou313/AIsearcher_SICISP/find.py): 支持遗传算法（GA）、粒子群算法（PSO）和蚁群算法（ACO）三种方法搜索最优点位，并按照y=H*x（y为甲烷观测能力，即适应度，越高越好；H为足迹矩阵；x为局部甲烷通量）计算观测能力，不断优化观测站选址点位。

## 使用说明
1. 确保已安装所需的依赖库，如numpy、rasterio、gdal等。
2. 运行[interpolation.py](https://github.com/AaronChou313/AIsearcher_SICISP/interpolation.py)进行甲烷通量场tif文件的插值处理。
3. 根据需要选择运行[find.py](https://github.com/AaronChou313/AIsearcher_SICISP/find.py)中的遗传算法、粒子群算法或蚁群算法来搜索最优观测站点位。

## 算法说明
- **遗传算法 (GA)**: 使用二进制编码和Gray码转换来进行个体表示，通过交叉和变异操作进化种群，寻找最优解。
- **粒子群算法 (PSO)**: 模拟鸟群觅食行为，每个粒子代表一个可能的解决方案，通过更新速度和位置来寻找最优解。
- **蚁群算法 (ACO)**: 模仿蚂蚁寻找食物路径的行为，利用信息素浓度指导搜索过程，逐步构建出较优路径。

## 数据准备
- 提供的甲烷通量场tif文件需放置在`Data/Global_Fuel_Exploitation_Inventory_v2_2019_Total_Fuel_Exploitation.tif`路径下。
- 足迹文件应存放在`Data/footprints/`目录下，并命名为`foot_lat_lon.csv`格式。

## 结果输出
- 最优解及相关的适应度值将被记录并保存至`result/`目录下的相应子文件夹中。
- 包括适应度变化曲线图、最优解的shapefile文件以及最优解的融合足迹矩阵为tiff文件在内的结果将被生成。