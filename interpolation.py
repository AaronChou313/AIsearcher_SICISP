from osgeo import gdal

# 输入和输出文件路径
input_file = 'Data/Global_Fuel_Exploitation_Inventory_v2_2019_Total_Fuel_Exploitation.tif'
output_file = 'Data/interpolated_gdal.tif'

# 打开原始数据集
ds = gdal.Open(input_file)

# 设置目标分辨率
target_resolution = 0.01  # 度

# 使用 Warp 进行重采样
warp_options = gdal.WarpOptions(
    xRes=target_resolution,     # X 分辨率
    yRes=target_resolution,     # Y 分辨率（注意：GDAL 中为正值）
    resampleAlg='bilinear',     # 插值方法：bilinear (可选 nearest, cubic, etc.)
    dstSRS=ds.GetProjection(),  # 保持原有坐标系统
    creationOptions=['COMPRESS=LZW']  # 可选压缩方式
)

# 执行插值操作
gdal.Warp(output_file, ds, options=warp_options)

# 关闭数据集
ds = None