import numpy as np
import os
import rasterio
import random
from functools import lru_cache
import matplotlib.pyplot as plt
from datetime import datetime
from osgeo import gdal, ogr, osr

block_centers = [
    [(35.7, 112.4), (35.7, 112.5), (35.7, 112.6), (35.7, 112.7), (35.7, 112.8), (35.7, 112.9), (35.7, 113.0)],
    [(35.6, 112.4), (35.6, 112.5), (35.6, 112.6), (35.6, 112.7), (35.6, 112.8), (35.6, 112.9), (35.6, 113.0)],
    [(35.5, 112.4), (35.5, 112.5), (35.5, 112.6), (35.5, 112.7), (35.5, 112.8), (35.5, 112.9), (35.5, 113.0)],
    [(35.4, 112.4), (35.4, 112.5), (35.4, 112.6), (35.4, 112.7), (35.4, 112.8), (35.4, 112.9), (35.4, 113.0)],
    [(35.3, 112.4), (35.3, 112.5), (35.3, 112.6), (35.3, 112.7), (35.3, 112.8), (35.3, 112.9), (35.3, 113.0)],
    [(35.2, 112.4), (35.2, 112.5), (35.2, 112.6), (35.2, 112.7), (35.2, 112.8), (35.2, 112.9), (35.2, 113.0)],
    [(35.1, 112.4), (35.1, 112.5), (35.1, 112.6), (35.1, 112.7), (35.1, 112.8), (35.1, 112.9), (35.1, 113.0)]
]

class Processor:
    def __init__(self):
        self.best_solution = None
        self.best_fitness = float('-inf')
        self.fitness_history = []
        self.best_merged_H = None
        self.best_H_lat = None
        self.best_H_lon = None
        self.merged_H = None
        self.H_lat = None
        self.H_lon = None
    # 计算适应度
    def calc_fitness(self, small_cells):
        # 1. 获取所有足迹矩阵
        footprints = []
        for cell_row, cell_col in small_cells:
            block_lat, block_lon = self.get_block_geo(cell_row, cell_col)
            # 获取小方格足迹矩阵即足迹中心经纬度
            footprint, (lat, lon) = self.trim_footprint(cell_row, cell_col, block_lat, block_lon)
            # 添加到列表
            footprints.append((footprint, (lat, lon)))
        # 2. 将五个小方格足迹矩阵合并为大矩阵H
        self.merged_H, (self.H_lat, self.H_lon) = self.merge_footprint(footprints)
        # 3. 读取通量场并提取对应的通量子区域 x*
        tif_path = "Data/interpolated_gdal.tif"
        flux_array, (flux_lat, flux_lon), resolution = self.read_tif_flux(tif_path)
        x_star = self.get_flux_subset(flux_array, flux_lat, flux_lon, self.H_lat, self.H_lon, self.merged_H.shape, resolution)
        # 3. 计算 y* 
        y_star = self.calculate_y_star(self.merged_H, x_star)
        # 返回浓度增强值 y*
        return y_star
    # 根据小方格行列数计算大方格中心经纬度
    def get_block_geo(self, cell_row, cell_col):
        block_row = cell_row // 10
        block_col = cell_col // 10
        block_lat, block_lon = block_centers[block_row][block_col]
        return block_lat, block_lon
    # 获取小方格在大方格中的局部行列数
    def get_cell_relative_pos(self, row, col):
        return (row%10, col%10)
    # 获取足迹文件名
    def get_file_name(self, block_lat, block_lon):
        SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
        full_path = os.path.join(SCRIPT_DIR, "Data", "footprints", f"foot_{block_lat:.1f}_{block_lon:.1f}.csv")
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"足迹文件 {full_path} 不存在")
        return full_path
    # 获取并修整足迹矩阵
    def trim_footprint(self, row, col, block_lat, block_lon, footprint_size=(270, 270)):
        # 获取并加载足迹文件
        file_name = self.get_file_name(block_lat, block_lon)
        original_footprint = np.loadtxt(file_name, delimiter=',')

        # 获取小方格在大方格内的位置 (0~9, 0~9)
        sub_row, sub_col = self.get_cell_relative_pos(row, col)

        # 小方格分辨率：0.01°
        small_step = 0.01

        # 大方格中心经纬度
        center_lat, center_lon = block_lat, block_lon

        # 小方格中心经纬度 = 大方格中心 + 偏移量
        lat = center_lat - (4.5 - sub_row) * small_step
        lon = center_lon + (sub_col - 4.5) * small_step

        # 每个小方格对应足迹矩阵中 1 个像素
        lat_offset_pixel = 4.5 - sub_row
        lon_offset_pixel = sub_col - 4.5

        # 创建新的足迹矩阵
        new_footprint = np.zeros(footprint_size, dtype=np.float32)

        # 原始足迹中的裁剪区域范围
        src_start_row = int(max(lat_offset_pixel, 0))
        src_end_row = int(min(original_footprint.shape[0], footprint_size[0] + lat_offset_pixel))
        src_start_col = int(max(lon_offset_pixel, 0))
        src_end_col = int(min(original_footprint.shape[1], footprint_size[1] + lon_offset_pixel))

        # 新矩阵的目标区域范围
        dst_start_row = int(max(-lat_offset_pixel, 0))
        dst_end_row = dst_start_row + (src_end_row - src_start_row)
        dst_start_col = int(max(-lon_offset_pixel, 0))
        dst_end_col = dst_start_col + (src_end_col - src_start_col)

        # 复制数据到新矩阵中
        new_footprint[dst_start_row:dst_end_row, dst_start_col:dst_end_col] = \
            original_footprint[src_start_row:src_end_row, src_start_col:src_end_col]

        return new_footprint, (lat, lon)
    # 合并多个足迹矩阵
    def merge_footprint(self, footprints_with_centers, footprint_size=(270, 270)):
        # 设置全局分辨率
        resolution = 0.01  # 每个像素代表 0.01 度

        # 提取所有足迹的经纬度
        latitudes = [lon_lat[0] for _, lon_lat in footprints_with_centers]
        longitudes = [lon_lat[1] for _, lon_lat in footprints_with_centers]

        # 计算融合中心（所有足迹的几何中心）
        center_lat = sum(latitudes) / len(latitudes)
        center_lon = sum(longitudes) / len(longitudes)

        # 计算偏移所需的最大范围，确保能容纳所有足迹
        max_offset_lat = max(abs(lat - center_lat) for lat in latitudes)
        max_offset_lon = max(abs(lon - center_lon) for lon in longitudes)

        # 转换为像素级偏移
        offset_pixel_y = int(max_offset_lat / resolution) + footprint_size[0]
        offset_pixel_x = int(max_offset_lon / resolution) + footprint_size[1]

        # 创建足够大的融合矩阵
        merge_size = (offset_pixel_y * 2, offset_pixel_x * 2)
        merged_footprint = np.zeros(merge_size, dtype=np.float32)

        # 全局中心点在合并矩阵中的坐标（居中）
        center_x = merge_size[1] // 2
        center_y = merge_size[0] // 2

        for footprint, (lat, lon) in footprints_with_centers:
            # 计算该足迹中心距离融合中心的经纬度差值（单位：度）
            delta_lat = lat - center_lat
            delta_lon = lon - center_lon

            # 转换为像素级偏移
            offset_row = int(delta_lat / resolution)
            offset_col = int(delta_lon / resolution)

            # 确定在合并矩阵中的起始位置
            start_row = center_y + offset_row - footprint.shape[0] // 2
            start_col = center_x + offset_col - footprint.shape[1] // 2

            # 遍历每个像素，合并时取最大值
            for i in range(footprint.shape[0]):
                for j in range(footprint.shape[1]):
                    x = start_row + i
                    y = start_col + j
                    if 0 <= x < merged_footprint.shape[0] and 0 <= y < merged_footprint.shape[1]:
                        merged_footprint[x, y] = max(merged_footprint[x, y], footprint[i, j])

        return merged_footprint, (center_lat, center_lon)
    # 读取通量场tif文件
    def read_tif_flux(self, tif_path):
        with rasterio.open(tif_path) as src:
            flux_array = src.read(1).astype(np.float32)  # 假设只有一个波段
            transform = src.transform
            lat_top_left = transform[5]  # 左上角纬度
            lon_top_left = transform[2]  # 左上角经度
            resolution = transform[0]    # 分辨率（x方向）
        return flux_array, (lat_top_left, lon_top_left), resolution
    # 获取通量场子区域
    def get_flux_subset(self, flux_array, flux_lat, flux_lon, footprint_lat, footprint_lon, footprint_size, resolution=0.01):
        rows, cols = flux_array.shape
        
        # 计算足迹矩阵中心在通量场中的像素坐标
        pixel_row = int((flux_lat - footprint_lat) / resolution)
        pixel_col = int((footprint_lon - flux_lon) / resolution)

        # 裁剪通量子区域
        start_row = max(pixel_row - footprint_size[0] // 2, 0)
        end_row = min(pixel_row + footprint_size[0] // 2, rows)
        start_col = max(pixel_col - footprint_size[1] // 2, 0)
        end_col = min(pixel_col + footprint_size[1] // 2, cols)

        return flux_array[start_row:end_row, start_col:end_col]
    # 计算y*
    def calculate_y_star(self, H, x_star, epsilon=0.0):
        # 确保矩阵尺寸一致
        if H.shape != x_star.shape:
            raise ValueError(f"足迹矩阵 {H.shape} 和通量矩阵 {x_star.shape} 尺寸不一致！")

        # 计算 y* = sum(H * x*) + ε
        return np.sum(H * x_star) + epsilon
    # 缓存计算适应度（将元组格式输入缓存化，避免重复计算相同组合）
    @lru_cache(maxsize=None)
    def cached_calc_fitness(self, small_cells_tuple):
        return self.calc_fitness(small_cells_tuple)
    # 保存运行数据
    def save_run_data(self, algorithm_name, params):
        result_dir = os.path.join("result", algorithm_name)
        self.ensure_dir(result_dir)

        # 生成时间戳文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(result_dir, f"run_{timestamp}")
        os.makedirs(run_dir)

        # 保存图像
        plt.figure()
        plt.plot(range(len(self.fitness_history)), self.fitness_history, label="Best Fitness")
        plt.xlabel("Iteration")
        plt.ylabel("Fitness")
        plt.title(f"{algorithm_name} Convergence")
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(run_dir, "fitness_curve.png"))
        plt.close()

        # 保存最优解shapefile文件
        shp_dir = os.path.join(run_dir, "best_solution.shp")
        self.save_points_to_shapefile(self.best_solution, shp_dir)

        # 保存最优解的融合足迹矩阵为tiff文件
        tiff_dir = os.path.join(run_dir, "best_merged_H.tif")
        self.save_merged_H_as_geotiff(resolution=0.01, output_path=tiff_dir)

        # 保存日志文件
        log_path = os.path.join(run_dir, "run_log.txt")
        with open(log_path, "w") as f:
            f.write("=== 算法运行日志 ===\n")
            f.write(f"时间戳: {timestamp}\n\n")

            f.write("== 参数配置 ==\n")
            for key, value in params.items():
                f.write(f"{key}: {value}\n")

            f.write("\n== 迭代记录 ==\n")
            for i, fit in enumerate(self.fitness_history):
                f.write(f"Iteration {i+1}: {fit:.6f}\n")

            f.write("\n== 最优解地理坐标 ==\n")
            for row, col in self.best_solution:
                lat, lon = self.cell_to_geo(row, col)
                f.write(f"({lat:.2f}, {lon:.2f})\n")
            f.write(f"最终适应度 y*: {self.best_fitness:.6f}\n")

        print(f"[INFO] 已保存运行结果至: {run_dir}")
    # 将小方格行列号转换为地理经纬度
    def cell_to_geo(self, cell_row, cell_col):
        # 获取大方格中心经纬度
        block_lat, block_lon = self.get_block_geo(cell_row, cell_col)
        
        # 获取小方格在大方格内的相对位置 (0~9, 0~9)
        sub_row, sub_col = self.get_cell_relative_pos(cell_row, cell_col)
        
        # 小方格分辨率：0.01°
        small_step = 0.01
        
        # 计算小方格中心经纬度
        lat = block_lat - (4.5 - sub_row) * small_step
        lon = block_lon + (sub_col - 4.5) * small_step
        
        return (lat, lon)
    # 确保目标目录存在
    def ensure_dir(self, directory):
        if not os.path.exists(directory):
            os.makedirs(directory)
    # 保存最终点位为shapefile文件
    def save_points_to_shapefile(self, points, output_path):
        # 创建一个 shapefile 数据源
        driver = ogr.GetDriverByName('ESRI Shapefile')
        data_source = driver.CreateDataSource(output_path)

        # 创建一个点图层
        spatial_ref = osr.SpatialReference()
        spatial_ref.ImportFromEPSG(4326)  # WGS84 坐标系
        layer = data_source.CreateLayer('points', spatial_ref, ogr.wkbPoint)

        # 定义字段
        field_name = ogr.FieldDefn('name', ogr.OFTString)
        field_name.SetWidth(24)
        layer.CreateField(field_name)

        # 将点添加到图层中
        for i, (row, col) in enumerate(points):
            lat, lon = self.cell_to_geo(row, col)
            feature = ogr.Feature(layer.GetLayerDefn())
            feature.SetField("id", i + 1)
            geom = ogr.Geometry(ogr.wkbPoint)
            geom.AddPoint(float(lon), float(lat))  # 注意坐标顺序是经度在前，纬度在后
            feature.SetGeometry(geom)
            layer.CreateFeature(feature)
            feature.Destroy()

        # 关闭数据源
        data_source.Destroy()
    # 保存融合足迹矩阵为GeoTIFF文件
    def save_merged_H_as_geotiff(self, resolution=0.01, output_path=None):
        if output_path is None:
            output_path = os.path.join("merged_footprint.tif")

        height, width = self.best_merged_H.shape

        # 计算左上角经纬度
        top_left_lat = self.best_H_lat + (height // 2) * resolution
        top_left_lon = self.best_H_lon - (width // 2) * resolution

        # 创建 GeoTIFF 数据集
        driver = gdal.GetDriverByName("GTiff")
        dataset = driver.Create(
            output_path,
            width,
            height,
            1,
            gdal.GDT_Float32
        )

        # 设置仿射变换
        dataset.SetGeoTransform((
            top_left_lon,         # 左上角经度
            resolution,           # x像素分辨率
            0,                    # 旋转参数（无）
            top_left_lat,         # 左上角纬度
            0,                    # 旋转参数（无）
            -resolution           # y像素分辨率（负值表示向下增长）
        ))

        # 设置投影（WGS84）
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(4326)
        dataset.SetProjection(srs.ExportToWkt())

        # 写入矩阵数据
        band = dataset.GetRasterBand(1)
        band.WriteArray(self.best_merged_H)
        band.FlushCache()

        dataset = None  # 关闭数据集


class GeneticAlgorithm:
    def __init__(self, pop_size=20, generations=100, mutation_rate=0.05):
        self.pop_size = pop_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.processor = Processor()
    # 整数转Gray码
    def int_to_gray(self, n):
        return n ^ (n >> 1)
    # Gray码转整数
    def gray_to_int(self, g):
        i = g
        while g >> 1:
            i ^= g
            g >>= 1
        return i
    # 二进制编码
    def encode_coordinate(self, row, col):
        row_gray = self.int_to_gray(row)
        col_gray = self.int_to_gray(col)
        return (
            format(row_gray, '07b'),  # 7位二进制字符串
            format(col_gray, '07b')
        )
    # 二进制解码
    def decode_coordinate(self, row_bin, col_bin):
        row_gray = int(row_bin, 2)
        col_gray = int(col_bin, 2)
        return (
            self.gray_to_int(row_gray),
            self.gray_to_int(col_gray)
        )
    # 初始化个体
    def _init_individual(self):
        return [
            self.encode_coordinate(random.randint(0, 69), random.randint(0, 69))
            for _ in range(5)
        ]
    # 交叉
    def _crossover(self, parent1, parent2):
        child = []
        for (r1, c1), (r2, c2) in zip(parent1, parent2):
            if random.random() < 0.5:
                child.append((r1[:3] + r2[3:], c1[:3] + c2[3:]))
            else:
                child.append((r2[:3] + r1[3:], c2[:3] + c1[3:]))
        return child
    # 变异
    def _mutate(self, individual):
        mutated = []
        for row_bin, col_bin in individual:
            row_list = list(row_bin)
            col_list = list(col_bin)
            for i in range(len(row_list)):
                if random.random() < self.mutation_rate:
                    row_list[i] = '1' if row_list[i] == '0' else '0'
            for i in range(len(col_list)):
                if random.random() < self.mutation_rate:
                    col_list[i] = '1' if col_list[i] == '0' else '0'
            mutated.append((''.join(row_list), ''.join(col_list)))
        return mutated
    # 运行算法
    def run(self):
        # 初始化种群
        population = [self._init_individual() for _ in range(self.pop_size)]
        # 迭代进化
        for gen in range(self.generations):
            fitnesses = []
            for ind in population:
                decoded = [self.decode_coordinate(r, c) for r, c in ind]
                fit = self.processor.cached_calc_fitness(tuple(decoded))
                fitnesses.append((fit, ind))

            current_best = max(fitnesses, key=lambda x: x[0])
            if current_best[0] > self.processor.best_fitness:
                self.processor.best_fitness = current_best[0]
                self.processor.best_solution = [self.decode_coordinate(r, c) for r, c in current_best[1]]
                self.processor.best_merged_H = self.processor.merged_H
                self.processor.best_H_lat  = self.processor.H_lat
                self.processor.best_H_lon  = self.processor.H_lon

            self.processor.fitness_history.append(self.processor.best_fitness)
            print(f"[GA] Generation {gen+1}/{self.generations}, Best Fitness: {self.processor.best_fitness:.4f}")

            # 选择
            fitnesses.sort(reverse=True, key=lambda x: x[0])
            selected = [ind for _, ind in fitnesses[:self.pop_size // 2]]

            # 交叉与变异
            next_population = []
            while len(next_population) < self.pop_size:
                parent1, parent2 = random.sample(selected, 2)
                child = self._crossover(parent1, parent2)
                child = self._mutate(child)
                next_population.append(child)

            population = next_population
        
        # 保存运行数据
        params = {
            "pop_size": ga.pop_size,
            "generations": ga.generations,
            "mutation_rate": ga.mutation_rate
        }
        self.processor.save_run_data("GA", params)


class ParticleSwarmOptimization:
    def __init__(self, particles=20, iterations=50, w=0.7, c1=1.5, c2=1.5):
        self.particles = particles
        self.iterations = iterations
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.processor = Processor()

    class Particle:
        def __init__(self, w, c1, c2, processor):
            self.w = w
            self.c1 = c1
            self.c2 = c2
            self.position = [(random.randint(0, 69), random.randint(0, 69)) for _ in range(5)]
            self.velocity = [(random.uniform(-1, 1), random.uniform(-1, 1)) for _ in range(5)]
            self.best_pos = self.position.copy()
            self.processor = processor
            self.fitness = self.processor.cached_calc_fitness(tuple(self.position))

        def update_velocity(self, global_best):
            new_velocity = []
            for i in range(5):
                r1, r2 = random.random(), random.random()
                vx = self.w * self.velocity[i][0] + self.c1 * r1 * (self.best_pos[i][0] - self.position[i][0]) + self.c2 * r2 * (global_best[i][0] - self.position[i][0])
                vy = self.w * self.velocity[i][1] + self.c1 * r1 * (self.best_pos[i][1] - self.position[i][1]) + self.c2 * r2 * (global_best[i][1] - self.position[i][1])
                new_velocity.append((vx, vy))
            self.velocity = new_velocity

        def update_position(self):
            new_pos = []
            for i in range(5):
                nx = int(self.position[i][0] + self.velocity[i][0])
                ny = int(self.position[i][1] + self.velocity[i][1])
                nx = np.clip(nx, 0, 69)
                ny = np.clip(ny, 0, 69)
                new_pos.append((nx, ny))
            self.position = new_pos
            self.fitness = self.processor.cached_calc_fitness(tuple(self.position))
            if self.fitness > self.processor.cached_calc_fitness(tuple(self.best_pos)):
                self.best_pos = self.position.copy()

    def run(self):
        particles_list = [self.Particle(self.w, self.c1, self.c2, self.processor) for _ in range(self.particles)]
        self.processor.best_solution = max(particles_list, key=lambda p: p.fitness).position

        for it in range(self.iterations):
            for particle in particles_list:
                particle.update_velocity(self.processor.best_solution)
                particle.update_position()
                if particle.fitness > self.processor.cached_calc_fitness(tuple(self.processor.best_solution)):
                    self.processor.best_solution = particle.position.copy()
                    self.processor.best_merged_H = self.processor.merged_H
                    self.processor.best_H_lat  = self.processor.H_lat
                    self.processor.best_H_lon  = self.processor.H_lon
            current_best_fit = self.processor.cached_calc_fitness(tuple(self.processor.best_solution))
            self.processor.fitness_history.append(current_best_fit)
            print(f"[PSO] Iteration {it+1}/{self.iterations}, Best Fitness: {current_best_fit:.4f}")
        self.processor.best_fitness = self.processor.cached_calc_fitness(tuple(self.processor.best_solution))

        params = {
            "particles": pso.particles,
            "iterations": pso.iterations,
            "w": pso.w,
            "c1": pso.c1,
            "c2": pso.c2
        }
        self.processor.save_run_data("PSO", params)


class AntColonyOptimization:
    def __init__(self, ants=10, iterations=30, evaporation_rate=0.5, alpha=1, beta=2):
        self.ants = ants
        self.iterations = iterations
        self.evaporation_rate = evaporation_rate
        self.alpha = alpha
        self.beta = beta
        self.pheromone_grid = np.ones((70, 70))
        self.processor = Processor()

    def _select_cell(self):
        flat_probs = self.pheromone_grid.flatten()
        idx = np.random.choice(len(flat_probs), p=flat_probs / flat_probs.sum())
        return divmod(idx, 70)

    def run(self):
        for iter in range(self.iterations):
            solutions = []
            for _ in range(self.ants):
                solution = []
                visited = set()
                for _ in range(5):
                    row, col = self._select_cell()
                    while (row, col) in visited:
                        row, col = self._select_cell()
                    solution.append((row, col))
                    visited.add((row, col))
                fit = self.processor.cached_calc_fitness(tuple(solution))
                solutions.append((fit, solution))

            # 信息素蒸发
            self.pheromone_grid *= (1 - self.evaporation_rate)

            # 更新信息素
            for fit, sol in solutions:
                for r, c in sol:
                    self.pheromone_grid[r][c] += fit

            # 跟踪最优解
            current_best = max(solutions, key=lambda x: x[0])
            if current_best[0] > self.processor.best_fitness:
                self.processor.best_fitness = current_best[0]
                self.processor.best_solution = current_best[1]
                self.processor.best_merged_H = self.processor.merged_H
                self.processor.best_H_lat  = self.processor.H_lat
                self.processor.best_H_lon  = self.processor.H_lon
            self.processor.fitness_history.append(self.processor.best_fitness)

            print(f"[ACO] Iteration {iter+1}/{self.iterations}, Best Fitness: {self.processor.best_fitness:.4f}")
        params = {
            "ants": self.ants,
            "iterations": self.iterations,
            "evaporation_rate": self.evaporation_rate,
            "alpha": self.alpha,
            "beta": self.beta
        }
        self.processor.save_run_data("ACO", params)
    

if __name__ == "__main__":
    algorithm = "aco"  # ga / pso / aco

    if algorithm == "ga":
        ga = GeneticAlgorithm(pop_size=20, generations=300, mutation_rate=0.05)
        ga.run()
    elif algorithm == "pso":
        pso = ParticleSwarmOptimization(particles=40, iterations=200, w=0.7, c1=2, c2=2)
        pso.run()
    elif algorithm == "aco":
        aco = AntColonyOptimization(ants=10, iterations=200, evaporation_rate=0.5, alpha=1, beta=2)
        aco.run()
    else:
        print("不支持的算法！")
        exit()
