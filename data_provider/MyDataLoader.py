import os
import pandas as pd
from pykalman import KalmanFilter
from scipy.stats import skew, kurtosis
import quaternion
from scipy.spatial.transform import Rotation as R
from collections import Counter
import numpy as np

import warnings

warnings.filterwarnings('ignore')


class MyDataLoader:

    def __init__(self, root_path, file_list=None, limit_size=None, flag=None, window_size=64, step_size=32):

        self.max_windows_num = 0
        self.root_path = root_path
        self.window_size = window_size
        self.step_size = step_size
        self.enc_in = 0
        self.windows, self.labels, self.windows_stats = self.load_all(root_path, file_list=file_list, flag=flag)

        # 假设 self.labels 包含所有标签
        self.label_counts = Counter(self.labels)
        print("标签分布：", self.label_counts)

        self.all_IDs = list(range(len(self.windows)))
        self.num_class = len(set(self.labels))
        print("类别数：", self.num_class)

    def load_all(self, root_path, file_list=None, flag=None):

        all_data = []
        # 初始化空的DataFrame用于存储标签和trip_id的对应关系
        labels = []

        # 遍历所有驾驶员
        for driver in os.listdir(root_path):
            driver_dir = os.path.join(root_path, driver)

            # 遍历每个驾驶员的所有行程
            for trip in os.listdir(driver_dir):
                trip_dir = os.path.join(driver_dir, trip)

                # 检查当前项是否为目录，如果不是，则跳过
                if not os.path.isdir(trip_dir):
                    continue

                print("**********************************************************")
                print("读取文件夹：", trip)

                if os.path.exists(trip_dir + "/clean_data.csv"):
                    print("文件夹：", trip, "已经存在清洗后的数据！")
                    temp_data = pd.read_csv(os.path.join(trip_dir, 'clean_data.csv'))
                else:

                    os_type = trip.split('_')[3]

                    # 读取每个行程的CSV文件
                    accelerometer = self.read_with_template(os.path.join(trip_dir, 'Accelerometer.csv'),
                                                            ['time', 'seconds_elapsed', 'x', 'y', 'z'])

                    accelerometer = self.resample_time_series(accelerometer, interval='100ms')

                    orientation = self.read_with_template(os.path.join(trip_dir, 'Orientation.csv'),
                                                          ['time', 'qw', 'qx', 'qy', 'qz', 'roll', 'pitch', 'yaw'])

                    orientation = self.resample_time_series(orientation, interval='100ms')

                    gravity = self.read_with_template(os.path.join(trip_dir, 'Gravity.csv'), ['time', 'x', 'y', 'z'])

                    gravity = self.resample_time_series(gravity, interval='100ms')

                    gyroscope = self.read_with_template(os.path.join(trip_dir, 'Gyroscope.csv'),
                                                        ['time', 'x', 'y', 'z'])

                    gyroscope = self.resample_time_series(gyroscope, interval='100ms')

                    magnetometer = self.read_with_template(os.path.join(trip_dir, 'Magnetometer.csv'),
                                                           ['time', 'x', 'y', 'z'])

                    magnetometer = self.resample_time_series(magnetometer, interval='100ms')

                    gps = self.read_with_template(os.path.join(trip_dir, 'Location.csv'),
                                                  ['time', 'bearing', 'bearingAccuracy', 'speed',
                                                   'speedAccuracy', 'altitude', 'latitude', 'longitude'])

                    gps['speed'] = gps['speed'] * 3.6  # 将速度从m/s转换为km/h

                    gps = self.resample_time_series(gps, interval='100ms')

                    temp_data = self.merge_data(accelerometer, gravity, gyroscope, magnetometer, gps, orientation)

                    # 重定向
                    self.reorientation(temp_data, os_type)

                    # 对加速度进行卡尔曼滤波
                    self.data_KalmanFilter3D(temp_data, 1e-6, 1e-5)

                    # 保存清洗后的数据
                    temp_data.to_csv(os.path.join(trip_dir, 'clean_data.csv'), index=True)

                    # self.acc_clean_figure(temp_data)

                start_time = pd.to_datetime(temp_data['time'].min())
                end_time = pd.to_datetime(temp_data['time'].max())

                # 计算行程长度
                time_len = end_time - start_time

                # 删除多个与时间相关的列
                time_related_columns = ['time', 'seconds_elapsed']  # 这里添加所有您想删除的时间相关列的名称

                temp_data = temp_data.drop(time_related_columns, axis=1)

                if temp_data.isnull().any().any():
                    print("Warning: There are missing values in the data")
                    # 打印出含有空值的列
                    print("Columns with missing values:", temp_data.columns[temp_data.isnull().any()])

                    # 使用上一个值填充空值
                    temp_data.fillna(0, inplace=True)

                trip_id = trip.split('_')[0] + '_' + trip.split('_')[1]  # 假设行程ID是文件夹名称的前两个元素

                temp_data['trip_id'] = trip_id

                temp_data = self.create_sliding_windows(temp_data, self.window_size, self.step_size)

                # 将segments转化为窗口的列表
                windows = [temp_data[i] for i in range(temp_data.shape[0])]

                all_data.extend(windows)

                # 提取行程标签
                trip_label = trip.split('_')[2]  # 标签是文件夹名称的第三个元素

                # 创建一个与窗口数量相同的标签数组
                window_labels = [trip_label] * len(windows)

                labels.extend(window_labels)

                print("行程ID：", trip_id, "   行程标签：", trip_label, "    行程长度：", time_len)

                print("**********************************************************")

        print("数据读取结束！共读取", len(all_data), "次行程。")

        # 假设 windows 是之前你创建的窗口列表
        windows_stats = []

        # 遍历每个窗口
        for window in all_data:
            # 计算基本统计信息
            mean = np.mean(window, axis=0)
            std = np.std(window, axis=0)
            min_val = np.min(window, axis=0)
            max_val = np.max(window, axis=0)

            # 计算分位数
            q25 = np.percentile(window, 25, axis=0)
            median = np.median(window, axis=0)
            q75 = np.percentile(window, 75, axis=0)

            # 计算峰度和偏度
            kurt = kurtosis(window, axis=0, fisher=True)  # Fisher’s definition (normalized) default is True
            skewness = skew(window, axis=0)

            # 假设 kurt 是一个包含峰度值的 NumPy 数组
            if np.isnan(kurt).any():
                # 至少有一个元素是 NaN，将其替换为0或其他默认值
                kurt = np.nan_to_num(kurt, nan=0)
                # 或者，如果只想检查而不替换，可以这样做：

            # 假设 kurt 是一个包含峰度值的 NumPy 数组
            if np.isnan(skewness).any():
                # 至少有一个元素是 NaN，将其替换为0或其他默认值
                skewness = np.nan_to_num(skewness, nan=0)
                # 或者，如果只想检查而不替换，可以这样做：

            # 组合所有统计数据到一个列表中
            stats = [
                mean, std, min_val, max_val, q25, median, q75, kurt, skewness
            ]

            # 将统计数据转换为 numpy 数组并添加到 windows_stats 列表中
            stats_array = np.array(stats).flatten()  # 可以选择保持结构或展平
            windows_stats.append(stats_array)

        # 确定样本最大窗口数
        # self.max_windows_num = max([len(data) for data in all_data])

        return all_data, labels, windows_stats

    # 重采样
    def resample_time_series(self, data, interval):

        print("重采样开始！")

        # 假设 temp_data 是您的 DataFrame，并且 'time' 列包含了时间戳
        data['time'] = pd.to_datetime(data['time'], unit='ns')  # 转换时间戳，unit 参数根据时间戳单位调整

        # 如果时间戳原本不是 UTC，先将其本地化为 UTC，然后转换为中国时间
        data['time'] = data['time'].dt.tz_localize('UTC').dt.tz_convert('Asia/Shanghai')

        data.set_index('time', inplace=True)  # 设置时间戳为索引

        # 生成均匀时间索引
        start_time = data.index.min()
        end_time = data.index.max()

        uniform_index = pd.date_range(start=start_time, end=end_time, freq=interval)

        # 合并原始时间索引和均匀时间索引
        combined_index = data.index.union(uniform_index)

        # 重设数据索引
        data_combined = data.reindex(combined_index)

        # 进行插值
        data_interpolated = data_combined.interpolate(method='linear')

        # 筛选出只包含均匀时间索引的部分
        data_uniform = data_interpolated.reindex(uniform_index)

        data_uniform = data_uniform.reset_index()

        data_uniform.rename(columns={'index': 'time'}, inplace=True)

        print("重采样结束！")

        return data_uniform

    # 欧拉角转旋转矩阵
    def euler_to_rot(self, euler):
        r = R.from_euler('zyx', euler, degrees=True)
        rotation_matrix = r.as_matrix()
        return rotation_matrix

    def phone_to_ENU_to_NED_to_car(self, data_phone, phone_quaternion, yaw_car, os_type):

        # 使用四元数直接对手机坐标系的数据进行旋转
        data_ENU_quat = phone_quaternion * quaternion.quaternion(0, *data_phone) * phone_quaternion.inverse()
        data_ENU = np.array([data_ENU_quat.x, data_ENU_quat.y, data_ENU_quat.z])

        ## Code Availability

        # Part of the code used in this project will be made publicly available after the related research paper is
        # accepted for publication.We will update this repository with the code at that time.
        # For any inquiries, please contact workzhu@outlook.com.

        return data_car

    def reorientation(self, data, os_type):
        print("重定向开始！")

        for index, row in data.iterrows():

            acc_phone = np.matrix(row[['x_acc', 'y_acc', 'z_acc']].values).T

            gra_phone = np.matrix(row[['x_gra', 'y_gra', 'z_gra']].values).T

            phone_quaternion = quaternion.quaternion(row['qw'], row['qx'], row['qy'], row['qz'])

            ## Code Availability

            # Part of the code used in this project will be made publicly available after the related research paper is
            # accepted for publication.We will update this repository with the code at that time.
            # For any inquiries, please contact workzhu@outlook.com.

            else:
                print(f"Unexpected shape for acc_car_array at index {index}")

        print("重定向结束！")

    def data_KalmanFilter3D(self, data, process_noise=1e-5, observation_noise=1e-4):
        print("卡尔曼滤波开始！")

        # 从DataFrame中提取加速度数据
        acc_reo = data[['x_acc', 'y_acc', 'z_acc']].to_numpy()

        # 从DataFrame中提取重力数据
        gra_reo = data[['x_gra', 'y_gra', 'z_gra']].to_numpy()

        # 初始状态设为零向量 [ax, ay, az]
        initial_state_mean = np.zeros(3)

        # 初始状态协方差设为单位矩阵
        initial_state_covariance = np.eye(3)

        # 状态转移矩阵（对于加速度计数据，使用单位矩阵）
        transition_matrix = np.eye(3)

        # 观测矩阵（单位矩阵，假设所有状态都可以直接观测到）
        observation_matrix = np.eye(3)

        # 过程噪声协方差（可调参数）
        process_noise_covariance = np.eye(3) * process_noise

        # 观测噪声协方差（可调参数）
        observation_noise_covariance = np.eye(3) * observation_noise

        ## Code Availability

        # Part of the code used in this project will be made publicly available after the related research paper is
        # accepted for publication.We will update this repository with the code at that time.
        # For any inquiries, please contact workzhu@outlook.com.

        # 将过滤后的加速度数据添加到原始DataFrame中
        for i, col in enumerate(['x_gra_clean', 'y_gra_clean', 'z_gra_clean']):
            data[col] = filtered_state_means[:, i]

        print("卡尔曼滤波结束！")

        return data

        # 相同采样率10Hz数据融合

    def merge_data(self, accelerometer, gravity, gyroscope, magnetometer, gps, orientation):

        accelerometer.rename(columns={'x': 'x_acc', 'y': 'y_acc', 'z': 'z_acc'}, inplace=True)

        gravity.rename(columns={'x': 'x_gra', 'y': 'y_gra', 'z': 'z_gra'}, inplace=True)

        gyroscope.rename(columns={'x': 'x_gyro', 'y': 'y_gyro', 'z': 'z_gyro'}, inplace=True)

        magnetometer.rename(columns={'x': 'x_mag', 'y': 'y_mag', 'z': 'z_mag'}, inplace=True)

        tempdata = pd.merge_asof(accelerometer, orientation, on='time', suffixes=('_acc', '_ori'),
                                         direction="nearest")

        print('merge accelerometer and orientation,mergeData shape: ', tempdata.shape)

        tempdata = pd.merge_asof(tempdata, gravity, on='time', suffixes=('_gra', '_merge'),
                                         direction="nearest")

        print('merge gravity and mergeData,mergeData shape: ', tempdata.shape)

        tempdata = pd.merge_asof(tempdata, gyroscope, on='time', suffixes=('_gyro', '_merge'),
                                         direction="nearest")

        print('merge gyroscope and mergeData,mergeData shape: ', tempdata.shape)

        tempdata = pd.merge_asof(tempdata, magnetometer, on='time', suffixes=('_mag', '_merge'),
                                         direction="nearest")

        print('merge magnetometer and mergeData,mergeData shape: ', tempdata.shape)

        merge_data_10_hz = pd.merge_asof(tempdata, gps, on='time', suffixes=('_gps', '_merge'),
                                         direction="nearest")

        print('merge GPS and mergeData,mergeData shape: ', merge_data_10_hz.shape)

        return merge_data_10_hz

    def read_with_template(self, filename, template):
        df = pd.read_csv(filename, usecols=template)
        return df

    def slide_window(self, rows, window_size, step_size):
        '''
        函数功能：
        生成切片列表截取数据，按指定窗口宽度的50%重叠生成；
        --------------------------------------------------
        参数说明：
        rows：excel文件中的行数；
        size：窗口宽度；
        '''

        start = 0
        s_num = (rows - window_size) // step_size  # 计算滑动次数
        new_rows = window_size + (step_size * s_num)  # 为保证窗口数据完整，丢弃不足窗口宽度的采样数据

        ## Code Availability

        # Part of the code used in this project will be made publicly available after the related research paper is
        # accepted for publication.We will update this repository with the code at that time.
        # For any inquiries, please contact workzhu@outlook.com.
            start += step_size

    def create_sliding_windows(self, temp_data, window_size, step_size):
        """
        在 DataFrame 的每个样本内创建滑动窗口。

        参数:
        window_size (int): 每个窗口的大小。
        step_size (int): 创建下一个窗口时向前移动的步数。

        返回:
        windows (list): 包含所有窗口的列表。
        samples (list): 包含每个窗口对应的样本索引的列表。
        padding_masks (list): 每个窗口的填充掩码列表。
        """

        features_name = [
            'roll', 'pitch', 'yaw',
            # 'z_gra_clean', 'y_gra_clean', 'x_gra_clean',
            'z_mag', 'y_mag', 'x_mag',
            'speed', 'bearing',
            # 'altitude', 'longitude', 'latitude',
            'x_acc_clean', 'y_acc_clean', 'z_acc_clean',
            # 'x_acc', 'y_acc', 'z_acc',
            'x_gyro', 'y_gyro', 'z_gyro',
        ]

        self.enc_in = len(features_name)

        # 构造一个切片，方便填充数据
        segments = np.empty((0, window_size, len(features_name)), dtype=np.float64)

        for start, end in self.slide_window(temp_data.shape[0], window_size, step_size):  # 调用滑动窗口函数，通过yield实现滑动效果；
            temporary = []  # 每次存放各个特征的序列片段
            for feature in features_name:
                temporary.append(temp_data[feature][start:end])

            # 将数据通过stack方法堆叠成样本 shape为（none  窗口数, sw_width 窗口长度, features  特征数）;
            segments = np.vstack([segments, np.dstack(temporary)])  # 堆叠为三维数组

        print("segments shape:", segments.shape)
        return segments
