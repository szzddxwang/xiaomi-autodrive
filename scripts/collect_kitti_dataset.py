import argparse
import math
import queue
import random
from pathlib import Path
import carla
import numpy as np

def make_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)  # 如果目录不存在就创建，如果已经存在就无视并继续运行


def build_camera_intrinsic(width: int, height: int, fov: float) -> np.ndarray:  # 定义计算相机内参矩阵的函数
    focal = width / (2.0 * math.tan(math.radians(fov) / 2.0))  # 根据图像宽度和 FOV 计算焦距

    k = np.identity(3)  # 创建 3×3 单位矩阵，作为相机内参矩阵 K
    k[0, 0] = focal  # 设置 x 方向焦距
    k[1, 1] = focal  # 设置 y 方向焦距
    k[0, 2] = width / 2.0  # 设置图像中心点横坐标
    k[1, 2] = height / 2.0  # 设置图像中心点纵坐标

    return k  # 返回相机内参矩阵


def get_image_point(location: carla.Location, k: np.ndarray, world_to_camera: np.ndarray):
    point = np.array([location.x, location.y, location.z, 1.0]) # 把 CARLA 的 Location 转换成齐次坐标
    point_camera = np.dot(world_to_camera, point)

    x = point_camera[1]  # y 对应图像横向坐标
    y = -point_camera[2]  # z 取负后对应图像纵向坐标
    z = point_camera[0]  # x 表示相机前方深度

    if z <= 0.1:  # 如果点在相机后方或距离太近
        return None

    point_img = np.dot(k, np.array([x, y, z]))  # 将 3D 点投影到图像平面
    point_img[0] /= point_img[2]
    point_img[1] /= point_img[2]  # 归一化

    return point_img[0], point_img[1], z


def get_2d_bbox(actor: carla.Actor, camera: carla.Sensor, k: np.ndarray, image_w: int, image_h: int):  # Bounding box 部分
    try:
        bbox = actor.bounding_box  # 获取 3D bounding box
        vertices = bbox.get_world_vertices(actor.get_transform())  # 获取8个世界坐标顶点
        world_to_camera = np.array(camera.get_transform().get_inverse_matrix())  # 获取世界坐标到相机坐标的变换矩阵
    except Exception:
        return None

    points = []  # 保存投影后的 2D 点
    depths = []  # 保存每个投影点的深度

    for vertex in vertices:  # 遍历 bounding box 的 8 个顶点
        projected = get_image_point(vertex, k, world_to_camera)  # 将当前 3D 顶点投影到 2D 图像

        if projected is not None:
            u, v, depth = projected
            points.append((u, v))  # 投影点坐标
            depths.append(depth)  # 投影点深度

    if len(points) < 4:
        return None

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]

    xmin = max(0, min(xs))  # 计算 2D 框左边界
    ymin = max(0, min(ys))  # 上边界
    xmax = min(image_w - 1, max(xs))  # 右边界
    ymax = min(image_h - 1, max(ys))  # 下边界

    if xmax <= xmin or ymax <= ymin:
        return None

    area = (xmax - xmin) * (ymax - ymin)  # 计算 2D bounding box 面积

    if area < 100:
        return None

    return xmin, ymin, xmax, ymax, min(depths)


def carla_type_to_kitti(actor: carla.Actor) -> str:
    type_id = actor.type_id.lower()

    if type_id.startswith("sensor."):  # 如果 actor 是相机、LiDAR 等传感器
        return "ignore"

    if type_id.startswith("controller."):  # 如果 actor 是行人控制器等控制器
        return "ignore"

    if "walker" in type_id or "pedestrian" in type_id:  # 如果 actor 是行人
        return "Pedestrian"

    if "vehicle" in type_id:  # 如果 actor 是车
        return "Car"

    return "other"  # 其他类型为 other


def write_calib_file(path: Path, k: np.ndarray):  # KITTI-like
    p2 = np.zeros((3, 4))  # 创建 3×4 投影矩阵 P2
    p2[:3, :3] = k  # 将 3×3 相机内参矩阵放入 P2 左侧

    with open(path, "w", encoding="utf-8") as f:  # 打开标定文件准备写入
        f.write("P0: " + " ".join(["0" for _ in range(12)]) + "\n")  # 写入 P0
        f.write("P1: " + " ".join(["0" for _ in range(12)]) + "\n")  # 写入 P1
        f.write("P2: " + " ".join([f"{x:.6f}" for x in p2.reshape(-1)]) + "\n")  # 写入 P2 相机投影矩阵
        f.write("P3: " + " ".join(["0" for _ in range(12)]) + "\n")  # 写入 P3
        f.write("R0_rect: 1 0 0 0 1 0 0 0 1\n")  # 写入矫正矩阵，当前使用单位矩阵
        f.write("Tr_velo_to_cam: " + " ".join(["0" for _ in range(12)]) + "\n")  # 写入 LiDAR 到相机外参
        f.write("Tr_imu_to_velo: " + " ".join(["0" for _ in range(12)]) + "\n")  # 写入 IMU 到 LiDAR 外参


def save_lidar_as_bin(lidar_measurement: carla.LidarMeasurement, path: Path):  # 定义保存 LiDAR 点云的函数
    points = np.frombuffer(lidar_measurement.raw_data, dtype=np.float32)  # 从 CARLA LiDAR 原始数据中读取 float32 点云
    points = np.reshape(points, (-1, 4))  # 将点云整理为 N×4 格式，每个点为 x、y、z和intensity
    points.tofile(path)  # 保存文件


def spawn_npc_vehicles(world, traffic_manager, number_of_vehicles):  # 生成车辆
    vehicles = []

    blueprints = world.get_blueprint_library()  # 获取 CARLA 蓝图库
    vehicle_bps = blueprints.filter("vehicle.*")  # 获取车辆蓝图
    spawn_points = world.get_map().get_spawn_points()  # 获取车辆生成点

    random.shuffle(spawn_points)  # 让车辆随机分布

    for spawn_point in spawn_points[:number_of_vehicles]:
        bp = random.choice(vehicle_bps)  # 随机选择一种车

        if bp.has_attribute("color"):
            color = random.choice(bp.get_attribute("color").recommended_values)  # 随机选择一种颜色
            bp.set_attribute("color", color)

        vehicle = world.try_spawn_actor(bp, spawn_point)  # 生成车辆

        if vehicle is not None:
            vehicle.set_autopilot(True, traffic_manager.get_port())  # 开启自动驾驶
            vehicles.append(vehicle)

    return vehicles


def spawn_walkers(world, number_of_walkers):  #行人
    walkers = []
    controllers = []

    blueprints = world.get_blueprint_library()
    walker_bps = blueprints.filter("walker.pedestrian.*")
    controller_bp = blueprints.find("controller.ai.walker")

    for _ in range(number_of_walkers):  # 循环生成行人
        spawn_location = world.get_random_location_from_navigation()  # 随机获取一个行人生成位置

        if spawn_location is None:
            continue

        walker_bp = random.choice(walker_bps)
        walker = world.try_spawn_actor(walker_bp, carla.Transform(spawn_location))  # 生成行人

        if walker is None:
            continue

        controller = world.try_spawn_actor(controller_bp, carla.Transform(), walker)

        if controller is None:
            continue

        try:
            controller.start()  # 行人控制器
            controller.go_to_location(world.get_random_location_from_navigation())  # 设置行人的目标移动位置
            controller.set_max_speed(1.4)  # 设置行人最大速度
        except RuntimeError:
            continue
        walkers.append(walker)
        controllers.append(controller)

    return walkers, controllers


def main():
    parser = argparse.ArgumentParser(description="Collect CARLA simulation data and save it in KITTI-like format.")  # 命令行参数解析器

    parser.add_argument("--host", default="172.21.240.1", help="CARLA server host")  # CARLA 地址
    parser.add_argument("--port", default=2000, type=int, help="CARLA server port")  # CARLA 端口
    parser.add_argument("--frames", default=1000, type=int, help="Number of frames to collect")  # 采集帧数
    parser.add_argument("--output", default="../dataset", help="Output dataset directory")  # 输出数据集目录
    parser.add_argument("--vehicles", default=30, type=int, help="Number of NPC vehicles")  # NPC 车辆数量
    parser.add_argument("--walkers", default=0, type=int, help="Number of NPC walkers")  # 行人数量
    parser.add_argument("--width", default=1242, type=int, help="Camera image width")  # 相机图像宽度
    parser.add_argument("--height", default=375, type=int, help="Camera image height")  # 相机图像高度
    parser.add_argument("--fov", default=90.0, type=float, help="Camera field of view")  # 相机视场角

    args = parser.parse_args()

    output_dir = Path(args.output).resolve()

    image_dir = output_dir / "image_2"  # RGB 图像保存目录
    label_dir = output_dir / "label_2"  # 标注文件保存目录
    lidar_dir = output_dir / "velodyne"  # LiDAR 点云保存目录
    calib_dir = output_dir / "calib"  # 标定文件保存目录

    for directory in [image_dir, label_dir, lidar_dir, calib_dir]:
        make_dir(directory)  # 创建输出目录

    client = carla.Client(args.host, args.port)
    client.set_timeout(30.0)

    world = client.get_world()
    original_settings = world.get_settings()

    traffic_manager = client.get_trafficmanager(8000)  # 获取 Traffic Manager
    traffic_manager.set_global_distance_to_leading_vehicle(2.5)  #车辆之间的安全距离
    traffic_manager.set_synchronous_mode(True)  # 同步模式

    sensors = []
    vehicles = []
    walkers = []
    walker_controllers = []
    ego_vehicle = None

    try:
        settings = world.get_settings()  # 获取当前 world 设置
        settings.synchronous_mode = True  # 开启同步模式
        settings.fixed_delta_seconds = 0.1  # 设置固定仿真步长为 0.1 秒
        world.apply_settings(settings)

        blueprints = world.get_blueprint_library()  # 获取 CARLA 蓝图库

        ego_bp = random.choice(blueprints.filter("vehicle.tesla.model3"))  # 生成四轮车俩
        spawn_point = random.choice(world.get_map().get_spawn_points())  # 随机选择一个车辆生成点

        ego_vehicle = world.spawn_actor(ego_bp, spawn_point)
        ego_vehicle.set_autopilot(True, traffic_manager.get_port())  # 开启自动驾驶
        vehicles.append(ego_vehicle)

        npc_vehicles = spawn_npc_vehicles(world, traffic_manager, args.vehicles)  # 生成其他车辆
        vehicles.extend(npc_vehicles)  # 将车辆加入车辆列表

        if args.walkers > 0:
            walkers, walker_controllers = spawn_walkers(world, args.walkers)

        camera_bp = blueprints.find("sensor.camera.rgb")  # 获取 RGB Camera 蓝图
        camera_bp.set_attribute("image_size_x", str(args.width))  # 相机图像宽度
        camera_bp.set_attribute("image_size_y", str(args.height))  # 相机图像高度
        camera_bp.set_attribute("fov", str(args.fov))  # 相机视场角

        camera_transform = carla.Transform(carla.Location(x=1.5, y=0.0, z=2.4), carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0))  # 相机安装位置和角度

        camera = world.spawn_actor(camera_bp, camera_transform, attach_to=ego_vehicle)  # 将相机绑定到主视角车上
        sensors.append(camera)  # 保存相机传感器

        lidar_bp = blueprints.find("sensor.lidar.ray_cast")  # 获取 LiDAR 蓝图
        lidar_bp.set_attribute("channels", "32")  # LiDAR 通道数
        lidar_bp.set_attribute("range", "60")  # LiDAR 探测范围
        lidar_bp.set_attribute("points_per_second", "56000")  # LiDAR 每秒点数
        lidar_bp.set_attribute("rotation_frequency", "10")  # LiDAR 旋转频率
        lidar_bp.set_attribute("upper_fov", "10")  # LiDAR 上视场角
        lidar_bp.set_attribute("lower_fov", "-30")  # LiDAR 下视场角

        lidar_transform = carla.Transform(carla.Location(x=0.0, y=0.0, z=2.5), carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0))  # LiDAR 安装位置和角度

        lidar = world.spawn_actor(lidar_bp, lidar_transform, attach_to=ego_vehicle)
        sensors.append(lidar)

        image_queue = queue.Queue()  # 图像数据队列
        lidar_queue = queue.Queue()  # LiDAR 数据队列

        camera.listen(image_queue.put)  # 相机监听函数
        lidar.listen(lidar_queue.put)  # LiDAR 监听函数

        k = build_camera_intrinsic(args.width, args.height, args.fov)  # 计算相机内参矩阵

        print("Start collecting data...")
        print(f"Output directory: {output_dir}")
        print(f"Frames: {args.frames}")
        print(f"Vehicles: {args.vehicles}")
        print(f"Walkers: {args.walkers}")

        for frame_id in range(args.frames):  # 按帧循环采集数据
            world.tick()

            image = image_queue.get(timeout=10.0)  # 从相机队列中读取当前帧图像
            lidar_data = lidar_queue.get(timeout=10.0)  # 从 LiDAR 队列中读取当前帧点云

            file_id = f"{frame_id:06d}"  # 生成文件编号

            image_path = image_dir / f"{file_id}.png"
            label_path = label_dir / f"{file_id}.txt"
            lidar_path = lidar_dir / f"{file_id}.bin"
            calib_path = calib_dir / f"{file_id}.txt"

            image.save_to_disk(str(image_path))  # 保存当前帧 RGB 图像
            save_lidar_as_bin(lidar_data, lidar_path)  # 保存当前帧 LiDAR 点云
            write_calib_file(calib_path, k)  # 保存当前帧相机标定文件

            labels = []

            try:
                ego_location = ego_vehicle.get_location()  # 获取主视角车辆的当前世界坐标
            except RuntimeError:
                print("Ego vehicle is not available.")
                break

            actor_list = world.get_actors()  # 获取当前 world 中所有的物品
            target_actors = list(actor_list)  # 未知类型标注为 other

            for actor in target_actors:
                if ego_vehicle is not None and actor.id == ego_vehicle.id:
                    continue

                obj_type = carla_type_to_kitti(actor)  # 将 CARLA actor 类型转换为 KITTI-like 类别

                if obj_type == "ignore":
                    continue

                try:
                    actor_location = actor.get_location()
                    distance = ego_location.distance(actor_location)
                except RuntimeError:
                    continue

                if distance > 60:
                    continue

                bbox = get_2d_bbox(actor, camera, k, args.width, args.height)

                if bbox is None:
                    continue

                xmin, ymin, xmax, ymax, depth = bbox  # 解包 2D 框坐标和深度

                try:
                    extent = actor.bounding_box.extent  # 获取 bounding box 的半尺寸
                except Exception:
                    continue

                height = extent.z * 2.0  # 目标高度
                width = extent.y * 2.0  # 目标宽度
                length = extent.x * 2.0  # 目标长度

                line = (  # 按 KITTI-like 格式生成一行标注
                    f"{obj_type} "  # 写入目标类别
                    f"0.00 0 0.00 "  # 写入 truncated、occluded、alpha
                    f"{xmin:.2f} {ymin:.2f} {xmax:.2f} {ymax:.2f} "  # 写入 2D bounding box
                    f"{height:.2f} {width:.2f} {length:.2f} "  # 写入 3D 尺寸
                    f"0.00 0.00 {depth:.2f} 0.00"  # 写入位置和 rotation_y，
                )

                labels.append(line)  # 将当前目标标注加入当前帧 labels 列表

            with open(label_path, "w", encoding="utf-8") as f:  # 打开当前帧 label 文件
                for line in labels:  # 遍历当前帧所有标注
                    f.write(line + "\n")  # 写入一行标注

            if frame_id % 50 == 0:  # 每 50 帧显示一次采集进度
                print(f"Collected frame {frame_id}/{args.frames}, labels: {len(labels)}")  # 打印当前采集进度和目标数量

        print("Data collection finished successfully.")  # 打印数据采集完成提示

    finally:
        print("Cleaning up sensors and restoring settings...")

        for sensor in sensors:  # 遍历所有传感器
            try:
                if sensor is not None:  # 如果传感器对象存在
                    sensor.stop()  # 停止传感器监听
            except Exception:
                pass

        try:
            world.apply_settings(original_settings)  # 恢复程序启动前的设置
        except Exception:
            pass

        try:
            traffic_manager.set_synchronous_mode(False)  # 关闭 Traffic Manager 同步模式
        except Exception:
            pass

        print("Done. Please close and restart CARLA Server to clean all remaining actors.")


if __name__ == "__main__":
    main()




