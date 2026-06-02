from pathlib import Path
import cv2


def draw_labels(image_path: Path, label_path: Path, output_path: Path):  #标注框
    image = cv2.imread(str(image_path))  # 使用 OpenCV 读取输入图像

    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    if not label_path.exists():
        raise FileNotFoundError(f"Label file not found: {label_path}")

    with open(label_path, "r", encoding="utf-8") as f:  # 打开当前图像对应的 label 文件
        lines = f.readlines()  # 读取文件

    for line in lines:  # 遍历每一行标注
        parts = line.strip().split()

        if len(parts) < 15:  # 如果字段数量不足
            continue

        obj_type = parts[0]  # 第 0 个字段为目标类别

        xmin = int(float(parts[4]))  # 第 4 个字段为左边界
        ymin = int(float(parts[5]))  # 第 5 个字段为上边界
        xmax = int(float(parts[6]))  # 第 6 个字段为右边界
        ymax = int(float(parts[7]))  # 第 7 个字段为下边界

        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)  # 绘制绿色矩形框

        cv2.putText(  # 在图像上绘制类别文字
            image,
            obj_type,  #文字内容
            (xmin, max(0, ymin - 5)),  #文字位置
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    cv2.imwrite(str(output_path), image)  # 保存图像
    print(f"Saved visualization to {output_path}")


if __name__ == "__main__":
    root = Path("../dataset")  # 设置数据集根目录

    frame_ids = [
        "000000",  # 第 0 帧
        "000100",  # 第 100 帧
        "000500"  # 第 500 帧
    ]

    for frame_id in frame_ids:
        image_path = root / "image_2" / f"{frame_id}.png"
        label_path = root / "label_2" / f"{frame_id}.txt"
        output_path = root / f"vis_{frame_id}.png"

        draw_labels(image_path, label_path, output_path)