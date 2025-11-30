import zipfile
from pathlib import Path
from typing import Any

from PIL import Image

from astrbot.api import logger

from .tl_utils import get_plugin_data_dir


def split_image(
    image_path: str,
    rows: int = 6,
    cols: int = 4,
    output_dir: str | None = None,
    bboxes: list[dict[str, Any]] | None = None,
) -> list[str]:
    """
    将图片切分为指定行列的网格
    若提供 bboxes，则按识别出的裁剪框切割

    Args:
        image_path: 源图片路径
        rows: 行数，默认6行
        cols: 列数，默认4列
        output_dir: 输出目录，如果不指定则使用插件数据目录下的 split_output
        bboxes: 可选的裁剪框列表，格式 [{"x": int, "y": int, "width": int, "height": int}]

    Returns:
        List[str]: 切分后的图片文件路径列表，按顺序排列
    """
    try:
        with Image.open(image_path) as img:
            width, height = img.size

            # 如果图片是横向的（宽 > 高），交换行列数（网格模式）
            if not bboxes and width > height and rows > cols:
                rows, cols = cols, rows
                logger.debug(f"检测到横向图片，自动调整切割网格为 {rows}行 x {cols}列")

            # 如果未指定输出目录，则使用插件的标准数据目录
            if not output_dir:
                data_dir = get_plugin_data_dir()
                output_dir_path = data_dir / "split_output"
            else:
                output_dir_path = Path(output_dir)

            # 获取源文件名（不含扩展名和路径）作为子目录，避免文件混淆
            base_name = Path(image_path).stem
            # 最终存储目录: .../split_output/base_name/
            final_output_dir = output_dir_path / base_name

            final_output_dir.mkdir(parents=True, exist_ok=True)

            output_files = []

            if bboxes:
                for idx, box in enumerate(bboxes, 1):
                    try:
                        x = int(box.get("x", 0))
                        y = int(box.get("y", 0))
                        w = int(box.get("width", 0))
                        h = int(box.get("height", 0))
                    except Exception:
                        logger.debug(f"跳过无效裁剪框: {box}")
                        continue

                    if w <= 0 or h <= 0:
                        logger.debug(f"跳过尺寸异常裁剪框: {box}")
                        continue

                    left = max(0, x)
                    upper = max(0, y)
                    right = min(width, left + w)
                    lower = min(height, upper + h)

                    if right <= left or lower <= upper:
                        logger.debug(f"跳过越界裁剪框: {box}")
                        continue

                    piece = img.crop((left, upper, right, lower))
                    file_name = f"{base_name}_{idx}.png"
                    file_path = final_output_dir / file_name
                    piece.save(file_path, "PNG")
                    output_files.append(str(file_path))
            else:
                # 网格裁剪模式
                piece_width = width // cols
                piece_height = height // rows

                for r in range(rows):
                    for c in range(cols):
                        left = c * piece_width
                        upper = r * piece_height
                        right = left + piece_width
                        lower = upper + piece_height

                        piece = img.crop((left, upper, right, lower))

                        idx = r * cols + c + 1
                        file_name = f"{base_name}_{idx}.png"
                        file_path = final_output_dir / file_name

                        piece.save(file_path, "PNG")
                        output_files.append(str(file_path))

            return output_files

    except Exception as e:
        logger.error(f"Error splitting image: {e}")
        return []


def create_zip(files: list[str], output_filename: str | None = None) -> str | None:
    """
    将文件列表打包成zip

    Args:
        files: 文件路径列表
        output_filename: 输出zip文件名（包含路径）。如果不指定，则使用第一个文件的目录 + 目录名.zip

    Returns:
        str: zip文件路径，失败返回None
    """
    if not files:
        return None

    try:
        if not output_filename:
            first_file = Path(files[0])
            dir_path = first_file.parent
            dir_name = dir_path.name
            # 输出到目录的同级，即 .../split_output/base_name.zip
            output_filename_path = dir_path.parent / f"{dir_name}.zip"
            output_filename = str(output_filename_path)

        with zipfile.ZipFile(output_filename, "w", zipfile.ZIP_DEFLATED) as zipf:
            for file in files:
                file_path = Path(file)
                zipf.write(file_path, file_path.name)

        return output_filename
    except Exception as e:
        logger.error(f"Error creating zip: {e}")
        return None
