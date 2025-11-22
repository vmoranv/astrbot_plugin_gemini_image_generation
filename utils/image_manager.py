"""
图片文件管理工具
统一管理图片的保存和清理
"""

from datetime import datetime, timedelta
from pathlib import Path

from astrbot.api import logger


async def cleanup_old_images(images_dir: Path | None = None):
    """
    清理超过15分钟的图像文件

    Args:
        images_dir (Path): images 目录路径，如果为None则使用默认路径
    """
    try:
        # 默认路径：插件根目录下的 images 文件夹
        if images_dir is None:
            images_dir = Path(__file__).parent.parent / "images"

        if not images_dir.exists():
            return

        current_time = datetime.now()
        cutoff_time = current_time - timedelta(minutes=15)

        # 查找 images 目录下的所有图像文件（支持新旧两种命名格式）
        image_patterns = [
            "gemini_image_*.png",  # 旧格式（来自 ttp.py）
            "gemini_image_*.jpg",
            "gemini_image_*.jpeg",
            "gemini_advanced_image_*.png",  # 新格式（来自 api_client.py）
            "gemini_advanced_image_*.jpg",
            "gemini_advanced_image_*.jpeg",
        ]

        cleaned_count = 0
        for pattern in image_patterns:
            for file_path in images_dir.glob(pattern):
                try:
                    # 获取文件的修改时间
                    file_mtime = datetime.fromtimestamp(file_path.stat().st_mtime)

                    # 如果文件超过15分钟，删除它
                    if file_mtime < cutoff_time:
                        file_path.unlink()
                        cleaned_count += 1
                        logger.debug(f"已清理过期图像: {file_path.name}")

                except Exception as e:
                    logger.warning(f"清理文件 {file_path} 时出错: {e}")

        if cleaned_count > 0:
            logger.debug(f"共清理 {cleaned_count} 个过期图像文件")

    except Exception as e:
        logger.error(f"图像清理过程出错: {e}")


async def save_image_data(
    image_data: bytes, image_format: str, images_dir: Path | None = None
) -> str | None:
    """
    保存图像数据到文件，并自动清理旧图片

    Args:
        image_data (bytes): 图像数据
        image_format (str): 图像格式 (png, jpg, jpeg)
        images_dir (Path): images 目录路径，如果为None则使用默认路径

    Returns:
        str: 保存的文件路径，失败返回 None
    """
    try:
        # 默认路径：插件根目录下的 images 文件夹
        if images_dir is None:
            images_dir = Path(__file__).parent.parent / "images"

        images_dir.mkdir(exist_ok=True)

        # 先清理旧图像
        await cleanup_old_images(images_dir)

        # 生成唯一文件名（使用时间戳和微秒）
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"gemini_advanced_image_{timestamp}.{image_format or 'png'}"
        image_path = images_dir / filename

        logger.debug(f"💾 准备写入文件: {image_path}, 大小: {len(image_data)} bytes")

        # 保存文件
        with open(image_path, "wb") as f:
            f.write(image_data)

        logger.debug(f"✅ 图像已保存: {image_path} (大小: {len(image_data)} bytes)")
        return str(image_path)

    except Exception as e:
        logger.error(f"保存图像失败: {e}")
        return None


async def save_base64_image(
    base64_string: str, image_format: str = "png", images_dir: Path | None = None
) -> str | None:
    """
    保存 base64 图像数据到文件，并自动清理旧图片

    Args:
        base64_string (str): base64 编码的图像数据
        image_format (str): 图像格式 (png, jpg, jpeg)
        images_dir (Path): images 目录路径，如果为None则使用默认路径

    Returns:
        str: 保存的文件路径，失败返回 None
    """
    try:
        import base64

        image_data = base64.b64decode(base64_string)
        return await save_image_data(image_data, image_format, images_dir)
    except Exception as e:
        logger.error(f"Base64 解码失败: {e}")
        return None
