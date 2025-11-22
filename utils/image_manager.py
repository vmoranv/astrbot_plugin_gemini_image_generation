"""
图片文件管理工具
统一管理图片的保存和清理
"""

import asyncio
import base64
from datetime import datetime, timedelta
from pathlib import Path

import aiohttp

from astrbot.api import logger
from astrbot.api.star import StarTools


def get_plugin_data_dir() -> Path:
    """获取插件数据目录"""
    return StarTools.get_data_dir("astrbot_plugin_gemini_image_generation")


async def cleanup_old_images(images_dir: Path | None = None):
    """
    清理超过15分钟的图像文件

    Args:
        images_dir (Path): images 目录路径，如果为None则使用默认路径
    """
    try:
        # 默认路径：插件根目录下的 images 文件夹
        if images_dir is None:
            images_dir = get_plugin_data_dir() / "images"

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


async def download_qq_avatar(
    user_id: str, cache_name: str, images_dir: Path | None = None
) -> str | None:
    """
    下载QQ头像并转换为base64格式

    Args:
        user_id (str): QQ用户ID
        cache_name (str): 缓存文件名前缀
        images_dir (Path): images目录路径，如果为None则使用默认路径

    Returns:
        str: base64格式的头像数据，失败返回None
    """
    try:
        # 默认路径
        if images_dir is None:
            images_dir = get_plugin_data_dir() / "images"

        cache_dir = images_dir / "avatar_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        avatar_file = cache_dir / f"{cache_name}_avatar.jpg"

        # 检查缓存
        if avatar_file.exists() and avatar_file.stat().st_size > 1000:
            with open(avatar_file, "rb") as f:
                cached_data = f.read()
            base64_data = base64.b64encode(cached_data).decode("utf-8")
            logger.debug(f"使用缓存的头像: {avatar_file}")
            return f"data:image/jpeg;base64,{base64_data}"

        # 下载头像
        avatar_url = f"https://q4.qlogo.cn/headimg_dl?dst_uin={user_id}&spec=640"
        logger.debug(f"获取QQ头像URL: {avatar_url}")

        timeout = aiohttp.ClientTimeout(total=5)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(avatar_url) as response:
                if response.status != 200:
                    logger.debug(f"下载头像失败: HTTP {response.status}")
                    return None

                image_data = await response.read()

                # 检查是否是有效图片（过滤默认头像）
                if len(image_data) < 1000:
                    logger.debug("头像文件过小，可能是默认头像，跳过")
                    return None

                # 保存到缓存
                with open(avatar_file, "wb") as f:
                    f.write(image_data)
                logger.debug(f"头像已缓存: {avatar_file} ({len(image_data)} bytes)")

                base64_data = base64.b64encode(image_data).decode("utf-8")
                return f"data:image/jpeg;base64,{base64_data}"

    except asyncio.TimeoutError:
        logger.debug(f"下载头像超时: {user_id}")
        return None
    except Exception as e:
        logger.debug(f"下载转换头像失败: {e}")
        return None


async def cleanup_avatar_cache(cache_name: str, images_dir: Path | None = None):
    """
    清理指定的头像缓存文件

    Args:
        cache_name (str): 要清理的缓存文件名前缀
        images_dir (Path): images目录路径，如果为None则使用默认路径
    """
    try:
        if images_dir is None:
            images_dir = get_plugin_data_dir() / "images"

        cache_dir = images_dir / "avatar_cache"
        if not cache_dir.exists():
            return

        patterns = [
            f"{cache_name}_avatar.jpg",
            f"{cache_name}_avatar.jpeg",
            f"{cache_name}_avatar.png",
        ]
        cleaned_count = 0

        for pattern in patterns:
            for file_path in cache_dir.glob(pattern):
                try:
                    file_path.unlink()
                    cleaned_count += 1
                    logger.debug(f"已清理头像缓存: {file_path.name}")
                except Exception as e:
                    logger.warning(f"清理头像缓存文件 {file_path} 时出错: {e}")

        if cleaned_count > 0:
            logger.debug(f"共清理 {cleaned_count} 个头像缓存文件")

    except Exception as e:
        logger.error(f"头像缓存清理过程出错: {e}")


class AvatarManager:
    """头像管理器，统一管理头像的获取、使用和清理"""

    def __init__(self, images_dir: Path | None = None):
        self.images_dir = images_dir or get_plugin_data_dir() / "images"
        self.used_avatars = []  # 记录本次使用的头像缓存名称

    async def get_avatar(self, user_id: str, cache_name: str) -> str | None:
        """获取头像并记录使用"""
        avatar_data = await download_qq_avatar(user_id, cache_name, self.images_dir)
        if avatar_data:
            self.used_avatars.append(cache_name)
        return avatar_data

    async def cleanup_used_avatars(self):
        """清理所有使用过的头像缓存"""
        for cache_name in self.used_avatars:
            await cleanup_avatar_cache(cache_name, self.images_dir)
        self.used_avatars.clear()
        logger.debug("已清理本次使用的所有头像缓存")


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
            images_dir = get_plugin_data_dir() / "images"

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


async def file_to_base64(file_path: str) -> str | None:
    """
    将文件转换为 base64 编码字符串

    Args:
        file_path (str): 文件路径，可以是本地路径或 URL

    Returns:
        str: base64 编码的图片数据（带 data URI scheme），失败返回 None
    """
    try:
        import base64
        from urllib.parse import urlparse

        import aiohttp

        # 判断是 URL 还是本地文件
        parsed = urlparse(file_path)
        is_url = bool(parsed.scheme and parsed.netloc)

        image_data = None

        if is_url:
            # URL 路径 - 下载图片
            async with aiohttp.ClientSession() as session:
                async with session.get(file_path, timeout=10) as response:
                    if response.status == 200:
                        image_data = await response.read()
                    else:
                        logger.error(f"下载图片失败: HTTP {response.status}")
                        return None
        else:
            # 本地文件路径
            with open(file_path, "rb") as f:
                image_data = f.read()

        if not image_data:
            logger.error("无法读取图片数据")
            return None

        # 检测图片格式
        ext = Path(file_path).suffix.lower()
        if ext in [".jpg", ".jpeg"]:
            mime_type = "image/jpeg"
        elif ext in [".png"]:
            mime_type = "image/png"
        elif ext in [".webp"]:
            mime_type = "image/webp"
        else:
            # 尝试从文件头检测
            if image_data.startswith(b"\xff\xd8"):
                mime_type = "image/jpeg"
            elif image_data.startswith(b"\x89PNG"):
                mime_type = "image/png"
            elif image_data.startswith(b"RIFF") and image_data[8:12] == b"WEBP":
                mime_type = "image/webp"
            else:
                mime_type = "image/jpeg"  # 默认使用 jpeg

        # 转换为 base64
        base64_data = base64.b64encode(image_data).decode("utf-8")
        result = f"data:{mime_type};base64,{base64_data}"

        logger.debug(f"✓ 文件转换为 base64 成功: {file_path} ({len(image_data)} bytes)")
        return result

    except Exception as e:
        logger.error(f"文件转换为 base64 失败: {e}")
        return None
