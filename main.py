"""
AstrBot Gemini 图像生成插件主文件
支持 Google 官方 API 和 OpenAI 兼容格式 API，提供生图和改图功能，支持智能头像参考
"""

from __future__ import annotations

import asyncio
import base64
from pathlib import Path
from typing import Any

import aiohttp

from astrbot.api import logger
from astrbot.api.all import Image, Reply
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.star import Context, Star, register

from .utils.api_client import (
    APIError,
    ApiRequestConfig,
    GeminiAPIClient,
    get_api_client,
)
from .utils.image_manager import AvatarManager


@register(
    "astrbot_plugin_gemini_image_generation",
    "piexian",
    "Gemini图像生成插件，支持生图和改图，可以自动获取头像作为参考",
    "v1.0.0",
)
class GeminiImageGenerationPlugin(Star):
    def __init__(self, context: Context, config: dict[str, Any]):
        super().__init__(context)
        self.config = config
        self.api_client: GeminiAPIClient | None = None
        self.avatar_manager = AvatarManager()

        # 加载配置
        self._load_config()

    def get_tool_timeout(self, event: AstrMessageEvent | None = None) -> int:
        """获取当前聊天环境的 tool_call_timeout 配置"""
        try:
            # 如果提供了事件，尝试获取特定聊天环境的配置
            if event:
                umo = event.unified_msg_origin
                chat_config = self.context.get_config(umo=umo)
                return chat_config.get("provider_settings", {}).get(
                    "tool_call_timeout", 60
                )

            # 否则使用默认配置
            default_config = self.context.get_config()
            return default_config.get("provider_settings", {}).get(
                "tool_call_timeout", 60
            )
        except Exception as e:
            logger.warning(f"获取 tool_call_timeout 配置失败: {e}，使用默认值 60 秒")
            return 60

    async def get_avatar_reference(self, event: AstrMessageEvent) -> list[str]:
        """获取头像作为参考图像，支持群头像和用户头像（直接HTTP下载）"""
        avatar_images = []
        download_tasks = []

        try:
            # 检查是否需要获取群头像
            if hasattr(event, "group_id") and event.group_id:
                group_id = str(event.group_id)
                prompt = event.message_str.lower()

                # 群头像获取的几种情况：
                # 1. 明确提到群相关关键词
                # 2. 在群聊中且启用了自动头像参考且触发了生图指令
                group_avatar_keywords = [
                    "群头像",
                    "本群",
                    "我们的群",
                    "这个群",
                    "群标志",
                    "群图标",
                ]
                explicit_group_request = any(
                    keyword in prompt for keyword in group_avatar_keywords
                )

                # 判断是否应该获取群头像
                should_get_group_avatar = explicit_group_request or (
                    self.auto_avatar_reference
                    and any(
                        keyword in prompt
                        for keyword in [
                            "生图",
                            "绘图",
                            "画图",
                            "生成图片",
                            "制作图片",
                            "改图",
                            "修改",
                        ]
                    )
                )

                if should_get_group_avatar:
                    if explicit_group_request:
                        logger.info(
                            f"检测到明确的群头像关键词，准备获取群 {group_id} 的头像"
                        )
                    else:
                        logger.info(
                            f"群聊中生图指令触发，自动获取群 {group_id} 的头像作为参考"
                        )

                    # 群头像暂时跳过，因为QQ群头像需要特殊API
                    logger.info("群头像功能暂未实现，跳过")

            # 获取头像逻辑
            # 获取头像：优先获取@用户头像，如果无@用户则获取发送者头像
            mentioned_users = await self.parse_mentions(event)

            if mentioned_users:
                # 有@用户：只获取被@用户的头像
                for user_id in mentioned_users:
                    logger.info(f"[AVATAR] 获取@用户头像: {user_id}")
                    download_tasks.append(
                        self._download_qq_avatar(str(user_id), f"mentioned_{user_id}")
                    )
            else:
                # 无@用户：获取发送者头像
                if (
                    hasattr(event, "message_obj")
                    and hasattr(event.message_obj, "sender")
                    and hasattr(event.message_obj.sender, "user_id")
                ):
                    sender_id = str(event.message_obj.sender.user_id)
                    logger.info(f"[AVATAR] 获取发送者头像: {sender_id}")
                    download_tasks.append(
                        self._download_qq_avatar(sender_id, f"sender_{sender_id}")
                    )

            # 执行下载任务
            if download_tasks:
                logger.info(
                    f"[AVATAR_DEBUG] 开始并发下载 {len(download_tasks)} 个头像..."
                )
                try:
                    # 设置总体超时时间为8秒，避免单个下载拖慢整体
                    results = await asyncio.wait_for(
                        asyncio.gather(*download_tasks, return_exceptions=True),
                        timeout=8.0,
                    )

                    # 处理结果
                    for result in results:
                        if isinstance(result, str) and result:
                            avatar_images.append(result)
                        elif isinstance(result, Exception):
                            logger.warning(f"头像下载任务失败: {result}")

                    logger.info(
                        f"头像下载完成，成功获取 {len(avatar_images)} 个头像，即将返回"
                    )

                except asyncio.TimeoutError:
                    logger.warning("头像下载总体超时，跳过剩余头像下载")
                except Exception as e:
                    logger.error(f"并发下载头像时发生错误: {e}")

        except Exception as e:
            logger.error(f"获取头像参考失败: {e}")

        return avatar_images

    async def _download_qq_avatar(self, user_id: str, cache_name: str) -> str | None:
        """直接下载QQ头像，参考lmarena插件的实现"""
        try:
            # QQ头像URL格式，使用q4服务器
            avatar_url = f"https://q4.qlogo.cn/headimg_dl?dst_uin={user_id}&spec=640"
            self.log_info(f"获取QQ头像URL: {avatar_url}")

            # 下载并转换头像
            avatar_data = await self._download_and_convert_avatar(
                avatar_url, f"qq_user_{cache_name}"
            )
            return avatar_data

        except Exception as e:
            logger.warning(f"获取QQ用户 {user_id} 头像失败: {e}")
            return None

    async def _get_user_avatar(self, bot, user_id: str, cache_name: str) -> str | None:
        """获取指定用户的头像"""
        try:
            # QQ头像URL格式
            avatar_url = f"https://q1.qlogo.cn/g?b=qq&nk={user_id}&s=640"
            self.log_info(f"获取用户头像URL: {avatar_url}")

            # 下载并转换头像
            avatar_data = await self._download_and_convert_avatar(
                avatar_url, f"user_{cache_name}"
            )
            return avatar_data

        except Exception as e:
            logger.warning(f"获取用户 {user_id} 头像失败: {e}")
            return None

    async def _download_and_convert_avatar(
        self, avatar_url: str, cache_name: str
    ) -> str | None:
        """下载并转换头像为base64格式（优化版本，减少超时时间）"""
        try:
            # 检查缓存
            cache_dir = Path(__file__).parent / "images" / "avatar_cache"
            cache_dir.mkdir(parents=True, exist_ok=True)
            avatar_file = cache_dir / f"{cache_name}_avatar.jpg"

            # 如果缓存文件存在且不为空，直接使用
            if avatar_file.exists() and avatar_file.stat().st_size > 1000:
                with open(avatar_file, "rb") as f:
                    cached_data = f.read()
                base64_data = base64.b64encode(cached_data).decode("utf-8")
                self.log_debug(f"使用缓存的头像: {avatar_file}")
                return f"data:image/jpeg;base64,{base64_data}"

            # 设置较短的超时时间，避免阻塞
            timeout = aiohttp.ClientTimeout(total=5)  # 5秒超时
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(avatar_url) as response:
                    if response.status != 200:
                        self.log_debug(f"下载头像失败: HTTP {response.status}")
                        return None

                    image_data = await response.read()

                    # 检查是否是有效图片（过滤默认头像）
                    if len(image_data) < 1000:
                        self.log_debug("头像文件过小，可能是默认头像，跳过")
                        return None

                    # 保存头像到缓存目录
                    with open(avatar_file, "wb") as f:
                        f.write(image_data)

                    self.log_debug(
                        f"头像已缓存: {avatar_file} ({len(image_data)} bytes)"
                    )

                    # 返回base64编码的图片数据
                    base64_data = base64.b64encode(image_data).decode("utf-8")
                    return f"data:image/jpeg;base64,{base64_data}"

        except asyncio.TimeoutError:
            self.log_debug(f"下载头像超时: {avatar_url}")
            return None
        except Exception as e:
            self.log_debug(f"下载转换头像失败: {e}")
            return None

    async def should_use_avatar(self, event: AstrMessageEvent) -> bool:
        """判断是否应该使用头像作为参考"""
        # 首先检查配置是否启用了自动头像参考
        self.log_info(
            f"[AVATAR_DEBUG] 检查auto_avatar_reference: {self.auto_avatar_reference}"
        )
        if not self.auto_avatar_reference:
            return False

        if not hasattr(event, "message_str"):
            self.log_info("[AVATAR_DEBUG] event没有message_str属性")
            return False

        prompt = event.message_str.lower()
        self.log_info(f"[AVATAR_DEBUG] 检查消息: '{prompt}'")

        # 更模糊的头像触发条件
        avatar_keywords = [
            # 直接头像相关
            "头像",
            "根据我",
            "按照我",
            "基于我",
            "参考我",
            "我的头像",
            "我的",
            # 修改相关（包含各种变体）
            "修改",
            "改图",
            "改成",
            "改为",
            "变成",
            "变",
            "换成",
            "替换",
            "调整",
            "优化",
            "重做",
            "重新",
            "换风格",
            # @触发（在parse_mentions中处理）
            # 指令相关
            "生图",
            "绘图",
            "画图",
            "生成图片",
            "制作图片",
        ]

        found_keywords = [keyword for keyword in avatar_keywords if keyword in prompt]
        self.log_info(f"[AVATAR_DEBUG] 找到的关键词: {found_keywords}")
        return len(found_keywords) > 0

    async def parse_mentions(self, event: AstrMessageEvent) -> list[int]:
        """解析消息中的@用户，返回用户ID列表"""
        mentioned_users = []

        try:
            # 使用框架提供的方法获取消息组件
            messages = event.get_messages()

            for msg_component in messages:
                # 检查是否是@组件
                if hasattr(msg_component, "qq") and str(msg_component.qq) != str(
                    event.get_self_id()
                ):
                    mentioned_users.append(int(msg_component.qq))
                    self.log_debug(f"解析到@用户: {msg_component.qq}")

        except Exception as e:
            logger.warning(f"解析@用户失败: {e}")

        return mentioned_users

    def _load_config(self):
        """从配置加载所有设置"""
        # API 密钥列表
        self.api_keys = self.config.get("openrouter_api_keys", [])
        if not isinstance(self.api_keys, list):
            self.api_keys = [self.api_keys] if self.api_keys else []

        # API 设置
        api_settings = self.config.get("api_settings", {})
        self.api_type = api_settings.get("api_type", "google")
        self.api_base = api_settings.get("custom_api_base", "")
        self.model = api_settings.get("model", "gemini-3-pro-image-preview")

        # 图像生成设置
        image_settings = self.config.get("image_generation_settings", {})
        self.resolution = image_settings.get("resolution", "1K")
        self.aspect_ratio = image_settings.get("aspect_ratio", "1:1")
        self.enable_grounding = image_settings.get("enable_grounding", False)
        self.max_reference_images = image_settings.get("max_reference_images", 6)
        self.enable_text_response = image_settings.get("enable_text_response", False)

        # 重试设置
        retry_settings = self.config.get("retry_settings", {})
        self.max_attempts_per_key = retry_settings.get("max_attempts_per_key", 3)
        self.enable_smart_retry = retry_settings.get("enable_smart_retry", True)
        self.total_timeout = retry_settings.get("total_timeout", 120)

        # 服务设置
        service_settings = self.config.get("service_settings", {})
        self.nap_server_address = service_settings.get(
            "nap_server_address", "localhost"
        )
        self.nap_server_port = service_settings.get("nap_server_port", 3658)
        self.auto_avatar_reference = service_settings.get(
            "auto_avatar_reference", False
        )

        # 日志设置
        self.verbose_logging = service_settings.get("verbose_logging", False)

        # 初始化 API 客户端
        if self.api_keys:
            self.api_client = get_api_client(self.api_keys)
            self.log_info("✓ API 客户端已初始化")
            self.log_info(f"  - 类型: {self.api_type}")
            self.log_info(f"  - 模型: {self.model}")
            self.log_info(f"  - 密钥数量: {len(self.api_keys)}")
            if self.api_base:
                self.log_info(f"  - 自定义 API Base: {self.api_base}")
        else:
            logger.warning("✗ 未配置 API 密钥")

    def log_info(self, message: str):
        """根据配置输出info或debug级别日志"""
        if self.verbose_logging:
            logger.info(message)
        else:
            logger.debug(message)

    def log_debug(self, message: str):
        """输出debug级别日志"""
        logger.debug(message)

    async def initialize(self):
        """插件初始化"""
        if self.api_client:
            logger.info("🎨 Gemini 图像生成插件已加载")
        else:
            logger.error("✗ API 客户端初始化失败，请检查配置")

    async def _collect_reference_images(self, event: AstrMessageEvent) -> list[str]:
        """从消息和回复中提取参考图片，并转换为base64格式"""
        reference_images = []
        max_images = self.max_reference_images

        if not hasattr(event, "message_obj") or not event.message_obj:
            return reference_images

        message_chain = event.message_obj.message
        if not message_chain:
            return reference_images

        # 从当前消息提取图片
        for component in message_chain:
            if isinstance(component, Image) and len(reference_images) < max_images:
                try:
                    # 使用 convert_to_base64() 方法直接获取 base64 数据
                    base64_data = await component.convert_to_base64()
                    if base64_data:
                        reference_images.append(base64_data)
                        logger.debug(
                            f"✓ 从当前消息提取图片 (当前: {len(reference_images)}/{max_images})"
                        )
                    else:
                        logger.warning("✗ 图片转换失败")
                except Exception as e:
                    logger.warning(f"✗ 提取图片失败: {e}")

        # 从回复消息提取图片
        for component in message_chain:
            if isinstance(component, Reply) and component.chain:
                for reply_comp in component.chain:
                    if (
                        isinstance(reply_comp, Image)
                        and len(reference_images) < max_images
                    ):
                        try:
                            # 使用 convert_to_base64() 方法直接获取 base64 数据
                            base64_data = await reply_comp.convert_to_base64()
                            if base64_data:
                                reference_images.append(base64_data)
                                self.log_debug("✓ 从回复消息提取图片")
                            else:
                                logger.warning("✗ 回复图片转换失败")
                        except Exception as e:
                            logger.warning(f"✗ 提取回复图片失败: {e}")

        self.log_info(f"📸 共收集到 {len(reference_images)} 张参考图片")
        return reference_images

    async def _send_image_with_fallback(self, image_path: str) -> Image:
        """发送图片，优先使用 callback_api_base（优化版本，避免网络阻塞）"""
        callback_api_base = self.context.get_config().get("callback_api_base")

        if not callback_api_base:
            self.log_debug("未配置 callback_api_base，使用本地文件发送")
            return Image.fromFileSystem(image_path)

        try:
            # 尝试生成网络链接，但设置超时控制
            image_component = Image.fromFileSystem(image_path)
            download_url = await asyncio.wait_for(
                image_component.convert_to_web_link(),
                timeout=5.0,  # 5秒超时
            )
            self.log_debug("成功生成下载链接")
            return Image.fromURL(download_url)
        except asyncio.TimeoutError:
            logger.warning("生成下载链接超时，退回到本地文件")
            return Image.fromFileSystem(image_path)
        except (OSError, ConnectionError, TimeoutError) as e:
            logger.warning(f"网络/文件操作失败: {e}，退回到本地文件")
            return Image.fromFileSystem(image_path)
        except Exception as e:
            logger.error(f"发送图片出错: {e}，退回到本地文件")
            return Image.fromFileSystem(image_path)

    async def _generate_image_core(
        self,
        event: AstrMessageEvent,
        prompt: str,
        reference_images: list[str],
        avatar_reference: list[str],
    ) -> tuple[bool, str | None]:
        """
        核心图像生成方法（不yield，直接返回结果）

        Args:
            event: 消息事件
            prompt: 提示词
            reference_images: 从消息中提取的参考图片（base64）
            avatar_reference: 头像图片（base64）

        Returns:
            tuple[bool, str | None]: (是否成功, 结果消息或错误消息)
        """
        if not self.api_client:
            return False, "❌ 错误: API 客户端未初始化，请联系管理员配置 API 密钥"

        # 合并所有参考图片
        all_reference_images = []
        if reference_images:
            all_reference_images.extend(reference_images)
        if avatar_reference:
            all_reference_images.extend(avatar_reference)

        # 限制参考图片数量
        if (
            all_reference_images
            and len(all_reference_images) > self.max_reference_images
        ):
            logger.warning(
                f"参考图片数量 ({len(all_reference_images)}) 超过限制 ({self.max_reference_images})，将截取前 {self.max_reference_images} 张"
            )
            all_reference_images = all_reference_images[: self.max_reference_images]

        # 构建请求配置
        response_modalities = "TEXT_IMAGE" if self.enable_text_response else "IMAGE"
        request_config = ApiRequestConfig(
            model=self.model,
            prompt=prompt,
            api_type=self.api_type,
            api_base=self.api_base,
            resolution=self.resolution,
            aspect_ratio=self.aspect_ratio,
            enable_grounding=self.enable_grounding,
            response_modalities=response_modalities,
            reference_images=all_reference_images if all_reference_images else None,
        )

        # 日志记录
        self.log_info("🎨 图像生成请求:")
        self.log_info(f"  模型: {self.model}")
        self.log_info(f"  API 类型: {self.api_type}")
        self.log_info(
            f"  参考图片: {len(all_reference_images) if all_reference_images else 0} 张"
        )

        # 发送请求
        try:
            self.log_info("🚀 开始调用API生成图像...")
            start_time = asyncio.get_event_loop().time()

            # 计算合理的超时时间，确保总时间不超过 tool_call_timeout
            tool_timeout = self.get_tool_timeout(event)
            api_timeout = min(
                self.total_timeout, tool_timeout / max(self.max_attempts_per_key, 1)
            )
            logger.info(
                f"[TIMEOUT] tool_call_timeout={tool_timeout}s, api_timeout={api_timeout}s, max_retries={self.max_attempts_per_key}"
            )

            image_url, image_path, text_content = await self.api_client.generate_image(
                config=request_config,
                max_retries=self.max_attempts_per_key,
                total_timeout=api_timeout,
            )

            end_time = asyncio.get_event_loop().time()
            api_duration = end_time - start_time
            self.log_info(f"✅ API调用完成，耗时: {api_duration:.2f}秒")

            if image_path and Path(image_path).exists():
                # 文件传输（如果需要）
                if self.nap_server_address and self.nap_server_address != "localhost":
                    self.log_info("📤 检测到远程服务器配置，开始文件传输...")
                    from .utils.file_send_server import send_file

                    try:
                        remote_path = await asyncio.wait_for(
                            send_file(
                                image_path,
                                HOST=self.nap_server_address,
                                PORT=self.nap_server_port,
                            ),
                            timeout=10.0,
                        )
                        if remote_path:
                            image_path = remote_path
                    except asyncio.TimeoutError:
                        logger.warning("⚠️ 文件传输超时，使用本地文件")
                    except Exception as e:
                        logger.warning(f"⚠️ 文件传输失败: {e}，将使用本地文件")

                # 发送图片和文本（如果有）
                self.log_info("📨 准备发送结果...")

                result_components = []

                if text_content:
                    self.log_info(f"📝 检测到文本内容，长度: {len(text_content)} 字符")
                    result_components.append(event.plain_result(text_content).result)

                image_component = await self._send_image_with_fallback(image_path)
                result_components.append(image_component)

                await event.send(event.chain_result(result_components))

                return True, None
            else:
                error_msg = f"❌ 图像文件不存在或路径无效: {image_path}"
                logger.error(error_msg)
                return False, error_msg

        except APIError as e:
            error_msg = f"❌ 图像生成失败: {e.message}"
            if e.status_code == 429:
                error_msg += "\n💡 可能原因：API 速率限制或额度耗尽"
            elif e.status_code == 402:
                error_msg += "\n💡 可能原因：API 额度不足"
            elif e.status_code == 403:
                error_msg += "\n💡 可能原因：API 密钥无效或权限不足"
            logger.error(error_msg)
            return False, error_msg

        except Exception as e:
            logger.error(f"生成图像时发生未预期的错误: {e}", exc_info=True)
            return False, f"❌ 生成图像时发生错误: {str(e)}"

    @filter.llm_tool(name="gemini_image_generation")
    async def generate_image_tool(
        self,
        event: AstrMessageEvent,
        prompt: str,
        use_reference_images: str,
        include_user_avatar: str = "false",
        **kwargs,
    ):
        """
        使用 Gemini 模型生成或修改图像的高级工具

        当用户请求图像生成或绘画时，调用此函数。

        **重要判断逻辑：**

        1. **用户使用以下词语时，强烈建议设置 use_reference_images="true" 和 include_user_avatar="true"**：
           - "改成", "改为", "变成", "换成", "替换", "调整", "修改", "优化", "重做", "重新", "改图", "换风格"
           - "基于", "根据", "按照", "参考", "依照", "以...为基础", "以...为参考"
           - 当句子中出现"我的"、"我的头发"、"我的脸"等描述时，表示需要用户本人作为参考
           - 示例："把我的头发改成黑色", "把图片变成动漫风格", "根据我的头像生成图片", "让我的眼睛变大"

        2. **当用户说"按照我", "根据我", "基于我", "参考我", "我的头像", "我的"时**：
           - 必须设置 use_reference_images="true" 和 include_user_avatar="true"
           - 会自动获取当前用户的头像作为参考

        3. **当用户@某个用户时**：
           - 必须设置 use_reference_images="true" 和 include_user_avatar="true"
           - 会自动获取被@用户的头像作为参考

        4. **当用户消息中包含图片时**：
           - 如果用户明确说"基于这张图片", "修改这张图"等，设置 use_reference_images="true"
           - 图片可以是用户直接上传的，也可以是引用回复的消息中的图片

        **提示词优化指南：**

        1. **手办模型生成**：
        "请将此照片中的主要对象精确转换为写实的、杰作级别的 1/7 比例 PVC 手办。
        在手办旁边应放置一个盒子：盒子正面应有一个大型清晰的透明窗口，印有主要艺术作品、产品名称、品牌标志、条形码，以及一个小规格或真伪验证面板。盒子的角落还必须贴有小价签。同时，在后方放置一个电脑显示器，显示器屏幕需要显示该手办的 ZBrush 建模过程。
        在包装盒前方，手办应放置在圆形塑料底座上。手办必须有 3D 立体感和真实感，PVC 材质的纹理需要清晰表现。如果背景可以设置为室内场景，效果会更好。"

        2. **Q版手办模型**：
        "请将此照片中的主要对象精确转换为写实的、杰作级别的 1/7 比例 PVC 手办。
        在此手办的一侧后方，应放置一个盒子：在盒子正面，显示我输入的原始图像，带有主题艺术作品、产品名称、品牌标志、条形码，以及一个小规格或真伪验证面板。盒子的一个角落还必须贴有小价签。同时，在后方放置一个电脑显示器，显示器屏幕需要显示该手办的 ZBrush 建模过程。
        在包装盒前方，手办应放置在圆形塑料底座上。手办必须有 3D 立体感和真实感，PVC 材质的纹理需要清晰表现。如果背景可以设置为室内场景，效果会更好。"

        **质量要求：**
        - 修复任何缺失部分时，必须没有执行不佳的元素
        - 修复人体手办时（如适用），身体部位必须自然，动作必须协调，所有部位比例必须合理
        - 如果原始照片不是全身照，请尝试补充手办使其成为全身版本
        - 人物表情和动作必须与照片完全一致
        - 手办头部不应显得太大，腿部不应显得太短，手办不应看起来矮胖（除非明确是Q版设计）
        - 对于动物手办，应减少毛发的真实感和细节层次，使其更像手办而不是真实的原始生物
        - 不应有外轮廓线，手办绝不能是平面的
        - 注意近大远小的透视关系

        Args:
            prompt(string): 图像生成或修改的描述
            use_reference_images(string): 是否使用上下文中的参考图片（true/false）。当用户意图是"修改"、"变成"、"基于"、"改成"等时，必须设置为"true"
            include_user_avatar(string): 是否包含用户头像作为参考图像（true/false）。当用户说"根据我"、"我的头像"或明显需要用户本人图像时，设置为"true"
        """
        if not self.api_client:
            yield event.plain_result(
                "❌ 错误: API 客户端未初始化，请联系管理员配置 API 密钥"
            )
            return

        # 收集参考图片（从消息中提取的图片，包括当前消息和引用回复中的图片）
        reference_images = []
        if str(use_reference_images).lower() in {"true", "1", "yes", "y", "是"}:
            reference_images = await self._collect_reference_images(event)

        # 自动获取头像作为参考
        avatar_reference = []

        # 直接信任Gemini API的判断
        avatar_value = str(include_user_avatar).lower()
        self.log_info(f"[AVATAR_DEBUG] include_user_avatar参数: {avatar_value}")

        if avatar_value in {"true", "1", "yes", "y", "是"}:
            self.log_info("[AVATAR_DEBUG] Gemini API建议获取头像，开始获取...")
            try:
                avatar_reference = await self.get_avatar_reference(event)
                self.log_info(
                    f"[AVATAR_DEBUG] 头像获取完成，返回结果: {len(avatar_reference) if avatar_reference else 0} 个"
                )
            except Exception as e:
                logger.error(f"头像获取失败: {e}", exc_info=True)
                avatar_reference = []

            if avatar_reference:
                self.log_info(f"成功获取 {len(avatar_reference)} 个头像作为参考图像")
                for i, avatar in enumerate(avatar_reference):
                    self.log_info(f"  - 头像{i + 1}: {avatar[:50]}...")
            else:
                self.log_info("未能获取头像，继续使用其他参考图像或纯文本生成")
        else:
            self.log_info("[AVATAR_DEBUG] Gemini API未建议获取头像，跳过头像获取")

        # 调用核心生成方法
        success, error_msg = await self._generate_image_core(
            event=event,
            prompt=prompt,
            reference_images=reference_images,
            avatar_reference=avatar_reference,
        )

        if not success and error_msg:
            yield event.plain_result(error_msg)

        # 清理使用的头像缓存
        try:
            await self.avatar_manager.cleanup_used_avatars()
        except Exception as e:
            logger.warning(f"清理头像缓存失败: {e}")

    @filter.command_group("生图")
    def generate_group(self):
        """图像生成命令组"""
        pass

    @filter.command_group("改图")
    def modify_group(self):
        """图像修改命令组"""
        pass

    @generate_group.command("帮助")
    async def show_help(self, event: AstrMessageEvent):
        """显示插件使用帮助"""
        grounding_status = "✓ 启用" if self.enable_grounding else "✗ 禁用"
        smart_retry_status = "✓ 启用" if self.enable_smart_retry else "✗ 禁用"
        avatar_status = "✓ 启用" if self.auto_avatar_reference else "✗ 禁用"

        # 获取当前聊天环境的超时配置
        tool_timeout = self.get_tool_timeout(event)
        timeout_warning = ""
        if tool_timeout < 90:
            timeout_warning = f"\n⚠ 超时警告: 当前工具超时时间较短({tool_timeout}秒)\n→ 建议在框架配置中将 tool_call_timeout 设置为 90-120 秒"

        help_info = f"""🎨 Gemini 图像生成插件 - 使用帮助

【当前配置信息】
· 模型: {self.model}
· API 类型: {self.api_type}
· 自定义端点: {self.api_base or "默认"}
· API 密钥数: {len(self.api_keys)}
· 分辨率: {self.resolution}
· 长宽比: {self.aspect_ratio or "默认"}
· Google 搜索接地: {grounding_status}
· 最大参考图片: {self.max_reference_images}
· 文本响应: {"✓ 启用" if self.enable_text_response else "✗ 禁用"}
· 自动头像参考: {avatar_status}
· 智能重试: {smart_retry_status}
· 当前工具超时: {tool_timeout} 秒{timeout_warning}
· 每密钥最大重试: {self.max_attempts_per_key}
· API 总超时: {self.total_timeout} 秒

【指令使用方法】
1. 生图帮助 - 显示此帮助信息

2. 生图快速模式 <预设> <描述>
   使用预设参数快速生成图像
   预设: 头像(1:1)/海报(16:9)/壁纸(16:9)/卡片(3:2)/手机(9:16)
   示例: /生图快速模式 头像 可爱的猫

3. 改图 <描述>
   根据提示词修改或重做图像（直接输入改图命令）
   需要引用或上传图片作为参考
   示例: /改图 把头发改成红色
ran
4. 改图换风格 <风格> [描述]
   改变图像风格
   风格: 动漫/写实/水彩/油画等
   示例: /改图换风格 动漫
   示例: /改图换风格 油画 添加梦幻背景

5. 也可以直接使用自然语言与LLM对话，如:
   - 生成一张海边日落的图片
   - 把这张图片改成动漫风格
   - 根据我的头像生成一张手办

【进阶功能】
· 回复或引用图片时，会自动使用图片作为参考
· @某人可以使用该用户的头像作为参考
· 在提示词中包含"头像"等关键词，可自动获取头像
· 启用自动头像参考后，生图时自动使用发送者头像

【注意事项】
· 生成高质量图像可能需要较长时间
· 工具超时时间过短可能导致生成失败
· 建议添加多个API密钥以提高成功率
"""

        yield event.plain_result(help_info)

    @modify_group.command("")
    async def modify_image(self, event: AstrMessageEvent, prompt: str):
        """
                根据提示词修改或重做图像（默认命令）

                Args:
        yao            prompt: 修改描述a't"把头发改成红色"、"换个背景"、"画成动漫风格"等
        """
        # 对于改图，强制启用参考图像和头像检测
        reference_images = await self._collect_reference_images(event)
        avatar_reference = (
            await self.get_avatar_reference(event) if self.auto_avatar_reference else []
        )

        success, error_msg = await self._generate_image_core(
            event=event,
            prompt=f"根据参考图像修改图像：{prompt}",
            reference_images=reference_images,
            avatar_reference=avatar_reference,
        )

        if not success and error_msg:
            yield event.plain_result(error_msg)

    @modify_group.command("换风格")
    async def change_style(self, event: AstrMessageEvent, style: str, prompt: str = ""):
        """
        改变图像风格

        Args:
            style: 风格描述，如"动漫"、"写实"、"水彩"、"油画"等
            prompt: 额外的修改要求（可选）
        """
        full_prompt = f"将参考图像改为{style}风格"
        if prompt:
            full_prompt += f"，{prompt}"

        # 对于改图，强制启用参考图像和头像检测
        reference_images = await self._collect_reference_images(event)
        avatar_reference = (
            await self.get_avatar_reference(event) if self.auto_avatar_reference else []
        )

        success, error_msg = await self._generate_image_core(
            event=event,
            prompt=full_prompt,
            reference_images=reference_images,
            avatar_reference=avatar_reference,
        )

        if not success and error_msg:
            yield event.plain_result(error_msg)

    @generate_group.command("快速模式")
    async def quick_preset(self, event: AstrMessageEvent, preset: str, prompt: str):
        """
        使用预设参数快速生成图像

        Args:
            preset: 预设类型（头像/海报/壁纸/名片/手机）
            prompt: 图像描述
        """
        # 预设配置
        preset_configs = {
            "头像": {"resolution": "1K", "aspect_ratio": "1:1", "desc": "方形头像"},
            "poster": {"resolution": "2K", "aspect_ratio": "16:9", "desc": "横向海报"},
            "壁纸": {"resolution": "4K", "aspect_ratio": "16:9", "desc": "高清壁纸"},
            "card": {"resolution": "1K", "aspect_ratio": "3:2", "desc": "卡片式"},
            "mobile": {"resolution": "2K", "aspect_ratio": "9:16", "desc": "手机竖屏"},
        }

        # 支持中英文
        if preset not in preset_configs:
            presets_list = ", ".join(preset_configs.keys())
            yield event.plain_result(f"❌ 无效的预设。可用预设: {presets_list}")
            return

        preset_config = preset_configs[preset]

        yield event.plain_result(
            f"🎨 使用 {preset} 模式 ({preset_config['desc']}) 生成图像..."
        )

        # 临时修改配置
        old_resolution = self.resolution
        old_aspect_ratio = self.aspect_ratio

        try:
            self.resolution = preset_config["resolution"]
            self.aspect_ratio = preset_config["aspect_ratio"]

            # 调用核心生成函数
            reference_images = (
                await self._collect_reference_images(event)
                if "头像" in prompt or self.auto_avatar_reference
                else []
            )
            avatar_reference = (
                await self.get_avatar_reference(event)
                if ("头像" in prompt or self.auto_avatar_reference)
                else []
            )

            success, error_msg = await self._generate_image_core(
                event=event,
                prompt=prompt,
                reference_images=reference_images,
                avatar_reference=avatar_reference,
            )

            if not success and error_msg:
                yield event.plain_result(error_msg)

        finally:
            # 恢复原始配置
            self.resolution = old_resolution
            self.aspect_ratio = old_aspect_ratio

    async def terminate(self):
        """插件卸载时清理资源"""
        logger.info("🎨 Gemini 图像生成插件已卸载")
