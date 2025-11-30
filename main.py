"""
AstrBot Gemini 图像生成插件主文件
支持 Google 官方 API 和 OpenAI 兼容格式 API，提供生图和改图功能，支持智能头像参考
"""

from __future__ import annotations

import asyncio
import base64
import os
import time
import urllib.parse
from datetime import datetime
from pathlib import Path
from typing import Any

import aiohttp
from PIL import Image as PILImage
import yaml

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.message_components import At, Image, Reply
from astrbot.api.star import Context, Star, register
from astrbot.core.provider.entities import ProviderType

from .tl import create_zip, split_image
from .tl.enhanced_prompts import (
    enhance_prompt_for_figure,
    get_auto_modification_prompt,
    get_avatar_prompt,
    get_card_prompt,
    get_figure_prompt,
    get_generation_prompt,
    get_mobile_prompt,
    get_modification_prompt,
    get_poster_prompt,
    get_sticker_prompt,
    get_sticker_bbox_prompt,
    get_style_change_prompt,
    get_wallpaper_prompt,
)
from astrbot.api.provider import ProviderRequest
from .tl.tl_api import (
    APIClient,
    APIError,
    ApiRequestConfig,
    get_api_client,
)
from .tl.tl_utils import (
    AvatarManager,
    cleanup_old_images,
    download_qq_avatar,
    encode_file_to_base64,
    send_file,
)


@register(
    "astrbot_plugin_gemini_image_generation",
    "piexian",
    "Gemini图像生成插件，支持生图和改图，可以自动获取头像作为参考",
    "",
)
class GeminiImageGenerationPlugin(Star):
    def __init__(self, context: Context, config: dict[str, Any]):
        super().__init__(context)
        self.config = config
        # 从 metadata.yaml 读取版本号
        try:
            metadata_path = os.path.join(os.path.dirname(__file__), "metadata.yaml")
            with open(metadata_path, encoding="utf-8") as f:
                metadata = yaml.safe_load(f) or {}
                self.version = str(metadata.get("version", "")).strip()
        except Exception:
            self.version = ""
        if not self.version:
            self.version = "v1.0.0"
        self.api_client: APIClient | None = None
        self.avatar_manager = AvatarManager()
        self._cleanup_task: asyncio.Task | None = None

        # 加载配置
        self._load_config()

        # 启动定时清理任务
        self._start_cleanup_task()

    def _start_cleanup_task(self):
        """启动定时清理任务"""
        if self._cleanup_task and not self._cleanup_task.done():
            return

        async def cleanup_loop():
            while True:
                try:
                    await cleanup_old_images()
                    # 每30分钟执行一次
                    await asyncio.sleep(1800)
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.warning(f"清理任务异常: {e}")
                    await asyncio.sleep(300)

        self._cleanup_task = asyncio.create_task(cleanup_loop())
        logger.debug("定时清理任务已启动")

    async def terminate(self):
        """插件卸载/重载时调用"""
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            logger.debug("定时清理任务已停止")
        logger.info("🎨 Gemini 图像生成插件已卸载")

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
                prompt = event.wessage_str.lower()

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
                        download_qq_avatar(
                            str(user_id), f"mentioned_{user_id}", event=event
                        )
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
                        download_qq_avatar(
                            sender_id, f"sender_{sender_id}", event=event
                        )
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

    async def should_use_avatar(self, event: AstrMessageEvent) -> bool:
        """判断是否应该使用头像作为参考（只有在有@用户时才使用）"""
        logger.info(
            f"[AVATAR_DEBUG] 检查auto_avatar_reference: {self.auto_avatar_reference}"
        )
        if not self.auto_avatar_reference:
            return False

        # 检查是否有@用户
        mentioned_users = await self.parse_mentions(event)
        logger.info(f"[AVATAR_DEBUG] @用户数量: {len(mentioned_users)}")

        # 只有当有@用户时才获取头像
        return len(mentioned_users) > 0

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
        api_settings = self.config.get("api_settings", {})
        provider_id = api_settings.get("provider_id") or ""
        self.provider_id = provider_id
        self.vision_provider_id = api_settings.get("vision_provider_id") or ""
        # 视觉识别模型留空则使用提供商默认模型，这里不强制覆盖
        self.vision_model = (api_settings.get("vision_model") or "").strip()
        # 预先读取用户显式覆盖（如选择 openai、自定义 api_base/model）
        manual_api_type = (api_settings.get("api_type") or "").strip()
        manual_api_base = (api_settings.get("custom_api_base") or "").strip()
        manual_model = (api_settings.get("model") or "").strip()
        self.api_type = manual_api_type or ""
        self.api_base = manual_api_base
        self.model = manual_model or ""
        # 统一从 AstrBot 提供商读取密钥/端点/模型
        self.api_keys: list[str] = []

        image_settings = self.config.get("image_generation_settings", {})
        self.resolution = image_settings.get("resolution", "1K")
        self.aspect_ratio = image_settings.get("aspect_ratio", "1:1")
        self.enable_grounding = image_settings.get("enable_grounding", False)
        self.max_reference_images = image_settings.get("max_reference_images", 6)
        self.enable_text_response = image_settings.get("enable_text_response", False)
        self.enable_sticker_split = image_settings.get("enable_sticker_split", True)
        self.enable_sticker_zip = image_settings.get("enable_sticker_zip", False)
        self.preserve_reference_image_size = image_settings.get(
            "preserve_reference_image_size", False
        )
        self.enable_llm_crop = image_settings.get("enable_llm_crop", True)
        # 从配置中读取强制分辨率设置，默认为False
        self.force_resolution = image_settings.get("force_resolution", False)

        retry_settings = self.config.get("retry_settings", {})
        self.max_attempts_per_key = retry_settings.get("max_attempts_per_key", 3)
        self.enable_smart_retry = retry_settings.get("enable_smart_retry", True)
        self.total_timeout = retry_settings.get("total_timeout", 120)

        service_settings = self.config.get("service_settings", {})
        self.nap_server_address = service_settings.get(
            "nap_server_address", "localhost"
        )
        self.nap_server_port = service_settings.get("nap_server_port", 3658)
        self.auto_avatar_reference = service_settings.get(
            "auto_avatar_reference", False
        )
        self.verbose_logging = service_settings.get("verbose_logging", False)
        self.html_render_options = service_settings.get("html_render_options", {}) or {}
        try:
            quality_val = self.html_render_options.get("quality")
            if quality_val is not None:
                quality_int = int(quality_val)
                if 1 <= quality_int <= 100:
                    self.html_render_options["quality"] = quality_int
                else:
                    logger.warning("html_render_options.quality 超出范围(1-100)，已忽略")
                    self.html_render_options.pop("quality", None)
        except Exception:
            logger.warning("解析 html_render_options 失败，已忽略质量设置")
            self.html_render_options.pop("quality", None)

        limit_settings = self.config.get("limit_settings", {})
        raw_mode = str(limit_settings.get("group_limit_mode", "none")).lower()
        if raw_mode not in {"none", "whitelist", "blacklist"}:
            raw_mode = "none"
        self.group_limit_mode: str = raw_mode

        raw_group_list = limit_settings.get("group_limit_list", []) or []
        # 统一使用字符串形式保存群号，便于与 NapCat/QQ 等平台的群 ID 对齐
        self.group_limit_list: set[str] = {
            str(group_id).strip()
            for group_id in raw_group_list
            if str(group_id).strip()
        }

        self.enable_rate_limit: bool = bool(
            limit_settings.get("enable_rate_limit", False)
        )
        # 限流周期与次数做基础防御，避免异常配置导致错误
        period = limit_settings.get("rate_limit_period", 60)
        max_requests = limit_settings.get("max_requests_per_group", 5)
        try:
            self.rate_limit_period: int = max(int(period), 1)
        except (TypeError, ValueError):
            self.rate_limit_period = 60
        try:
            self.max_requests_per_group: int = max(int(max_requests), 1)
        except (TypeError, ValueError):
            self.max_requests_per_group = 5

        # 内部限流状态：按群维度统计请求时间戳
        self._rate_limit_buckets: dict[str, list[float]] = {}
        self._rate_limit_lock = asyncio.Lock()

        # 从 AstrBot 提供商管理器读取模型/密钥/端点
        try:
            provider_mgr = getattr(self.context, "provider_manager", None)
            provider = None
            if provider_mgr:
                if provider_id and hasattr(provider_mgr, "inst_map"):
                    provider = provider_mgr.inst_map.get(provider_id)
                if not provider:
                    provider = provider_mgr.get_using_provider(
                        ProviderType.CHAT_COMPLETION, None
                    )

            if provider:
                # 补全 provider_id，便于后续视觉识别调用
                if not self.provider_id:
                    self.provider_id = provider.provider_config.get("id", "")
                prov_type = str(provider.provider_config.get("type", "")).lower()
                # 如果用户未显式选择 api_type，则按提供商类型推断
                if not manual_api_type:
                    if "googlegenai" in prov_type or "gemini" in prov_type:
                        self.api_type = "google"
                    elif "openai" in prov_type:
                        self.api_type = "openai"
                    else:
                        logger.warning(
                            f"提供商 {provider.provider_config.get('id')} 类型 {prov_type} 非Gemini/OpenAI，可能无法生成图像"
                        )

                prov_model = (
                    provider.get_model()
                    or provider.provider_config.get("model_config", {}).get("model")
                )
                # 若用户未手填模型，则使用提供商模型
                if prov_model and not manual_model:
                    self.model = prov_model

                prov_keys = provider.get_keys() or []
                self.api_keys = [str(k).strip() for k in prov_keys if str(k).strip()]

                prov_base = provider.provider_config.get("api_base")
                # 若用户未手填自定义 base，则使用提供商 base
                if prov_base and not manual_api_base:
                    self.api_base = prov_base

                logger.info(
                    f"✓ 已从 AstrBot 提供商读取配置，类型={self.api_type} 模型={self.model} 密钥={len(self.api_keys)}"
                )
            else:
                logger.error("未找到可用的 AstrBot 提供商，无法读取模型/密钥，请在主配置中选择提供商")
        except Exception as e:
            logger.error(f"读取 AstrBot 提供商配置失败: {e}")

        if self.api_keys:
            self.api_client = get_api_client(self.api_keys)
            logger.info("✓ API 客户端已初始化")
            logger.info(f"  - 类型: {self.api_type}")
            logger.info(f"  - 模型: {self.model}")
            logger.info(f"  - 密钥数量: {len(self.api_keys)}")
            if self.api_base:
                logger.info(f"  - 自定义 API Base: {self.api_base}")
        else:
            logger.error("✗ 未读取到 API 密钥，请确认 AstrBot 提供商中已配置 key")

    async def _llm_detect_and_split(self, image_path: str) -> list[str]:
        """使用视觉 LLM 识别裁剪框后切割，失败返回空列表"""
        if not self.enable_llm_crop:
            logger.debug("[LLM_CROP] 已关闭视觉裁剪开关，跳过识别")
            return []

        # 若未单独配置视觉识别提供商，则不启用，以免占用生图模型
        if not self.vision_provider_id:
            logger.debug("[LLM_CROP] 未配置 vision_provider_id，跳过视觉裁剪")
            return []

        try:
            # 读取图片尺寸用于提示
            with PILImage.open(image_path) as img:
                width, height = img.size
            prompt = get_sticker_bbox_prompt(rows=6, cols=4)

            # 若图过大，先生成压缩副本以提升识别成功率
            image_urls: list[str] = []
            vision_input_path = image_path
            try:
                max_side = max(width, height)
                if max_side > 1200:
                    ratio = 1200 / max_side
                    new_w = int(width * ratio)
                    new_h = int(height * ratio)
                    img = img.resize((new_w, new_h))
                    tmp_path = Path("/tmp") / f"vision_crop_{Path(image_path).stem}.png"
                    img.save(tmp_path, format="PNG")
                    vision_input_path = str(tmp_path)
                    logger.debug(
                        f"[LLM_CROP] 生成压缩副本用于识别: {vision_input_path} ({new_w}x{new_h})"
                    )
            except Exception as e:
                logger.debug(f"[LLM_CROP] 压缩副本生成失败，使用原图: {e}")

            image_urls = [vision_input_path] if vision_input_path else []
            logger.info(
                f"[LLM_CROP] 调用视觉模型裁剪: provider={self.vision_provider_id} (使用默认模型)"
            )
            resp = await self.context.llm_generate(
                chat_provider_id=self.vision_provider_id,
                prompt=prompt,
                image_urls=image_urls,
                max_output_tokens=600,
                timeout=120, 
                on_llm_request=self._inject_vision_system_prompt,
            )
            text = self._extract_llm_text(resp)
            if not text:
                return []

            # 尝试解析 JSON 数组
            import json
            import re

            match = re.search(r"\[.*\]", text, re.S)
            json_str = match.group(0) if match else text
            json_str = json_str.replace("```json", "").replace("```", "").strip()
            bboxes = json.loads(json_str)
            if not isinstance(bboxes, list):
                return []

            # 过滤有效框
            clean_boxes = []
            for box in bboxes:
                try:
                    x = int(box.get("x", 0))
                    y = int(box.get("y", 0))
                    w = int(box.get("width", 0))
                    h = int(box.get("height", 0))
                except Exception:
                    continue
                if w > 0 and h > 0:
                    clean_boxes.append({"x": x, "y": y, "width": w, "height": h})

            if not clean_boxes:
                return []

            # 调用裁剪工具
            return await asyncio.to_thread(
                split_image,
                image_path,
                rows=6,
                cols=4,
                bboxes=clean_boxes,
            )
        except Exception as e:
            logger.debug(f"视觉识别裁剪失败: {e}")
            return []

    async def _inject_vision_system_prompt(
        self, event: AstrMessageEvent, req: ProviderRequest
    ):
        """为视觉裁剪请求注入 system_prompt，提示返回 JSON 裁剪框"""
        extra = (
            "你是视觉裁剪助手，只需按要求返回 JSON 数组，每个元素包含 x,y,width,height（像素）。"
            "禁止输出除 JSON 之外的任何内容。"
        )
        try:
            if req.system_prompt:
                req.system_prompt += "\n" + extra
            else:
                req.system_prompt = extra
        except Exception:
            pass

    def log_info(self, message: str):
        """根据配置输出info或debug级别日志"""
        if self.verbose_logging:
            logger.info(message)
        else:
            logger.debug(message)

    def log_debug(self, message: str):
        """输出debug级别日志"""
        logger.debug(message)

    @staticmethod
    def _is_valid_base64_image_str(value: str) -> bool:
        """粗略判断字符串是否为有效的 base64 图像数据或 data URL"""
        if not value:
            return False

        if value.startswith("data:image/"):
            return ";base64," in value

        try:
            base64.b64decode(value, validate=True)
            return True
        except Exception:
            return False

    @staticmethod
    def _clean_text_content(text: str) -> str:
        """清理文本内容，移除 markdown 图片链接等不可发送的内容"""
        if not text:
            return text

        import re

        text = re.sub(r"!\[.*?\]\(.*?\)", "", text)
        text = text.strip()

        return text

    @staticmethod
    def _extract_llm_text(resp: Any) -> str:
        """
        兼容 AstrBot LLMResponse 文本提取：
        - 优先 result_chain 中的 Plain 文本
        - 其次 output_text / response
        """
        try:
            if getattr(resp, "result_chain", None):
                chain = getattr(resp.result_chain, "chain", None)
                if isinstance(chain, list):
                    parts: list[str] = []
                    for comp in chain:
                        text_val = getattr(comp, "text", None)
                        if text_val:
                            parts.append(str(text_val))
                    if parts:
                        return " ".join(parts).strip()

            if getattr(resp, "output_text", None):
                return (resp.output_text or "").strip()
            if getattr(resp, "response", None):
                return (resp.response or "").strip()
        except Exception:
            return ""
        return ""

    def _filter_valid_reference_images(
        self, images: list[str] | None, source: str
    ) -> list[str]:
        """
        过滤出合法的 base64 / data URL 参考图像。

        NapCat 等平台的图片 file_id（例如 D127D0...jpg）会在这里被过滤掉，
        避免传给 Gemini 导致 Base64 解码错误。
        """
        if not images:
            return []

        valid: list[str] = []
        for img in images:
            if not isinstance(img, str) or not img:
                self.log_debug(f"跳过非字符串参考图像({source}): {type(img)}")
                continue

            if self._is_valid_base64_image_str(img):
                valid.append(img)
            else:
                self.log_debug(f"跳过非 base64 格式参考图像({source}): {img[:64]}...")

        return valid

    def _get_group_id_from_event(self, event: AstrMessageEvent) -> str | None:
        """从事件中解析群ID，仅在群聊场景下返回"""
        try:
            if hasattr(event, "group_id") and event.group_id:
                return str(event.group_id)
            message_obj = getattr(event, "message_obj", None)
            if message_obj and getattr(message_obj, "group_id", ""):
                return str(message_obj.group_id)
        except Exception as e:
            self.log_debug(f"获取群ID失败: {e}")
        return None

    async def _check_and_consume_limit(
        self, event: AstrMessageEvent
    ) -> tuple[bool, str | None]:
        """
        检查当前事件是否通过群聊黑/白名单和限流校验。

        返回:
            (是否允许继续执行, 不允许时的提示消息)
        """
        group_id = self._get_group_id_from_event(event)

        if not group_id:
            return True, None

        if self.group_limit_mode == "whitelist":
            if self.group_limit_list and group_id not in self.group_limit_list:
                return False, None
        elif self.group_limit_mode == "blacklist":
            if self.group_limit_list and group_id in self.group_limit_list:
                return False, None

        if not self.enable_rate_limit:
            return True, None

        now = time.monotonic()
        window_start = now - self.rate_limit_period

        async with self._rate_limit_lock:
            bucket = self._rate_limit_buckets.get(group_id, [])
            bucket = [ts for ts in bucket if ts >= window_start]

            if len(bucket) >= self.max_requests_per_group:
                earliest = bucket[0]
                retry_after = int(earliest + self.rate_limit_period - now)
                if retry_after < 0:
                    retry_after = 0

                self._rate_limit_buckets[group_id] = bucket
                return (
                    False,
                    f"⏱️ 本群在最近 {self.rate_limit_period} 秒内的生图请求次数已达上限（{self.max_requests_per_group} 次），请约 {retry_after} 秒后再试。",
                )

            bucket.append(now)
            self._rate_limit_buckets[group_id] = bucket

        return True, None

    async def initialize(self):
        """插件初始化"""
        if self.api_client:
            logger.info("🎨 Gemini 图像生成插件已加载")
        else:
            logger.error("✗ API 客户端初始化失败，请检查配置")

    async def _download_qq_image(self, url: str) -> str | None:
        """对QQ图床做特殊处理，补充Referer/UA后转为base64"""
        try:
            parsed = urllib.parse.urlparse(url)
            headers = {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0 Safari/537.36"
                ),
                "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
                "Connection": "keep-alive",
            }
            if parsed.netloc:
                headers["Referer"] = f"{parsed.scheme}://{parsed.netloc}"
            if "qpic.cn" in (parsed.netloc or ""):
                headers["Referer"] = "https://qun.qq.com"

            timeout = aiohttp.ClientTimeout(total=12, connect=5)
            async with aiohttp.ClientSession(headers=headers, trust_env=True) as session:
                async with session.get(url, timeout=timeout) as resp:
                    if resp.status != 200:
                        logger.warning(
                            f"QQ图片下载失败: HTTP {resp.status} {resp.reason} | {url[:80]}"
                        )
                        return None
                    data = await resp.read()
                    if not data:
                        logger.warning(f"QQ图片为空: {url[:80]}")
                        return None
                    mime = resp.headers.get("Content-Type", "image/jpeg")
                    if ";" in mime:
                        mime = mime.split(";", 1)[0]
                    base64_data = base64.b64encode(data).decode("utf-8")
                    return f"data:{mime};base64,{base64_data}"
        except Exception as e:
            logger.warning(f"QQ图片下载异常: {e} | {url[:80]}")
            return None

    async def _fetch_images_from_event(
        self, event: AstrMessageEvent, include_at_avatars: bool = False
    ) -> tuple[list[str], list[str]]:
        """
        综合提取事件中的图片：当前消息、引用消息及手动@用户头像

        返回 (消息/引用图片, 头像图片)
        """
        message_images: list[str] = []
        avatar_images: list[str] = []
        seen_sources: set[str] = set()
        seen_users: set[str] = set()
        conversion_cache: dict[str, str] = {}
        max_images = self.max_reference_images

        if not hasattr(event, "message_obj") or not event.message_obj:
            return message_images, avatar_images

        try:
            message_chain = event.get_messages()
        except Exception:
            message_chain = getattr(event.message_obj, "message", []) or []

        if not message_chain:
            return message_images, avatar_images

        self_id = None
        try:
            self_id = str(event.get_self_id())
        except Exception:
            try:
                self_id = str(getattr(event.message_obj, "self_id", None))
            except Exception:
                self_id = None

        def _is_auto_at(comp: At) -> bool:
            """区分自动@，兼容多种属性命名"""
            flags = [
                getattr(comp, "is_auto", None),
                getattr(comp, "auto", None),
                getattr(comp, "auto_at", None),
                getattr(comp, "autoAt", None),
            ]
            for flag in flags:
                if isinstance(flag, str):
                    flag_val = flag.lower() in {"true", "1", "yes", "y"}
                else:
                    flag_val = bool(flag)
                if flag_val:
                    return True
            return False

        async def convert_to_base64(img_source: str, origin: str) -> str | None:
            """将图片源转换为base64格式，内置QQ直链强化处理"""
            if not img_source:
                return None
            if img_source in conversion_cache:
                return conversion_cache[img_source]

            parsed_host = ""
            try:
                parsed_host = urllib.parse.urlparse(str(img_source)).netloc or ""
            except Exception:
                parsed_host = ""

            if parsed_host and "qpic.cn" in parsed_host:
                qq_data = await self._download_qq_image(str(img_source))
                if qq_data:
                    conversion_cache[img_source] = qq_data
                    return qq_data
                logger.warning(f"QQ图片直链处理失败，尝试通用流程: {img_source[:80]}")

            try:
                if not self.api_client:
                    logger.warning("API 客户端未初始化，无法转换图片为base64")
                    return None
                mime_type, base64_data = await self.api_client._normalize_image_input(
                    img_source
                )
                if base64_data:
                    data_url = (
                        f"data:{mime_type};base64,{base64_data}"
                        if mime_type
                        else base64_data
                    )
                    conversion_cache[img_source] = data_url
                    return data_url
                logger.debug(f"跳过无法识别的图片源({origin}): {str(img_source)[:80]}...")
            except Exception as e:
                logger.warning(
                    f"转换图片为base64失败({origin}): {repr(e)} | Source: {str(img_source)[:80]}"
                )
            return None

        async def handle_image_component(component, origin: str):
            if len(message_images) >= max_images:
                return

            img_source = None
            if isinstance(component, Image):
                if getattr(component, "url", None):
                    img_source = component.url
                elif getattr(component, "file", None):
                    img_source = component.file
            else:
                if getattr(component, "url", None):
                    img_source = component.url
                elif getattr(component, "file", None):
                    img_source = component.file

            if not img_source:
                return

            if img_source in seen_sources:
                self.log_debug(f"跳过重复图片源({origin}): {str(img_source)[:120]}")
                return

            seen_sources.add(img_source)
            base64_img = await convert_to_base64(str(img_source), origin)
            if base64_img:
                message_images.append(base64_img)
                self.log_debug(
                    f"✓ 从{origin}提取图片 (当前: {len(message_images)}/{max_images})"
                )

        async def handle_at_component(component: At, origin: str):
            if not include_at_avatars:
                return

            if _is_auto_at(component):
                self.log_debug(f"跳过自动@用户（{origin}）")
                return

            user_id = getattr(component, "qq", None) or getattr(
                component, "user_id", None
            )
            if not user_id:
                return

            user_id = str(user_id)
            if self_id and user_id == self_id:
                return
            if user_id in seen_users:
                return

            avatar_b64 = await self.avatar_manager.get_avatar(
                user_id, f"at_{user_id}", event=event
            )
            if avatar_b64:
                avatar_images.append(avatar_b64)
                seen_users.add(user_id)
                self.log_debug(f"✓ 获取@用户头像({origin}): {user_id}")
            else:
                self.log_debug(f"✗ 获取@用户头像失败({origin}): {user_id}")

        # 当前消息体处理
        for component in message_chain:
            try:
                if isinstance(component, Image):
                    await handle_image_component(component, "当前消息")
                elif isinstance(component, At):
                    await handle_at_component(component, "当前消息")
                elif isinstance(component, Reply) and component.chain:
                    for reply_comp in component.chain:
                        if isinstance(reply_comp, Image):
                            await handle_image_component(reply_comp, "引用消息")
                        elif isinstance(reply_comp, At):
                            await handle_at_component(reply_comp, "引用消息")
            except Exception as e:
                logger.warning(f"处理消息组件异常: {e}")

        # 如果需要头像但没有@，尝试回退到发送者头像
        if include_at_avatars and not avatar_images:
            try:
                sender_id = None
                if hasattr(event, "message_obj") and hasattr(
                    event.message_obj, "sender"
                ):
                    sender = event.message_obj.sender
                    sender_id = getattr(sender, "user_id", None) or getattr(
                        sender, "userId", None
                    )
                if sender_id and str(sender_id) not in seen_users:
                    sender_id = str(sender_id)
                    avatar_b64 = await self.avatar_manager.get_avatar(
                        sender_id, f"sender_{sender_id}", event=event
                    )
                    if avatar_b64:
                        avatar_images.append(avatar_b64)
                        seen_users.add(sender_id)
                        self.log_debug(f"✓ 回退获取发送者头像: {sender_id}")
            except Exception as e:
                logger.debug(f"回退获取发送者头像失败: {e}")

        # 截断数量，优先保留消息图片，再补充头像
        if len(message_images) > max_images:
            message_images = message_images[:max_images]
        remaining_slots = max(max_images - len(message_images), 0)
        if len(avatar_images) > remaining_slots:
            avatar_images = avatar_images[:remaining_slots]

        if message_images or avatar_images:
            logger.info(
                f"📸 已收集图片: 消息 {len(message_images)} 张，头像 {len(avatar_images)} 张"
            )
        else:
            logger.info("📸 未收集到有效参考图片，若需参考图可直接发送图片或检查网络权限")

        return message_images, avatar_images

    async def _generate_image_core_internal(
        self,
        event: AstrMessageEvent,
        prompt: str,
        reference_images: list[str],
        avatar_reference: list[str],
    ) -> tuple[bool, tuple[list[str], list[str], str | None, str | None] | str]:
        """
        内部核心图像生成方法，不发送消息，只返回结果

        Returns:
            tuple[bool, tuple[list[str], list[str], str | None, str | None] | str]:
            (是否成功, (图片URL列表, 图片路径列表, 文本内容, 思维签名) 或错误消息)
        """
        if not self.api_client:
            return False, (
                "❌ 无法生成图像：API 客户端尚未初始化。\n"
                "🧐 可能原因：API 配置或密钥缺失、加载失败。\n"
                "✅ 建议：先在配置文件中填写有效的 API 密钥并重启服务。"
            )

        valid_msg_images = self._filter_valid_reference_images(
            reference_images, source="消息图片"
        )
        valid_avatar_images = self._filter_valid_reference_images(
            avatar_reference, source="头像"
        )
        all_reference_images = valid_msg_images + valid_avatar_images

        if (
            all_reference_images
            and len(all_reference_images) > self.max_reference_images
        ):
            logger.warning(
                f"参考图片数量 ({len(all_reference_images)}) 超过限制 ({self.max_reference_images})，将截取前 {self.max_reference_images} 张"
            )
            all_reference_images = all_reference_images[: self.max_reference_images]

        # 计算截断后的数量
        final_msg_count = min(len(valid_msg_images), len(all_reference_images))
        final_avatar_count = len(all_reference_images) - final_msg_count

        if final_avatar_count > 0:
            prompt += f"""

[System Note]
The last {final_avatar_count} image(s) provided are User Avatars (marked as optional reference). You may use them for character consistency if needed, but they are NOT mandatory if they conflict with the requested style."""

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
            enable_smart_retry=self.enable_smart_retry,
            enable_text_response=self.enable_text_response,
            force_resolution=self.force_resolution,
            verbose_logging=self.verbose_logging,
        )

        logger.info("🎨 图像生成请求:")
        logger.info(f"  模型: {self.model}")
        logger.info(f"  API 类型: {self.api_type}")
        logger.info(
            f"  参考图片: {len(all_reference_images) if all_reference_images else 0} 张"
        )

        try:
            logger.info("🚀 开始调用API生成图像...")
            start_time = asyncio.get_event_loop().time()

            tool_timeout = self.get_tool_timeout(event)
            per_retry_timeout = min(self.total_timeout, tool_timeout)
            max_total_time = tool_timeout
            logger.info(
                f"[TIMEOUT] tool_call_timeout={tool_timeout}s, per_retry_timeout={per_retry_timeout}s, max_retries={self.max_attempts_per_key}, max_total_time={max_total_time}s"
            )

            (
                image_urls,
                image_paths,
                text_content,
                thought_signature,
            ) = await self.api_client.generate_image(
                config=request_config,
                max_retries=self.max_attempts_per_key,
                per_retry_timeout=per_retry_timeout,
                max_total_time=max_total_time,
            )

            end_time = asyncio.get_event_loop().time()
            api_duration = end_time - start_time
            logger.info(f"✅ API调用完成，耗时: {api_duration:.2f}秒")
            logger.info(
                f"🖼️ API 返回图片数量: {len(image_paths)}, URL 数量: {len(image_urls)}"
            )

            if thought_signature:
                logger.debug(f"🧠 思维签名: {thought_signature[:50]}...")

            resolved_paths: list[str] = []
            for idx, img_path in enumerate(image_paths):
                if not img_path:
                    continue
                if Path(img_path).exists():
                    resolved_path = img_path
                    if self.nap_server_address and self.nap_server_address != "localhost":
                        logger.info(f"📤 开始传输第 {idx + 1} 张图片到远程服务器...")
                        try:
                            remote_path = await asyncio.wait_for(
                                send_file(
                                    img_path,
                                    host=self.nap_server_address,
                                    port=self.nap_server_port,
                                ),
                                timeout=10.0,
                            )
                            if remote_path:
                                resolved_path = remote_path
                        except asyncio.TimeoutError:
                            logger.warning("⚠️ 文件传输超时，使用本地文件")
                        except Exception as e:
                            logger.warning(f"⚠️ 文件传输失败: {e}，将使用本地文件")
                    resolved_paths.append(resolved_path)
                else:
                    logger.warning(f"⚠️ 图像文件不存在或不可访问: {img_path}")
                    resolved_paths.append(img_path)

            image_paths = resolved_paths

            available_paths = [p for p in image_paths if p]
            available_urls = [u for u in image_urls if u]
            if available_paths or available_urls:
                logger.info(
                    f"📨 图像生成完成，准备返回结果，文件路径 {len(available_paths)} 张，URL {len(available_urls)} 张"
                )
                return True, (
                    image_urls,
                    image_paths,
                    text_content,
                    thought_signature,
                )

            error_msg = (
                "❌ 图像文件未找到，无法返回结果。\n"
                "🧐 可能原因：生成后保存文件失败，或远程传输路径无效。\n"
                "✅ 建议：检查临时目录写入权限与磁盘空间，必要时重试。"
            )
            logger.error(error_msg)
            return False, error_msg

        except APIError as e:
            status_part = f"（状态码 {e.status_code}）" if e.status_code is not None else ""
            error_msg = f"❌ 图像生成失败{status_part}：{e.message}"
            if e.status_code == 429:
                error_msg += "\n🧐 可能原因：请求过于频繁或额度已用完。\n✅ 建议：稍等片刻再试，或在配置中增加可用额度/开启智能重试。"
            elif e.status_code == 402:
                error_msg += "\n🧐 可能原因：账户余额不足或套餐到期。\n✅ 建议：充值或更换一组可用的 API 密钥后再试。"
            elif e.status_code == 403:
                error_msg += "\n🧐 可能原因：API 密钥无效、权限不足或访问受限。\n✅ 建议：核对密钥权限、检查 IP 白名单，必要时重新生成密钥。"
            elif e.status_code and 500 <= e.status_code < 600:
                error_msg += "\n🧐 可能原因：上游服务暂时不可用。\n✅ 建议：稍后重试，若频繁出现请联系服务提供方确认故障。"
            else:
                error_msg += "\n🧐 可能原因：请求参数异常或服务返回未知错误。\n✅ 建议：简化提示词/减少参考图后重试，并查看日志获取更多细节。"
            logger.error(error_msg)
            return False, error_msg

        except Exception as e:
            logger.error(f"生成图像时发生未预期的错误: {e}", exc_info=True)
            return False, f"❌ 生成图像时发生错误: {str(e)}"

    def _merge_available_images(
        self, image_paths: list[str] | None, image_urls: list[str] | None
    ) -> list[str]:
        """合并路径与URL，保持顺序并去重，避免同一图重复发送"""
        merged: list[str] = []
        seen: set[str] = set()

        for img in (image_paths or []) + (image_urls or []):
            if not img:
                continue
            if img in seen:
                continue
            seen.add(img)
            merged.append(img)

        return merged

    def _build_forward_image_component(self, image: str):
        """根据来源构造合并转发图片组件，优先使用本地文件"""
        from astrbot.api.message_components import Image as AstrImage
        from astrbot.api.message_components import Plain

        try:
            if not image:
                raise ValueError("空的图片地址")

            fs_candidate = image
            if image.startswith("file:///"):
                fs_candidate = image[8:]

            if os.path.exists(fs_candidate):
                return AstrImage.fromFileSystem(fs_candidate)
            if image.startswith(("http://", "https://")):
                return AstrImage.fromURL(image)

            return AstrImage(file=image)
        except Exception as e:
            logger.warning(f"构造图片组件失败: {e}")
            return Plain(f"[图片不可用: {image[:48]}]")

    async def _dispatch_send_results(
        self,
        event: AstrMessageEvent,
        image_urls: list[str] | None,
        image_paths: list[str] | None,
        text_content: str | None,
        thought_signature: str | None = None,
        scene: str = "默认",
    ):
        """
        根据内容数量选择发送模式：
        - 单图：链式富媒体发送（文本+图一起）
        - 总数<=4：链式富媒体发送（文本+多图一起）
        - 总数>4：合并转发
        """
        from astrbot.api import message_components as Comp

        cleaned_text = self._clean_text_content(text_content) if text_content else ""
        text_to_send = cleaned_text if (self.enable_text_response and cleaned_text) else ""

        available_images = self._merge_available_images(image_paths, image_urls)
        total_items = len(available_images) + (1 if text_to_send else 0)

        logger.info(
            f"[SEND] 场景={scene}，图片={len(available_images)}，文本={'1' if text_to_send else '0'}，总计={total_items}"
        )

        if not available_images:
            if cleaned_text:
                yield event.plain_result("⚠️ 当前模型只返回了文本，请检查模型配置或者重试")
                if text_to_send:
                    yield event.plain_result(f"📝 {text_to_send}")
            else:
                yield event.plain_result(
                    "❌ 未能成功生成图像。\n"
                    "🧐 可能原因：模型返回空结果、提示词冲突或参考图处理异常。\n"
                    "✅ 建议：简化描述、减少参考图数量后再试，或稍后重试。"
                )
            return

        # 单图直发
        if len(available_images) == 1:
            logger.info("[SEND] 采用单图直发模式")
            if text_to_send:
                # 富媒体链式发送：文本+图片
                yield event.chain_result(
                    [
                        Comp.Plain(f"\u200b📝 {text_to_send}"),
                        self._build_forward_image_component(available_images[0]),
                    ]
                )
            else:
                yield event.image_result(available_images[0])
            if thought_signature:
                logger.debug(f"🧠 思维签名: {thought_signature[:50]}...")
            return

        # 短链顺序发送
        if total_items <= 4:
            logger.info("[SEND] 采用短链富媒体发送模式")
            chain: list = []
            if text_to_send:
                chain.append(Comp.Plain(f"\u200b📝 {text_to_send}"))
            for img in available_images:
                chain.append(self._build_forward_image_component(img))
            if chain:
                yield event.chain_result(chain)
            if thought_signature:
                logger.debug(f"🧠 思维签名: {thought_signature[:50]}...")
            return

        # 合并转发
        logger.info("[SEND] 采用合并转发模式")
        from astrbot.api.message_components import Node, Plain, Image as AstrImage

        node_content = []
        if text_to_send:
            node_content.append(Plain(f"📝 {text_to_send}"))

        for idx, img in enumerate(available_images, 1):
            node_content.append(Plain(f"图片 {idx}:"))
            # 直接使用 Image 组件构建群合并转发节点
            try:
                img_component = None
                if img.startswith("file:///"):
                    fs_path = img[8:]
                    img_component = AstrImage.fromFileSystem(fs_path)
                elif os.path.exists(img):
                    img_component = AstrImage.fromFileSystem(img)
                elif img.startswith(("http://", "https://")):
                    img_component = AstrImage.fromURL(img)
                else:
                    img_component = AstrImage(file=img)

                node_content.append(img_component)
            except Exception as e:
                logger.warning(f"构造合并转发图片节点失败: {e}")
                node_content.append(Plain(f"[图片不可用: {img[:48]}]"))

        sender_id = "0"
        sender_name = "Gemini图像生成"
        try:
            if hasattr(event, "message_obj") and getattr(event, "message_obj", None):
                sender_id = getattr(event.message_obj, "self_id", "0")
        except Exception:
            pass

        node = Node(uin=sender_id, name=sender_name, content=node_content)
        # 群合并转发需用 chain_result 包裹 Node
        yield event.chain_result([node])

        if thought_signature:
            logger.debug(f"🧠 思维签名: {thought_signature[:50]}...")

    async def _quick_generate_image(
        self,
        event: AstrMessageEvent,
        prompt: str,
        use_avatar: bool = False,
        skip_figure_enhance: bool = False,
    ):
        """快捷图像生成"""
        if not self.api_client:
            yield event.plain_result("❌ API 客户端未初始化")
            return

        try:
            ref_images, avatars = await self._fetch_images_from_event(
                event, include_at_avatars=use_avatar
            )
            self.log_debug(
                f"[MODIFY_DEBUG] 收集到消息图片 {len(ref_images)} 张，头像 {len(avatars)} 个"
            )

            all_ref_images: list[str] = []
            all_ref_images.extend(
                self._filter_valid_reference_images(ref_images, source="消息图片")
            )
            if use_avatar:
                all_ref_images.extend(
                    self._filter_valid_reference_images(avatars, source="头像")
                )

            self.log_debug(f"[MODIFY_DEBUG] 有效参考图片总数: {len(all_ref_images)}")

            # 改图提示词增强 - 检测是否包含修改意图关键词
            modify_keywords = [
                "修改",
                "改图",
                "改成",
                "变成",
                "调整",
                "优化",
                "重做",
                "更换",
                "替换",
                "删除",
                "添加",
            ]
            is_modification_request = any(
                keyword in prompt for keyword in modify_keywords
            )
            self.log_debug(f"[MODIFY_DEBUG] 修改关键词匹配: {is_modification_request}")

            figure_keywords = ["手办", "figure", "模型", "手办化", "手办模型"]
            if (not skip_figure_enhance) and any(
                keyword in prompt.lower() for keyword in figure_keywords
            ):
                enhanced_prompt = enhance_prompt_for_figure(prompt)
                self.log_debug("[MODIFY_DEBUG] 使用手办化提示词增强")
            elif is_modification_request:
                # 对于改图请求，进一步强化提示词
                enhanced_prompt = get_auto_modification_prompt(prompt)
                self.log_debug("[MODIFY_DEBUG] 使用改图提示词增强")
            else:
                enhanced_prompt = prompt

            effective_resolution = self.resolution
            effective_aspect_ratio = self.aspect_ratio

            if (
                self.preserve_reference_image_size
                and is_modification_request
                and all_ref_images
            ):
                effective_resolution = None
                effective_aspect_ratio = None
                self.log_debug("[MODIFY_DEBUG] 保留参考图尺寸，不覆盖分辨率/比例")

            config = ApiRequestConfig(
                model=self.model,
                prompt=enhanced_prompt,
                api_type=self.api_type,
                api_base=self.api_base if self.api_base else None,
                resolution=effective_resolution,
                aspect_ratio=effective_aspect_ratio,
                enable_grounding=self.enable_grounding,
                reference_images=all_ref_images if all_ref_images else None,
                enable_smart_retry=self.enable_smart_retry,
                enable_text_response=self.enable_text_response,
                verbose_logging=self.verbose_logging,
            )

            # 记录改图请求的详细信息
            self.log_debug("[MODIFY_DEBUG] API请求配置:")
            self.log_debug(f"  - 提示词: {enhanced_prompt[:100]}...")
            self.log_debug(
                f"  - 参考图片数量: {len(all_ref_images) if all_ref_images else 0}"
            )
            self.log_debug(f"  - 是否改图请求: {is_modification_request}")
            self.log_debug(f"  - 模型: {self.model}")

            yield event.plain_result("🎨 生成中...")

            (
                image_urls,
                image_paths,
                text_content,
                thought_signature,
            ) = await self.api_client.generate_image(
                config=config,
                max_retries=self.max_attempts_per_key,
                per_retry_timeout=self.total_timeout,
                max_total_time=self.total_timeout * 2,
            )

            async for send_res in self._dispatch_send_results(
                event=event,
                image_urls=image_urls,
                image_paths=image_paths,
                text_content=text_content,
                thought_signature=thought_signature,
                scene="快捷生成",
            ):
                yield send_res

        except Exception as e:
            logger.error(f"快捷生成失败: {e}", exc_info=True)
            yield event.plain_result(
                f"❌ 快速生成时出现异常：{str(e)}\n"
                "🧐 可能原因：网络波动、配置缺失或依赖加载失败。\n"
                "✅ 建议：稍后重试，并检查 API 配置与日志定位具体问题。"
            )
        finally:
            try:
                await self.avatar_manager.cleanup_used_avatars()
            except Exception as e:
                logger.warning(f"清理头像缓存失败: {e}")

    def _enhance_prompt_for_figure(self, prompt: str) -> str:
        """手办化提示词增强（已废弃，保留兼容性）"""
        return enhance_prompt_for_figure(prompt)

    @filter.command("生图")
    async def generate_image(self, event: AstrMessageEvent, prompt: str):
        """
        生图指令

        Args:
            prompt: 图像描述
        """
        allowed, limit_message = await self._check_and_consume_limit(event)
        if not allowed:
            if limit_message:
                yield event.plain_result(limit_message)
            return

        use_avatar = await self.should_use_avatar(event)

        generation_prompt = get_generation_prompt(prompt)

        yield event.plain_result("🎨 开始生成图像...")

        async for result in self._quick_generate_image(
            event, generation_prompt, use_avatar
        ):
            yield result

    async def _handle_quick_mode(
        self,
        event: AstrMessageEvent,
        prompt: str,
        resolution: str,
        aspect_ratio: str,
        mode_name: str,
        prompt_func: Any = None,
        **kwargs,
    ):
        """处理快速模式的通用逻辑"""
        allowed, limit_message = await self._check_and_consume_limit(event)
        if not allowed:
            if limit_message:
                yield event.plain_result(limit_message)
            return

        yield event.plain_result(f"🎨 使用{mode_name}模式生成图像...")

        old_resolution = self.resolution
        old_aspect_ratio = self.aspect_ratio

        try:
            self.resolution = resolution
            self.aspect_ratio = aspect_ratio

            # 使用新提示词函数
            if prompt_func:
                full_prompt = prompt_func(prompt)
            else:
                full_prompt = prompt

            use_avatar = await self.should_use_avatar(event)

            async for result in self._quick_generate_image(
                event, full_prompt, use_avatar, **kwargs
            ):
                yield result

        finally:
            self.resolution = old_resolution
            self.aspect_ratio = old_aspect_ratio

    @filter.command_group("快速")
    def quick_mode_group(self):
        """快速模式指令组"""
        pass

    @quick_mode_group.command("头像")
    async def quick_avatar(self, event: AstrMessageEvent, prompt: str):
        """头像快速模式 - 1K分辨率，1:1比例"""
        async for result in self._handle_quick_mode(
            event, prompt, "1K", "1:1", "头像", get_avatar_prompt
        ):
            yield result

    @quick_mode_group.command("海报")
    async def quick_poster(self, event: AstrMessageEvent, prompt: str):
        """海报快速模式 - 2K分辨率，16:9比例"""
        async for result in self._handle_quick_mode(
            event, prompt, "2K", "16:9", "海报", get_poster_prompt
        ):
            yield result

    @quick_mode_group.command("壁纸")
    async def quick_wallpaper(self, event: AstrMessageEvent, prompt: str):
        """壁纸快速模式 - 4K分辨率，16:9比例"""
        async for result in self._handle_quick_mode(
            event, prompt, "4K", "16:9", "壁纸", get_wallpaper_prompt
        ):
            yield result

    @quick_mode_group.command("卡片")
    async def quick_card(self, event: AstrMessageEvent, prompt: str):
        """卡片快速模式 - 1K分辨率，3:2比例"""
        async for result in self._handle_quick_mode(
            event, prompt, "1K", "3:2", "卡片", get_card_prompt
        ):
            yield result

    @quick_mode_group.command("手机")
    async def quick_mobile(self, event: AstrMessageEvent, prompt: str):
        """手机快速模式 - 2K分辨率，9:16比例"""
        async for result in self._handle_quick_mode(
            event, prompt, "2K", "9:16", "手机", get_mobile_prompt
        ):
            yield result

    @quick_mode_group.command("手办化")
    async def quick_figure(self, event: AstrMessageEvent, prompt: str):
        """手办化快速模式 - 树脂收藏级手办效果"""
        # 解析参数
        style_type = 1
        clean_prompt = prompt

        if prompt:
            p_lower = prompt.lower()
            if p_lower.startswith("1") or "pvc" in p_lower:
                style_type = 1
                clean_prompt = prompt.replace("1", "", 1).replace("pvc", "", 1).strip()
            elif p_lower.startswith("2") or "gk" in p_lower:
                style_type = 2
                clean_prompt = prompt.replace("2", "", 1).replace("gk", "", 1).strip()

        full_prompt = get_figure_prompt(clean_prompt, style_type)

        async for result in self._handle_quick_mode(
            event,
            full_prompt,
            "2K",
            "3:2",
            "手办化",
            None,
            skip_figure_enhance=True,
        ):
            yield result

    @quick_mode_group.command("表情包")
    async def quick_sticker(self, event: AstrMessageEvent, prompt: str = ""):
        """表情包快速模式 - 4K分辨率，16:9比例，Q版LINE风格

        功能受配置文件控制：
        - enable_sticker_split: 是否自动切割图片
        - enable_sticker_zip: 是否打包发送（如果发送失败则使用合并转发）
        """
        allowed, limit_message = await self._check_and_consume_limit(event)
        if not allowed:
            if limit_message:
                yield event.plain_result(limit_message)
            return

        yield event.plain_result("🎨 使用表情包模式生成图像...")

        use_avatar = await self.should_use_avatar(event)
        reference_images, avatar_reference = await self._fetch_images_from_event(
            event, include_at_avatars=use_avatar
        )

        if not reference_images:
            yield event.plain_result(
                "❌ 表情包模式需要参考图才能生成一致的角色。\n"
                "🧐 可能原因：消息中未附带图片，或图片格式/大小不被支持。\n"
                "✅ 建议：请附上一张清晰的角色参考图（如头像或原表情）后再试。"
            )
            return

        # 如果没有开启切割功能，直接使用默认逻辑
        if not self.enable_sticker_split:
            full_prompt = get_sticker_prompt(prompt)
            old_resolution = self.resolution
            old_aspect_ratio = self.aspect_ratio

            try:
                self.resolution = "4K"
                self.aspect_ratio = "16:9"
                async for result in self._quick_generate_image(
                    event, full_prompt, use_avatar
                ):
                    yield result
            finally:
                self.resolution = old_resolution
                self.aspect_ratio = old_aspect_ratio
            return

        # 开启了切割功能，执行自定义逻辑
        full_prompt = get_sticker_prompt(prompt)
        old_resolution = self.resolution
        old_aspect_ratio = self.aspect_ratio

        try:
            self.resolution = "4K"
            self.aspect_ratio = "16:9"

            # 调用生图核心逻辑，但截获结果不直接发送
            sent_success = False
            split_files: list[str] = []

            success, result_data = await self._generate_image_core_internal(
                event=event,
                prompt=full_prompt,
                reference_images=reference_images,
                avatar_reference=avatar_reference,
            )

            if not success or not isinstance(result_data, tuple):
                error_msg = (
                    f"{result_data}\n🧐 可能原因：参考图不可用、网络波动或模型返回空结果。\n✅ 建议：确认图片可访问、简化提示词后再试。"
                    if isinstance(result_data, str)
                    else "❌ 表情包生成未成功。\n🧐 可能原因：模型未返回有效结果或参考图处理失败。\n✅ 建议：重新上传参考图或稍后再试。"
                )
                yield event.plain_result(error_msg)
                return

            image_urls, image_paths, text_content, thought_signature = result_data
            primary_image_path = next(
                (p for p in image_paths if p and Path(p).exists()), None
            )
            if not primary_image_path and image_urls:
                primary_image_path = image_urls[0]

            if not primary_image_path:
                yield event.plain_result(
                    "❌ 未获取到可用的表情源图。\n"
                    "🧐 可能原因：模型未返回图像或图像保存失败。\n"
                    "✅ 建议：检查日志后重试，或更换模型/提示词。"
                )
                return

            # 1. 切割图片
            yield event.plain_result("✂️ 正在切割图片...")
            try:
                # 优先尝试视觉识别裁剪，失败则回退网格裁剪
                split_files: list[str] = []
                if self.enable_llm_crop:
                    split_files = await self._llm_detect_and_split(primary_image_path)
                if not split_files:
                    split_files = await asyncio.to_thread(
                        split_image, primary_image_path, rows=6, cols=4
                    )
            except Exception as e:
                logger.error(f"切割图片时发生异常: {e}")
                split_files = []

            if not split_files:
                yield event.plain_result(
                    "❌ 图片切割失败，无法生成表情包切片。\n"
                    "🧐 可能原因：源图尺寸异常、裁剪依赖缺失或磁盘空间不足。\n"
                    "✅ 建议：尝试降低分辨率重新生成，检查本地裁剪依赖与磁盘空间后再试。"
                )
                yield event.image_result(primary_image_path)
                return

            # 2. 准备发送逻辑

            # 如果开启了ZIP，优先尝试发送ZIP
            if self.enable_sticker_zip:
                zip_path = await asyncio.to_thread(create_zip, split_files)
                if zip_path:
                    try:
                        from astrbot.api.message_components import File

                        file_comp = File(
                            file=zip_path, name=os.path.basename(zip_path)
                        )
                        yield event.chain_result([file_comp])
                        sent_success = True

                        yield event.image_result(primary_image_path)
                    except Exception as e:
                        logger.warning(f"发送ZIP失败: {e}")
                        yield event.plain_result(
                            "⚠️ 压缩包发送失败，降级使用合并转发"
                        )
                        sent_success = False
                else:
                    yield event.plain_result(
                        "❌ 压缩包创建失败，已尝试改用合并转发。\n"
                        "🧐 可能原因：临时目录无写权限或磁盘空间不足。\n"
                        "✅ 建议：清理磁盘或调整临时目录权限后重试，如仍失败可关闭 ZIP 发送。"
                    )
                    sent_success = False

            # 3. 如果没开启ZIP或者ZIP发送失败，发送合并转发
            if not sent_success:
                from astrbot.api.message_components import Image as AstrImage
                from astrbot.api.message_components import Node, Plain

                # 构造节点内容：原图 + 所有小图
                node_content = []
                # 原图预览
                node_content.append(Plain("原图预览："))
                try:
                    node_content.append(AstrImage.fromFileSystem(primary_image_path))
                except Exception:
                    pass
                node_content.append(Plain("表情包切片："))

                for file_path in split_files:
                    try:
                        node_content.append(AstrImage.fromFileSystem(file_path))
                    except Exception:
                        node_content.append(Plain(f"[切片发送失败]: {file_path}"))

                # 构造单个节点，包含所有图片
                node = Node(
                    uin=event.message_obj.self_id,
                    name="Gemini表情包生成",
                    content=node_content,
                )

                yield event.chain_result([node])

        finally:
            self.resolution = old_resolution
            self.aspect_ratio = old_aspect_ratio
            try:
                await self.avatar_manager.cleanup_used_avatars()
            except Exception:
                pass

    @filter.command("生图帮助")
    async def show_help(self, event: AstrMessageEvent):
        """显示插件使用帮助"""
        group_id = self._get_group_id_from_event(event)
        if group_id and self.group_limit_list:
            if (
                self.group_limit_mode == "blacklist"
                and group_id in self.group_limit_list
            ):
                return
            if (
                self.group_limit_mode == "whitelist"
                and group_id not in self.group_limit_list
            ):
                return

        grounding_status = "✓ 启用" if self.enable_grounding else "✗ 禁用"
        smart_retry_status = "✓ 启用" if self.enable_smart_retry else "✗ 禁用"
        avatar_status = "✓ 启用" if self.auto_avatar_reference else "✗ 禁用"

        limit_settings = self.config.get("limit_settings", {})
        enable_rate_limit = limit_settings.get("enable_rate_limit", False)
        rate_limit_period = limit_settings.get("rate_limit_period", 60)
        max_requests = limit_settings.get("max_requests_per_group", 5)
        rate_limit_status = (
            f"✓ {max_requests}次/{rate_limit_period}秒"
            if enable_rate_limit
            else "✗ 禁用"
        )

        tool_timeout = self.get_tool_timeout(event)
        timeout_warning = ""
        if tool_timeout < 90:
            timeout_warning = (
                f"⚠️ LLM工具超时时间较短({tool_timeout}秒)，建议设置为90-120秒"
            )

        try:
            metadata_path = os.path.join(os.path.dirname(__file__), "metadata.yaml")
            with open(metadata_path, encoding="utf-8") as f:
                metadata = yaml.safe_load(f)
                version = metadata.get("version", "v1.3.0")
        except Exception:
            version = "v1.3.0"

        try:
            # 获取主题配置
            service_settings = self.config.get("service_settings", {})
            theme_settings = service_settings.get("theme_settings", {})

            # 解析配置
            mode = theme_settings.get("mode", "cycle")
            cycle_config = theme_settings.get("cycle_config", {})
            single_config = theme_settings.get("single_config", {})

            # 确定要使用的模板文件名
            template_filename = "help_template_light"  # 默认值

            if mode == "single":
                # 单独模式
                template_filename = single_config.get(
                    "template_name", "help_template_light"
                )
            else:
                # 循环模式 (默认)
                day_start = cycle_config.get("day_start", 6)
                day_end = cycle_config.get("day_end", 18)
                day_template = cycle_config.get("day_template", "help_template_light")
                night_template = cycle_config.get(
                    "night_template", "help_template_dark"
                )

                current_hour = datetime.now().hour
                if day_start <= current_hour < day_end:
                    template_filename = day_template
                else:
                    template_filename = night_template

            # 自动补全 .html 后缀
            if not template_filename.endswith(".html"):
                template_filename += ".html"

            # 构建模板路径
            template_path = os.path.join(
                os.path.dirname(__file__), "templates", template_filename
            )

            # 检查文件是否存在，不存在则回退
            if not os.path.exists(template_path):
                logger.warning(f"模板文件不存在: {template_path}，将回退到默认模板")
                template_filename = "help_template_light.html"
                template_path = os.path.join(
                    os.path.dirname(__file__), "templates", template_filename
                )

                # 如果默认模板也不存在（极端情况），抛出异常让外层处理
                if not os.path.exists(template_path):
                    raise FileNotFoundError(f"找不到模板文件: {template_path}")

            # 准备模板数据
            template_data = {
                "title": f"Gemini 图像生成插件 {version}",
                # 以下字段是为了兼容可能使用了旧变量的模板，虽然新设计应该由css控制
                "model": self.model,
                "api_type": self.api_type,
                "resolution": self.resolution,
                "aspect_ratio": self.aspect_ratio or "默认",
                "api_keys_count": len(self.api_keys),
                "grounding_status": grounding_status,
                "avatar_status": avatar_status,
                "smart_retry_status": smart_retry_status,
                "tool_timeout": tool_timeout,
                "rate_limit_status": rate_limit_status,
                "timeout_warning": timeout_warning if timeout_warning else "",
                "enable_sticker_split": self.enable_sticker_split,
            }

            # 读取模板文件
            with open(template_path, encoding="utf-8") as f:
                jinja2_template = f.read()

            # 使用AstrBot的html_render方法
            render_opts = {}
            if self.html_render_options.get("quality") is not None:
                render_opts["quality"] = self.html_render_options["quality"]

            try:
                html_image_url = await self.html_render(
                    jinja2_template,
                    template_data,
                    options=render_opts or None,
                )
            except TypeError:
                # 兼容旧版不支持 options 的接口
                html_image_url = await self.html_render(jinja2_template, template_data)
            logger.info(f"HTML帮助图片生成成功 (使用模板: {template_filename})")
            yield event.image_result(html_image_url)

        except Exception as e:
            logger.error(f"HTML帮助图片生成失败: {e}")
            fallback_help = f"""🎨 Gemini 图像生成插件 {version}

基础指令:
• /生图 [描述] - 生成图像
• /快速 [预设] [描述] - 快速模式
• /改图 [描述] - 修改图像
• /换风格 [风格] - 风格转换
• /生图帮助 - 显示帮助

预设选项: 头像/海报/壁纸/卡片/手机/手办化

当前配置:
• 模型: {self.model}
• 分辨率: {self.resolution}
• API密钥: {len(self.api_keys)}个
• LLM工具超时: {tool_timeout}秒

系统状态:
• 搜索接地: {grounding_status}
• 自动头像: {avatar_status}
• 智能重试: {smart_retry_status}

⚠️ HTML渲染失败，使用文本模式显示

错误信息: {str(e)}"""
            yield event.plain_result(fallback_help)

    @filter.command("改图")
    async def modify_image(self, event: AstrMessageEvent, prompt: str):
        """
        根据提示词修改或重做图像（默认命令）

        Args:
            prompt: 修改描述，如"把头发改成红色"、"换个背景"、"画成动漫风格"等
        """
        allowed, limit_message = await self._check_and_consume_limit(event)
        if not allowed:
            if limit_message:
                yield event.plain_result(limit_message)
            return

        # 构造改图专用提示词，确保修改意图明确
        modification_prompt = get_modification_prompt(prompt)

        yield event.plain_result("🎨 开始修改图像...")

        # 根据配置决定是否使用头像参考
        use_avatar = await self.should_use_avatar(event)

        async for result in self._quick_generate_image(
            event, modification_prompt, use_avatar
        ):
            yield result

    @filter.command("换风格")
    async def change_style(self, event: AstrMessageEvent, style: str, prompt: str = ""):
        """
        改变图像风格

        Args:
            style: 风格描述，如"动漫"、"写实"、"水彩"、"油画"等
            prompt: 额外的修改要求（可选）
        """
        allowed, limit_message = await self._check_and_consume_limit(event)
        if not allowed:
            if limit_message:
                yield event.plain_result(limit_message)
            return

        full_prompt = get_style_change_prompt(style, prompt)

        use_avatar = await self.should_use_avatar(event)
        reference_images, avatar_reference = await self._fetch_images_from_event(
            event, include_at_avatars=use_avatar
        )

        yield event.plain_result("🎨 开始转换风格...")

        success, result_data = await self._generate_image_core_internal(
            event=event,
            prompt=full_prompt,
            reference_images=reference_images,
            avatar_reference=avatar_reference,
            is_modification=True,
        )

        if success and result_data:
            image_urls, image_paths, text_content, thought_signature = result_data
            async for send_res in self._dispatch_send_results(
                event=event,
                image_urls=image_urls,
                image_paths=image_paths,
                text_content=text_content,
                thought_signature=thought_signature,
                scene="换风格",
            ):
                yield send_res
        else:
            yield event.plain_result(result_data)

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
        使用 Gemini 模型生成或修改图像

        当用户请求图像生成、绘画、改图、换风格或手办化时调用此函数。

        判断逻辑：
        - 用户说"改成"、"变成"、"基于"、"修改"、"改图"等词时，设置 use_reference_images="true"
        - 用户说"根据我"、"我的头像"或@某人时，设置 use_reference_images="true" 和 include_user_avatar="true"
        - 用户消息中包含图片且明确要求"修改这张图"时，设置 use_reference_images="true"

        Args:
            prompt(string): 图像生成或修改的详细描述
            use_reference_images(string): 是否使用上下文中的参考图片，true或false。当用户意图是修改、变换或基于现有图片时设置为true
            include_user_avatar(string): 是否包含用户头像作为参考图像，true或false。当用户说"根据我"、"我的头像"或@某人时设置为true
        """
        allowed, limit_message = await self._check_and_consume_limit(event)
        if not allowed:
            if limit_message:
                yield event.plain_result(limit_message)
            return

        if not self.api_client:
            yield event.plain_result(
                "❌ 无法生成图像：API 客户端尚未初始化。\n"
                "🧐 可能原因：API 密钥未配置或加载失败。\n"
                "✅ 建议：在插件配置中填写有效密钥并重启服务。"
            )
            return

        reference_images = []
        avatar_reference = []

        avatar_value = str(include_user_avatar).lower()
        logger.info(f"[AVATAR_DEBUG] include_user_avatar参数: {avatar_value}")
        include_avatar = avatar_value in {"true", "1", "yes", "y", "是"}
        include_reference_images = str(use_reference_images).lower() in {
            "true",
            "1",
            "yes",
            "y",
            "是",
        }

        reference_images, avatar_reference = await self._fetch_images_from_event(
            event, include_at_avatars=include_avatar
        )

        if not include_reference_images:
            reference_images = []
        if not include_avatar:
            avatar_reference = []

        logger.info(
            f"[AVATAR_DEBUG] 收集到参考图: 消息 {len(reference_images)} 张，头像 {len(avatar_reference)} 张"
        )

        success, result_data = await self._generate_image_core_internal(
            event=event,
            prompt=prompt,
            reference_images=reference_images,
            avatar_reference=avatar_reference,
        )

        try:
            await self.avatar_manager.cleanup_cache()
        except Exception as e:
            logger.warning(f"清理头像缓存失败: {e}")

        if success and result_data:
            image_urls, image_paths, text_content, thought_signature = result_data
            async for send_res in self._dispatch_send_results(
                event=event,
                image_urls=image_urls,
                image_paths=image_paths,
                text_content=text_content,
                thought_signature=thought_signature,
                scene="LLM工具",
            ):
                yield send_res
        else:
            yield event.plain_result(result_data)
