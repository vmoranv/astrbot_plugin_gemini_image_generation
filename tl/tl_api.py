"""
API客户端模块
提供Google Gemini和OpenAI兼容API的客户端实现
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import hashlib
import json
import os
import re
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import aiohttp
from PIL import Image as PILImage

from astrbot.api import logger

try:
    from .tl_utils import (
        IMAGE_CACHE_DIR,
        SUPPORTED_IMAGE_MIME_TYPES,
        encode_file_to_base64,
        get_plugin_data_dir,
        save_base64_image,
        save_image_data,
        save_image_stream,
        coerce_supported_image,
        coerce_supported_image_bytes,
        normalize_image_input,
    )
except ImportError:
    from pathlib import Path

    async def save_base64_image(
        base64_data: str, image_format: str = "png"
    ) -> str | None:
        """占位符函数"""
        return None

    async def save_image_data(
        image_data: bytes, image_format: str = "png"
    ) -> str | None:
        """占位符函数"""
        return None

    async def save_image_stream(stream_reader, image_format: str = "png", target_path=None):
        return None

    def encode_file_to_base64(file_path, chunk_size: int = 65536) -> str:
        return ""

    def get_plugin_data_dir() -> Path:
        return Path(".")

    IMAGE_CACHE_DIR = get_plugin_data_dir() / "images" / "download_cache"
    SUPPORTED_IMAGE_MIME_TYPES = {
        "image/png",
        "image/jpeg",
        "image/webp",
        "image/heic",
        "image/heif",
    }

    def coerce_supported_image_bytes(mime_type, raw_bytes):
        return None, None

    def coerce_supported_image(mime_type, base64_data):
        return None, None

    async def normalize_image_input(image_input: Any, *, image_cache_dir=None, image_input_mode="auto"):
        return None, None


@dataclass
class ApiRequestConfig:
    """API 请求配置（基于 Gemini 官方文档）"""

    model: str
    prompt: str
    api_type: str = "openai"
    api_base: str | None = None
    api_key: str | None = None
    resolution: str | None = None
    aspect_ratio: str | None = None
    enable_grounding: bool = False
    response_modalities: str = "TEXT_IMAGE"  # 默认同时返回文本和图像
    max_tokens: int = 1000
    reference_images: list[str] | None = None
    response_text: str | None = None  # 存储文本响应
    enable_smart_retry: bool = True  # 智能重试开关
    enable_text_response: bool = False  # 文本响应开关
    force_resolution: bool = False  # 强制传递分辨率参数
    verbose_logging: bool = False  # 详细日志开关
    image_input_mode: str = "auto"  # auto/force_base64/prefer_url

    # 官方文档推荐参数
    temperature: float = 0.7  # 控制生成随机性，0.0-1.0
    seed: int | None = None  # 固定种子以确保一致性
    safety_settings: dict | None = None  # 安全设置


class APIError(Exception):
    """API 错误基类"""

    def __init__(self, message: str, status_code: int = None, error_type: str = None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.error_type = error_type


class GeminiAPIClient:
    """遵循官方 API 规范的 Gemini API 客户端

    特性：
    - 支持 Google 官方 API 和 OpenAI API
    - 支持自定义 API Base URL（反代）
    - 支持任意模型名称
    - 遵循官方 Gemini API 规范
    """

    # Google 官方 API 默认地址
    GOOGLE_API_BASE = "https://generativelanguage.googleapis.com/v1beta"

    # OpenAI API 默认地址
    OPENAI_API_BASE = "https://api.openai.com/v1"

    def __init__(self, api_keys: list[str]):
        """
        初始化 API 客户端

        Args:
            api_keys: API 密钥列表
        """
        self.api_keys = api_keys or []
        self.current_key_index = 0
        self._lock = asyncio.Lock()
        self.proxy = (
            os.environ.get("HTTPS_PROXY")
            or os.environ.get("https_proxy")
            or os.environ.get("HTTP_PROXY")
            or os.environ.get("http_proxy")
        )
        if self.proxy:
            logger.debug(f"检测到代理配置，使用代理: {self.proxy}")
        logger.debug(f"API 客户端已初始化，支持 {len(self.api_keys)} 个 API 密钥")
        self.verbose_logging: bool = False

    @staticmethod
    def _coerce_supported_image_bytes(
        mime_type: str | None, raw_bytes: bytes
    ) -> tuple[str | None, str | None]:
        return coerce_supported_image_bytes(mime_type, raw_bytes)

    @staticmethod
    def _coerce_supported_image(mime_type: str | None, base64_data: str) -> tuple[str | None, str | None]:
        return coerce_supported_image(mime_type, base64_data)

    async def get_next_api_key(self) -> str:
        """获取下一个 API 密钥"""
        async with self._lock:
            if not self.api_keys:
                raise ValueError("API 密钥列表不能为空")
            key = self.api_keys[self.current_key_index % len(self.api_keys)]
            return key

    async def rotate_api_key(self):
        """轮换到下一个 API 密钥"""
        async with self._lock:
            if len(self.api_keys) > 1:
                self.current_key_index = (self.current_key_index + 1) % len(
                    self.api_keys
                )
                logger.debug(
                    f"已轮换到下一个 API 密钥，当前索引: {self.current_key_index}"
                )

    @staticmethod
    async def _prepare_google_payload(config: ApiRequestConfig) -> dict[str, Any]:
        """准备 Google 官方 API 请求负载（遵循官方规范）"""
        parts = [{"text": config.prompt}]

        force_b64 = str(getattr(config, "image_input_mode", "auto")).lower() == "force_base64"
        enable_file_uri = (
            str(getattr(config, "image_input_mode", "auto")).lower() == "prefer_url"
        )
        allowed_file_uri_hosts = {"storage.googleapis.com", "lh3.googleusercontent.com", "gstatic.com"}

        if config.reference_images:
            for image_input in config.reference_images[:14]:
                image_str = str(image_input).strip()
                parsed = urllib.parse.urlparse(image_str)

                # 非强制 base64 且为可识别的 http(s) 链接时，优先尝试 fileUri（仅允许部分可信域名）
                if (
                    enable_file_uri
                    and not force_b64
                    and parsed.scheme in ("http", "https")
                    and parsed.netloc
                ):
                    netloc = parsed.netloc.lower()
                    if any(netloc == host or netloc.endswith("." + host) for host in allowed_file_uri_hosts):
                        parts.append({"fileData": {"fileUri": image_str}})
                        logger.debug(
                            "Gemini 官方接口使用 URL 参考图: %s",
                            image_str[:120],
                        )
                        continue
                    else:
                        logger.debug(
                            "URL 域名不在允许列表（或未启用 prefer_url），改用 base64 传输: %s",
                            image_str[:120],
                        )

                # 其他情况：转换为支持的 base64
                mime_type, data = await GeminiAPIClient._normalize_image_input(image_input)
                if not data:
                    if force_b64:
                        raise APIError(
                            f"参考图转 base64 失败（force_base64），输入类型: {type(image_input)}",
                            None,
                            "invalid_reference_image",
                        )
                    logger.warning(f"跳过无法识别/读取的参考图像: {type(image_input)}")
                    continue

                # 严格校验 base64，避免传入无效数据导致 inline_data 解码错误
                cleaned_data = data.strip().replace("\n", "")
                if cleaned_data != data:
                    data = cleaned_data
                try:
                    base64.b64decode(data, validate=True)
                except Exception:
                    if force_b64:
                        raise APIError(
                            "参考图 base64 校验失败（force_base64），请检查图片链接是否可访问或改用 URL 模式",
                            None,
                            "invalid_reference_image",
                        )
                    logger.warning(
                        "跳过无效的 base64 参考图像: %s...",
                        str(image_input)[:80],
                    )
                    continue

                parts.append({"inlineData": {"mimeType": mime_type, "data": data}})

        contents = [{"role": "user", "parts": parts}]

        generation_config = {"responseModalities": ["TEXT", "IMAGE"]}

        # 根据官方文档，图像生成必须同时包含 TEXT 和 IMAGE modalities
        # 这样可以确保返回图像而不是纯文本
        modalities_map = {
            "TEXT": ["TEXT"],
            "IMAGE": ["IMAGE"],
            "TEXT_IMAGE": ["TEXT", "IMAGE"],
        }

        # 获取配置的模态
        modalities = modalities_map.get(config.response_modalities, ["TEXT", "IMAGE"])

        # 确保包含图像模态
        if "IMAGE" not in modalities:
            logger.warning("配置中缺少 IMAGE modality，自动添加以支持图像生成")
            modalities.append("IMAGE")

        # 确保包含文本模态
        if "TEXT" not in modalities:
            logger.debug("添加 TEXT modality 以提供更好的兼容性")
            modalities.append("TEXT")

        generation_config["responseModalities"] = modalities
        logger.debug(f"响应模态: {modalities}")

        image_config = {}

        # 根据官方文档设置图像尺寸
        if config.resolution:
            resolution = config.resolution.upper()

            if resolution in ["1K", "1024x1024"]:
                image_config["image_size"] = "1K"
                logger.debug("设置图像尺寸: 1K")
            elif resolution in ["2K", "2048x2048"]:
                image_config["image_size"] = "2K"
                logger.debug("设置图像尺寸: 2K")
            elif resolution in ["4K", "4096x4096"]:
                image_config["image_size"] = "4K"
                logger.debug("设置图像尺寸: 4K")
            else:
                # 默认使用1K
                image_config["image_size"] = "1K"
                logger.warning(f"不支持的分辨率: {config.resolution}，使用默认尺寸 1K")

        # 设置长宽比
        if config.aspect_ratio and ":" in config.aspect_ratio:
            # 将长宽比转换为标准格式
            ratio_map = {
                "1:1": "1:1",
                "16:9": "16:9",
                "9:16": "9:16",
                "3:2": "3:2",
                "4:3": "4:3",
            }
            ratio = ratio_map.get(config.aspect_ratio, config.aspect_ratio)
            image_config["aspect_ratio"] = ratio
            logger.debug(f"设置长宽比: {ratio}")
        elif config.aspect_ratio:
            logger.warning(
                f"不支持的长宽比格式: {config.aspect_ratio}，将使用默认长宽比"
            )

        if image_config:
            generation_config["image_config"] = image_config

        # 添加官方文档推荐参数
        if config.temperature is not None:
            generation_config["temperature"] = config.temperature
        if config.seed is not None:
            generation_config["seed"] = config.seed
        if config.safety_settings:
            generation_config["safetySettings"] = config.safety_settings

        tools = []
        if config.enable_grounding:
            tools.append({"google_search": {}})

        payload = {"contents": contents, "generationConfig": generation_config}

        if tools:
            payload["tools"] = tools

        # 调试：记录 image_config
        if "image_config" in generation_config:
            logger.debug(
                f"实际发送的 image_config: {generation_config['image_config']}"
            )

        return payload

    @staticmethod
    async def _prepare_openai_payload(config: ApiRequestConfig) -> dict[str, Any]:
        """准备 OpenAI API 请求负载"""
        message_content = [
            {"type": "text", "text": f"Generate an image: {config.prompt}"}
        ]

        force_b64 = str(getattr(config, "image_input_mode", "auto")).lower() == "force_base64"
        def _ensure_valid_base64(data: str, context: str):
            try:
                cleaned = data.strip().replace("\n", "")
                base64.b64decode(cleaned, validate=True)
            except Exception:
                raise APIError(
                    f"参考图 base64 校验失败（force_base64），来源: {context}",
                    None,
                    "invalid_reference_image",
                )

        if config.reference_images:
            # 本地缓存避免重复处理同一引用图，记录耗时便于性能观察
            processed_cache: dict[str, dict[str, Any]] = {}
            supported_exts = {
                "jpg",
                "jpeg",
                "png",
                "webp",
                "gif",
                "bmp",
                "tif",
                "tiff",
                "heic",
                "avif",
            }
            total_start = time.perf_counter()

            for idx, image_input in enumerate(config.reference_images[:6]):
                per_start = time.perf_counter()
                image_str = str(image_input).strip()
                if not image_str:
                    logger.warning(f"跳过空白参考图像: idx={idx}")
                    continue

                if "&amp;" in image_str:
                    image_str = image_str.replace("&amp;", "&")

                # 命中缓存直接复用，避免重复 base64 处理
                if image_str in processed_cache:
                    logger.debug(f"参考图像命中缓存: idx={idx}")
                    message_content.append(processed_cache[image_str])
                    continue

                parsed = urllib.parse.urlparse(image_str)
                image_payload: dict[str, Any] | None = None

                try:
                    # 优先处理 http(s) URL，确保 scheme 和 netloc 合法
                    if parsed.scheme in ("http", "https") and parsed.netloc and not force_b64:
                        ext = Path(parsed.path).suffix.lower().lstrip(".")
                        if ext and ext not in supported_exts:
                            logger.debug(
                                "参考图像URL扩展名不在常见列表: idx=%s ext=%s url=%s",
                                idx,
                                ext,
                                image_str[:80],
                            )

                        image_payload = {
                            "type": "image_url",
                            "image_url": {"url": image_str},
                        }
                        logger.debug(
                            "OpenAI兼容API使用URL参考图: idx=%s ext=%s url=%s",
                            idx,
                            ext or "unknown",
                            image_str[:120],
                        )

                    # data URL：直接校验 base64，有效则不再重复转码
                    elif image_str.startswith("data:image/") and ";base64," in image_str:
                        header, _, data_part = image_str.partition(";base64,")
                        mime_type = header.replace("data:", "").lower()
                        try:
                            base64.b64decode(data_part, validate=True)
                        except (binascii.Error, ValueError) as e:
                            logger.warning(
                                "跳过无效的 data URL 参考图: idx=%s 错误=%s", idx, e
                            )
                            mime_type = None

                        if mime_type:
                            ext = mime_type.split("/")[-1]
                            if ext and ext not in supported_exts:
                                logger.debug(
                                    "data URL 图片格式不常见: idx=%s mime=%s", idx, mime_type
                                )
                            image_payload = {
                                "type": "image_url",
                                "image_url": {"url": image_str},
                            }
                            logger.debug(
                                "OpenAI兼容API使用data URL参考图: idx=%s mime=%s",
                                idx,
                                mime_type,
                            )

                    # 其他输入交给规范化逻辑，自动转换为 data URL
                    else:
                        mime_type, data = await GeminiAPIClient._normalize_image_input(
                            image_input
                        )
                        if not data:
                            if force_b64:
                                raise APIError(
                                    f"参考图转 base64 失败（force_base64），idx={idx}, type={type(image_input)}",
                                    None,
                                    "invalid_reference_image",
                                )
                            logger.warning(
                                "跳过无法识别/读取的参考图像: idx=%s type=%s",
                                idx,
                                type(image_input),
                            )
                            continue

                        if not mime_type or not mime_type.startswith("image/"):
                            logger.debug(
                                "未检测到明确的图片 MIME，默认使用 image/png: idx=%s",
                                idx,
                            )
                            mime_type = "image/png"

                        ext = mime_type.split("/")[-1]
                        if ext and ext not in supported_exts:
                            logger.debug(
                                "规范化后图片格式不常见: idx=%s mime=%s", idx, mime_type
                            )

                        if force_b64:
                            _ensure_valid_base64(data, f"idx={idx}")
                            payload_url = data.strip().replace("\n", "")
                        else:
                            payload_url = f"data:{mime_type};base64,{data}"

                        image_payload = {
                            "type": "image_url",
                            "image_url": {"url": payload_url},
                        }

                    if image_payload:
                        message_content.append(image_payload)
                        processed_cache[image_str] = image_payload
                        elapsed_ms = (time.perf_counter() - per_start) * 1000
                        logger.debug(
                            "参考图像处理完成: idx=%s 耗时=%.2fms 来源=%s",
                            idx,
                            elapsed_ms,
                            parsed.scheme or "normalized",
                        )
                except Exception as e:
                    logger.warning("处理参考图像时出现异常: idx=%s err=%s", idx, e)
                    continue

            total_elapsed_ms = (time.perf_counter() - total_start) * 1000
            if processed_cache:
                logger.debug(
                    "参考图像处理统计: 总数=%s 总耗时=%.2fms 平均=%.2fms",
                    len(processed_cache),
                    total_elapsed_ms,
                    total_elapsed_ms / len(processed_cache),
                )

        # OpenAI 兼容接口下：
        # - 使用 chat/completions
        # - modalities: ["image", "text"]
        # - image_config: {aspect_ratio, image_size}
        # - tools: [{google_search:{}}]（当启用搜索接地时）
        payload: dict[str, Any] = {
            "model": config.model,
            "messages": [{"role": "user", "content": message_content}],
            "max_tokens": config.max_tokens,
            "temperature": 0.7,
            "modalities": ["image", "text"],
        }

        # image_config 与 Gemini 3 Pro Image 模型相关的配置
        image_config: dict[str, Any] = {}

        if config.aspect_ratio:
            image_config["aspect_ratio"] = config.aspect_ratio

        # 仅在 Gemini 3 Pro Image 系列模型下传递 image_size
        model_name = (config.model or "").lower()
        is_gemini_image_model = (
            "gemini-3-pro-image" in model_name
            or "gemini-3-pro-preview" in model_name
            or config.force_resolution  # 如果强制开启，则忽略模型名称检查
        )

        if is_gemini_image_model and config.resolution:
            # 前端 router 侧直接传递 "1K"/"2K"/"4K"，这里保持一致
            image_config["image_size"] = config.resolution

        if image_config:
            payload["image_config"] = image_config

        # 与前端 router 一致：启用搜索接地时，通过 tools.google_search 控制
        if is_gemini_image_model and config.enable_grounding:
            payload["tools"] = [{"google_search": {}}]

        return payload

    @staticmethod
    async def _normalize_image_input(image_input: Any) -> tuple[str | None, str | None]:
        """统一调用 tl_utils 的参考图规范化逻辑"""
        return await normalize_image_input(
            image_input,
            image_cache_dir=IMAGE_CACHE_DIR,
            image_input_mode="auto",
        )


    async def _get_api_url(
        self, config: ApiRequestConfig
    ) -> tuple[str, dict[str, str], dict[str, Any]]:
        """
        根据配置获取 API URL、请求头和负载

        智能处理API路径前缀，无需手动输入/v1或/v1beta
        """
        # 确定 API 基础地址（支持反代）
        if config.api_base:
            api_base = config.api_base.rstrip("/")
            logger.debug(f"使用自定义 API Base: {api_base}")
        else:
            if config.api_type == "google":
                api_base = self.GOOGLE_API_BASE
            else:  # openai 兼容格式
                api_base = self.OPENAI_API_BASE

            logger.debug(f"使用默认 API Base ({config.api_type}): {api_base}")

        # 智能构建完整URL，自动添加正确的路径前缀（如果需要的话）
        if config.api_type == "google":
            # Google API 需要版本前缀
            if not config.api_base or api_base == self.GOOGLE_API_BASE:
                # 使用默认官方地址，直接使用完整路径
                url = f"{api_base}/models/{config.model}:generateContent"
            elif not any(api_base.endswith(suffix) for suffix in ["/v1beta", "/v1"]):
                # 自定义地址但没有版本前缀，自动添加
                url = f"{api_base}/v1beta/models/{config.model}:generateContent"
                logger.debug("为Google API自动添加v1beta前缀")
            else:
                # 已经包含版本前缀，直接使用
                url = f"{api_base}/models/{config.model}:generateContent"
                logger.debug("使用已包含版本前缀的Google API地址")

            payload = await self._prepare_google_payload(config)
            headers = {
                "x-goog-api-key": config.api_key,
                "Content-Type": "application/json",
            }
        else:
            # OpenAI 兼容格式
            if not config.api_base or api_base == self.OPENAI_API_BASE:
                # 使用默认地址，需要完整路径
                url = f"{api_base}/chat/completions"
            elif not any(api_base.endswith(suffix) for suffix in ["/v1", "/v1beta"]):
                # 自定义地址但没有版本前缀，自动添加
                url = f"{api_base}/v1/chat/completions"
                logger.debug("为OpenAI兼容API自动添加v1前缀")
            else:
                # 已经包含版本前缀，直接使用
                url = f"{api_base}/chat/completions"
                logger.debug("使用已包含版本前缀的OpenAI兼容API地址")

            payload = await self._prepare_openai_payload(config)
            headers = {
                "Authorization": f"Bearer {config.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/astrbot",
                "X-Title": "AstrBot Gemini Image Advanced",
            }

        logger.debug(f"智能构建API URL: {url}")
        return url, headers, payload

    async def generate_image(
        self,
        config: ApiRequestConfig,
        max_retries: int = 3,
        total_timeout: int = 120,
        per_retry_timeout: int = None,
        max_total_time: int = None,
    ) -> tuple[list[str], list[str], str | None, str | None]:
        """
        生成图像

        Args:
            config: 请求配置
            max_retries: 最大重试次数
            total_timeout: 总超时时间（秒）

        Returns:
            (image_urls, image_paths, text_content, thought_signature)，如果失败则返回空列表和None
        """
        if not self.api_keys:
            raise ValueError("未配置 API 密钥")

        if not config.api_key:
            config.api_key = await self.get_next_api_key()

        # 获取请求信息
        url, headers, payload = await self._get_api_url(config)

        logger.debug(f"使用 {config.model} (通过 {config.api_type}) 生成图像")
        logger.debug(f"API 端点: {url[:80]}...")

        if config.resolution or config.aspect_ratio:
            logger.debug(
                f"分辨率: {config.resolution or '默认'}, 长宽比: {config.aspect_ratio or '默认'}"
            )

        if config.api_base:
            logger.debug(f"使用自定义 API Base: {config.api_base}")

        # 同步详细日志开关，便于在内部网络请求中控制输出粒度
        self.verbose_logging = bool(getattr(config, "verbose_logging", False))

        return await self._make_request(
            url=url,
            payload=payload,
            headers=headers,
            api_type=config.api_type,
            model=config.model,
            max_retries=max_retries,
            total_timeout=total_timeout,
        )

    async def _make_request(
        self,
        url: str,
        payload: dict[str, Any],
        headers: dict[str, str],
        api_type: str,
        model: str,
        max_retries: int,
        total_timeout: int = 120,
    ) -> tuple[list[str], list[str], str | None, str | None]:
        """执行 API 请求并处理响应，每个重试有独立的超时控制"""

        current_retry = 0
        last_error = None

        while current_retry < max_retries:
            try:
                # 每个重试使用独立的超时控制，不共享总超时时间
                timeout = aiohttp.ClientTimeout(
                    total=total_timeout, sock_read=total_timeout
                )
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    logger.debug(f"发送请求（重试 {current_retry}/{max_retries - 1}）")
                    return await self._perform_request(
                        session, url, payload, headers, api_type, model
                    )

            except asyncio.CancelledError:
                # 只有框架取消才不重试（这是最顶层的超时）
                logger.debug("请求被框架取消（工具调用总超时），不再重试")
                timeout_msg = "图像生成时间过长，超出了框架限制。请尝试简化图像描述或在框架配置中增加 tool_call_timeout 到 90-120 秒。"
                raise APIError(timeout_msg, None, "cancelled") from None
            except Exception as e:
                error_msg = str(e)
                error_type = self._classify_error(e, error_msg)

                # 判断是否可重试的错误
                if self._is_retryable_error(error_type, e):
                    last_error = APIError(error_msg, None, error_type)
                    logger.warning(
                        f"可重试错误 (重试 {current_retry + 1}/{max_retries}): {error_msg}"
                    )

                    current_retry += 1
                    if current_retry < max_retries:
                        # 指数退避延迟：2秒、4秒、8秒……最大10秒
                        delay = min(2 ** (current_retry + 1), 10)
                        logger.debug(f"等待 {delay} 秒后重试...")
                        await asyncio.sleep(delay)
                        continue  # 继续下一次重试
                    else:
                        logger.error(f"达到最大重试次数 ({max_retries})，生成失败")
                else:
                    # 不可重试的错误，立即抛出
                    logger.error(f"不可重试错误: {error_msg}")
                    raise APIError(error_msg, None, error_type) from None

        # 如果都失败了，返回最后一次错误
        if last_error:
            raise last_error

        return [], [], None, None

    def _classify_error(self, exception: Exception, error_msg: str) -> str:
        """分类错误类型"""
        if isinstance(exception, asyncio.TimeoutError):
            return "timeout"
        elif "timeout" in error_msg.lower():
            return "timeout"
        elif "connection" in error_msg.lower():
            return "network"
        elif isinstance(exception, aiohttp.ClientError):
            return "network"
        else:
            return "unknown"

    def _is_retryable_error(self, error_type: str, exception: Exception) -> bool:
        """判断错误是否可重试"""
        # 特殊处理：未生成图像的重试
        if error_type == "no_image_retry":
            return True

        # 可重试的错误：超时、网络错误、服务器错误
        if error_type in ["timeout", "network"]:
            return True

        # HTTP 状态码判断
        if hasattr(exception, "status"):
            status = exception.status
            # 可重试：408, 500, 502, 503, 504
            # 不可重试：401, 402, 403, 422, 429（速率限制）
            if status in [408, 500, 502, 503, 504]:
                return True
            elif status in [401, 402, 403, 422, 429]:
                return False

        return True  # 默认重试未知错误

    async def _perform_request(
        self,
        session: aiohttp.ClientSession,
        url: str,
        payload: dict[str, Any],
        headers: dict[str, str],
        api_type: str,
        model: str,
    ) -> tuple[list[str], list[str], str | None, str | None]:
        """执行实际的HTTP请求"""
        logger.debug(f"发送请求到: {url[:100]}...")

        async with session.post(
            url, json=payload, headers=headers, proxy=self.proxy
        ) as response:
            logger.debug(f"响应状态: {response.status}")
            response_text = await response.text()

            # 解析 JSON 响应，添加错误处理
            try:
                response_data = json.loads(response_text) if response_text else {}
            except json.JSONDecodeError as e:
                logger.error(f"JSON 解析失败: {e}")
                logger.error(f"响应内容前500字符: {response_text[:500]}")
                raise APIError(
                    f"API 返回了无效的 JSON 响应: {e}", response.status
                ) from None

            if response.status == 200:
                logger.debug("API 调用成功")
                if api_type == "google":
                    return await self._parse_gresponse(response_data, session)
                else:  # openai 兼容格式
                    return await self._parse_openai_response(response_data, session)
            elif response.status in [429, 402, 403]:
                error_msg = response_data.get("error", {}).get(
                    "message", f"HTTP {response.status}"
                )
                logger.warning(f"API 配额/权限问题: {error_msg}")
                raise APIError(error_msg, response.status, "quota")
            else:
                error_msg = response_data.get("error", {}).get(
                    "message", f"HTTP {response.status}"
                )
                logger.warning(f"API 错误: {error_msg}")
                raise APIError(error_msg, response.status)

    async def _parse_gresponse(
        self, response_data: dict, session: aiohttp.ClientSession
    ) -> tuple[list[str], list[str], str | None, str | None]:
        """解析 Google 官方 API 响应"""
        import asyncio

        parse_start = asyncio.get_event_loop().time()
        logger.debug("🔍 开始解析API响应数据...")

        if "candidates" not in response_data or not response_data["candidates"]:
            if "promptFeedback" in response_data:
                feedback = response_data["promptFeedback"]
                logger.warning(f"请求被阻止: {feedback}")
            else:
                logger.error(f"响应中没有 candidates: {response_data}")
            return [], [], None, None

        candidates = response_data["candidates"]
        logger.debug(f"📝 找到 {len(candidates)} 个候选结果")

        image_urls: list[str] = []
        image_paths: list[str] = []
        text_chunks: list[str] = []
        thought_signature = None

        for idx, candidate in enumerate(candidates):
            finish_reason = candidate.get("finishReason")
            if finish_reason in ["SAFETY", "RECITATION"]:
                logger.warning(f"候选 {idx} 生成被阻止: {finish_reason}")
                continue

            content = candidate.get("content", {})
            parts = content.get("parts") or []
            logger.debug(f"📋 候选 {idx} 包含 {len(parts)} 个部分")

            for i, part in enumerate(parts):
                try:
                    logger.debug(f"检查候选 {idx} 的第 {i} 个part: {list(part.keys())}")

                    if "thoughtSignature" in part and not thought_signature:
                        thought_signature = part["thoughtSignature"]
                        logger.debug(f"🧠 找到思维签名: {thought_signature[:50]}...")

                    inline_data = part.get("inlineData") or part.get("inline_data")
                    if inline_data and not part.get("thought", False):
                        mime_type = (
                            inline_data.get("mimeType")
                            or inline_data.get("mime_type")
                            or "image/png"
                        )
                        base64_data = inline_data.get("data", "")

                        logger.debug(
                            f"🎯 找到图像数据 (候选{idx} 第{i + 1}部分): {mime_type}, 大小: {len(base64_data)} 字符"
                        )

                        if base64_data:
                            image_format = (
                                mime_type.split("/")[1] if "/" in mime_type else "png"
                            )

                            logger.debug("💾 开始保存图像文件...")
                            save_start = asyncio.get_event_loop().time()

                            saved_path = await save_base64_image(
                                base64_data, image_format
                            )

                            save_end = asyncio.get_event_loop().time()
                            logger.debug(
                                f"✅ 图像保存完成，耗时: {save_end - save_start:.2f}秒"
                            )

                            if saved_path:
                                image_paths.append(saved_path)
                                image_urls.append(saved_path)
                        else:
                            logger.warning(f"候选 {idx} 的第 {i} 个part有inlineData但data为空")
                    elif "thought" in part and part.get("thought", False):
                        logger.debug(f"候选 {idx} 的第 {i} 个part是思考内容")
                    elif "text" in part and not part.get("thought", False):
                        text_chunks.append(part.get("text", ""))
                    else:
                        logger.debug(
                            f"候选 {idx} 的第 {i} 个part不是图像也不是思考: {list(part.keys())}"
                        )
                except Exception as e:
                    logger.error(
                        f"处理候选 {idx} 的第 {i} 个part时出错: {e}", exc_info=True
                    )

        logger.debug(f"🖼️ 共找到 {len(image_paths)} 张图片")

        text_content = (
            " ".join(chunk for chunk in text_chunks if chunk).strip()
            if text_chunks
            else None
        )
        if text_content:
            logger.debug(f"🎯 找到文本内容: {text_content[:100]}...")

        if image_paths:
            parse_end = asyncio.get_event_loop().time()
            logger.debug(f"🎉 API响应解析完成，总耗时: {parse_end - parse_start:.2f}秒")
            return image_urls, image_paths, text_content, thought_signature

        if text_content:
            logger.warning("API只返回了文本响应，未生成图像，将触发重试")
            raise APIError(
                "图像生成失败：API只返回了文本响应，正在重试...",
                500,
                "no_image_retry",
            )

        logger.error("未在响应中找到图像数据")
        raise APIError(
            "图像生成失败：响应格式异常，未找到有效的图像数据", None, "invalid_response"
        )

    async def _parse_openai_response(
        self, response_data: dict, session: aiohttp.ClientSession
    ) -> tuple[list[str], list[str], str | None, str | None]:
        """解析 OpenAI API 响应"""

        image_urls: list[str] = []
        image_paths: list[str] = []
        text_content = None
        thought_signature = None

        if "choices" in response_data:
            choice = response_data["choices"][0]
            message = choice.get("message", {})
            content = message.get("content", "")

            text_chunks: list[str] = []
            image_candidates: list[str] = []
            extracted_urls: list[str] = []

            if isinstance(content, list):
                for part in content:
                    if not isinstance(part, dict):
                        continue

                    part_type = part.get("type")
                    if part_type == "text" and "text" in part:
                        text_val = str(part.get("text", ""))
                        text_chunks.append(text_val)
                        extracted_urls.extend(self._find_image_urls_in_text(text_val))
                    elif part_type == "image_url":
                        image_obj = part.get("image_url") or {}
                        if isinstance(image_obj, dict):
                            url_val = image_obj.get("url")
                            if url_val:
                                image_candidates.append(url_val)
            elif isinstance(content, str):
                text_chunks.append(content)
                extracted_urls.extend(self._find_image_urls_in_text(content))

            # 标准 images 字段（兼容 Gemini/OpenAI 混合格式）
            if "images" in message and message["images"]:
                for image_item in message["images"]:
                    if not isinstance(image_item, dict):
                        continue

                    # 典型格式：{"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
                    image_obj = image_item.get("image_url")
                    if isinstance(image_obj, dict):
                        url_val = image_obj.get("url")
                        if isinstance(url_val, str) and url_val:
                            image_candidates.append(url_val)
                    elif isinstance(image_obj, str) and image_obj:
                        image_candidates.append(image_obj)
                    # 退化格式：{"url": "..."}
                    elif isinstance(image_item.get("url"), str):
                        image_candidates.append(image_item["url"])

            # 合并在文本里解析到的图像 URL
            if extracted_urls:
                image_candidates.extend(extracted_urls)

            # 组装文本内容
            if text_chunks:
                text_content = " ".join([t for t in text_chunks if t]).strip() or None

            # 按顺序处理图像候选
            for candidate_url in image_candidates:
                if isinstance(candidate_url, str) and candidate_url.startswith(
                    "data:image/"
                ):
                    image_url, image_path = await self._parse_data_uri(candidate_url)
                elif isinstance(candidate_url, str):
                    # 对于可访问的 http(s) 链接，直接返回 URL，避免重复下载占用带宽
                    if candidate_url.startswith("http://") or candidate_url.startswith(
                        "https://"
                    ):
                        image_urls.append(candidate_url)
                        logger.debug(
                            f"🖼️ OpenAI 返回可直接访问的图像链接: {candidate_url}"
                        )
                        continue
                    image_url, image_path = await self._download_image(
                        candidate_url, session, use_cache=False
                    )
                else:
                    logger.warning(f"跳过非字符串类型的图像URL: {type(candidate_url)}")
                    continue

                if image_url or image_path:
                    if image_url:
                        image_urls.append(image_url)
                    if image_path:
                        image_paths.append(image_path)

            # content 中查找内联 data URI（文本里）
            extracted_urls: list[str] = []
            extracted_paths: list[str] = []

            if isinstance(content, str):
                extracted_urls, extracted_paths = await self._extract_from_content(
                    content
                )
            elif text_content:
                extracted_urls, extracted_paths = await self._extract_from_content(
                    text_content
                )

            if extracted_urls or extracted_paths:
                image_urls.extend(extracted_urls)
                image_paths.extend(extracted_paths)

        # OpenAI 格式
        elif "data" in response_data and response_data["data"]:
            for image_item in response_data["data"]:
                if "url" in image_item:
                    image_url, image_path = await self._download_image(
                        image_item["url"], session, use_cache=False
                    )
                    if image_url:
                        image_urls.append(image_url)
                    if image_path:
                        image_paths.append(image_path)
                elif "b64_json" in image_item:
                    image_path = await save_base64_image(image_item["b64_json"], "png")
                    if image_path:
                        # 直接使用文件路径，不使用 file:// URI（根据 AstrBot 文档要求）
                        image_urls.append(image_path)
                        image_paths.append(image_path)

        if image_urls or image_paths:
            logger.debug(f"🖼️ OpenAI 收集到 {len(image_paths) or len(image_urls)} 张图片")
            return image_urls, image_paths, text_content, thought_signature

        # 如果只有文本内容，也返回
        if text_content:
            # 如果配置了需要文本响应，且确实没有找到图片，这里应该报错触发重试而不是直接返回文本
            # 除非这是一个纯文本请求（但在生图插件里通常不是）
            logger.warning("OpenAI只返回了文本响应，未生成图像，将触发重试")
            raise APIError(
                "图像生成失败：API只返回了文本响应，正在重试...", 500, "no_image_retry"
            )

        logger.warning("OpenAI 响应格式不支持或未找到图像数据")
        return image_urls, image_paths, text_content, thought_signature

    async def _parse_data_uri(self, data_uri: str) -> tuple[str | None, str | None]:
        """解析 data URI 格式的图像"""
        try:
            if ";base64," not in data_uri:
                logger.error("无效的 data URI 格式")
                return None, None

            header, base64_data = data_uri.split(";base64,", 1)
            mime_type = header.replace("data:", "")
            format_type = mime_type.split("/")[1] if "/" in mime_type else "png"

            image_path = await save_base64_image(base64_data, format_type)
            if image_path:
                # 直接使用文件路径，不使用 file:// URI（根据 AstrBot 文档要求）
                image_url = image_path
                return image_url, image_path
        except Exception as e:
            logger.error(f"解析 data URI 失败: {e}")

        return None, None

    async def _extract_from_content(
        self, content: str
    ) -> tuple[list[str], list[str]]:
        """从文本内容中提取所有 data URI 图像，保持顺序"""
        pattern = r"data:image/([^;]+);base64,([A-Za-z0-9+/=\s]+)"
        matches = re.findall(pattern, content)

        image_urls: list[str] = []
        image_paths: list[str] = []

        for image_format, base64_string in matches:
            image_path = await save_base64_image(base64_string, image_format)
            if image_path:
                # 直接使用文件路径，不使用 file:// URI（根据 AstrBot 文档要求）
                image_url = image_path
                image_urls.append(image_url)
                image_paths.append(image_path)

        return image_urls, image_paths

    def _find_image_urls_in_text(self, text: str) -> list[str]:
        """从文本/Markdown中提取可用的 http(s) 图片链接"""
        if not text:
            return []

        # Markdown 图片语法与裸露的图片链接
        markdown_pattern = r"!\[[^\]]*\]\((https?://[^)]+)\)"
        raw_pattern = r"(https?://[^\s)]+\.(?:png|jpe?g|gif|webp|bmp|tiff|avif))(?:\b|$)"

        urls: list[str] = []
        seen: set[str] = set()
        for pattern in (markdown_pattern, raw_pattern):
            for match in re.findall(pattern, text, flags=re.IGNORECASE):
                cleaned = match.strip().replace("&amp;", "&")
                if cleaned not in seen:
                    seen.add(cleaned)
                    urls.append(cleaned)

        return urls

    async def _download_image(
        self,
        image_url: str,
        session: aiohttp.ClientSession,
        use_cache: bool = False,
    ) -> tuple[str | None, str | None]:
        """下载并保存图像，可选择是否使用缓存（默认关闭以避免返回旧图）"""
        cleaned_url = image_url.replace("&amp;", "&") if isinstance(image_url, str) else image_url
        parsed = urllib.parse.urlparse(cleaned_url)
        is_http = parsed.scheme in {"http", "https"}
        cache_key = (
            hashlib.sha256(cleaned_url.encode("utf-8")).hexdigest()
            if (use_cache and isinstance(cleaned_url, str))
            else None
        )

        # 针对 CQ 码图服务器增加专用请求头
        headers: dict[str, str] = {}
        if is_http:
            headers.update(
                {
                    "User-Agent": (
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0 Safari/537.36"
                    ),
                    "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
                    "Accept-Language": "zh-CN,zh;q=0.9",
                    "Connection": "keep-alive",
                }
            )
            if "gchat.qpic.cn" in (parsed.netloc or ""):
                headers["Referer"] = "https://qun.qq.com"
            elif parsed.scheme and parsed.netloc:
                headers["Referer"] = f"{parsed.scheme}://{parsed.netloc}"

        # 缓存命中直接返回，减少重复下载与内存占用
        if cache_key:
            try:
                IMAGE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
                cached = next(IMAGE_CACHE_DIR.glob(f"{cache_key}.*"), None)
                if cached and cached.exists() and cached.stat().st_size > 0:
                    logger.debug(f"图像下载命中缓存: {cleaned_url}")
                    return str(cached), str(cached)
            except Exception as e:
                logger.debug(f"检查图像缓存失败: {e}")

        max_retries = 1
        retry_interval = 1.0

        for attempt in range(1, max_retries + 1):
            try:
                logger.debug(
                    f"正在下载图像: {cleaned_url[:100]}... 尝试 {attempt}/{max_retries}"
                )

                async with session.get(
                    cleaned_url,
                    timeout=aiohttp.ClientTimeout(total=30),
                    proxy=self.proxy,
                    headers=headers or None,
                ) as response:
                    if response.status != 200:
                        try:
                            err_text = await response.text()
                        except Exception:
                            err_text = ""

                        response_reason = response.reason or ""
                        response_content_type = response.headers.get(
                            "Content-Type", ""
                        )
                        query_params = urllib.parse.parse_qs(parsed.query)
                        param_issues: list[str] = []

                        # 仅在出现 400 错误时进行参数合法性检查
                        if response.status == 400:
                            appid = (query_params.get("appid") or [None])[0]
                            if appid and not re.fullmatch(r"[A-Za-z0-9]+", appid):
                                param_issues.append("appid 格式异常（仅允许字母数字）")

                            fileid = (query_params.get("fileid") or [None])[0]
                            if fileid and not re.fullmatch(r"[A-Za-z0-9._-]+", fileid):
                                param_issues.append(
                                    "fileid 格式异常（仅允许字母数字、.、_、-）"
                                )

                            rkey = (query_params.get("rkey") or [None])[0]
                            if rkey and re.search(r"[^A-Za-z0-9._-]", rkey):
                                param_issues.append("rkey 包含特殊字符")

                            spec = (query_params.get("spec") or [None])[0]
                            if spec and not str(spec).isdigit():
                                param_issues.append("spec 参数应为数字")

                        # 根据响应内容与校验结果给出建议
                        suggestions: list[str] = []
                        if " " in cleaned_url or "%20" in cleaned_url:
                            suggestions.append("URL格式错误 → 检查URL编码")
                        if param_issues:
                            suggestions.append("参数错误 → 检查参数格式")
                        err_lower = err_text.lower() if err_text else ""
                        if any(keyword in err_lower for keyword in ["auth", "key"]):
                            suggestions.append("认证错误 → 检查API密钥")
                        if any(keyword in err_lower for keyword in ["limit", "频率", "限制"]):
                            suggestions.append("服务器限制 → 建议稍后重试")
                        if not suggestions:
                            suggestions.append("服务器限制 → 建议稍后重试")

                        logger.error(
                            "下载图像失败: HTTP %s %s url=%s 响应摘要=%s 建议=%s",
                            response.status,
                            response_reason,
                            cleaned_url,
                            err_text[:200],
                            "；".join(dict.fromkeys(suggestions)),
                        )

                        if self.verbose_logging:
                            logger.debug(
                                "HTTP 400 参数检查结果: %s",
                                "; ".join(param_issues) if param_issues else "未发现明显异常",
                            )
                            logger.debug("完整请求头: %s", headers or {})
                            logger.debug("User-Agent: %s", (headers or {}).get("User-Agent", ""))
                            logger.debug(
                                "Content-Type: %s, Accept: %s",
                                (headers or {}).get("Content-Type", "未设置"),
                                (headers or {}).get("Accept", "未设置"),
                            )
                            logger.debug(
                                "服务器响应详情: status=%s, reason=%s, phrase=%s, content-type=%s",
                                response.status,
                                response_reason,
                                getattr(response, "reason", ""),
                                response_content_type,
                            )
                            logger.debug(
                                "服务器响应体预览: %s",
                                err_text[:1000] if err_text else "<empty>",
                            )

                        if response.status == 400 and attempt < max_retries:
                            await asyncio.sleep(retry_interval * attempt)
                            continue
                        return None, None

                    content_type = response.headers.get("Content-Type", "")

                    if "/" in content_type:
                        image_format = content_type.split("/")[1].split(";")[0] or "png"
                    else:
                        image_format = "png"

                    target_path = None
                    if cache_key:
                        target_path = IMAGE_CACHE_DIR / f"{cache_key}.{image_format}"

                    image_path = await save_image_stream(
                        response.content, image_format, target_path=target_path
                    )
                    if image_path:
                        # 直接使用文件路径，不使用 file:// URI（根据 AstrBot 文档要求）
                        image_url_local = image_path
                        return image_url_local, image_path
            except aiohttp.ClientError as e:
                logger.error(f"下载图像发生网络异常: {e}")
            except Exception as e:
                logger.error(f"下载图像失败: {e}")

            if attempt < max_retries:
                await asyncio.sleep(retry_interval * attempt)

        return None, None


# 为了兼容性，创建APIClient别名
APIClient = GeminiAPIClient

# 全局 API 客户端实例
_api_client: GeminiAPIClient | None = None


def get_api_client(api_keys: list[str]) -> GeminiAPIClient:
    """获取或创建 API 客户端实例"""
    global _api_client
    if _api_client is None:
        _api_client = GeminiAPIClient(api_keys)
    return _api_client


def clear_api_client():
    """清除全局 API 客户端实例（用于测试）"""
    global _api_client
    _api_client = None
