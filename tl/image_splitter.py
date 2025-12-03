import zipfile
from pathlib import Path
from typing import Any, List, Tuple

from PIL import Image
import cv2
import numpy as np

from astrbot.api import logger
from .tl_utils import get_plugin_data_dir

class SmartMemeSplitter:
    """
    智能表情包切分器 - 结合投影法和网格优化
    核心思想：
    1. 边缘检测 + 形态学处理
    2. 投影法检测网格边界
    3. 两阶段优化（粗调 + 精调）
    """

    def __init__(self, min_gap=5, edge_threshold=10):
        """
        Args:
            min_gap: 最小间隙宽度（像素）
            edge_threshold: 边缘检测阈值
        """
        self.min_gap = min_gap
        self.edge_threshold = edge_threshold
        self.last_row_lines = []
        self.last_col_lines = []

    def fft_denoise(self, gray: np.ndarray, keep_ratio: float = 0.08) -> np.ndarray:
        """FFT 低通去噪，保留中心低频区域"""
        try:
            f = np.fft.fft2(gray)
            fshift = np.fft.fftshift(f)
            rows, cols = gray.shape
            crow, ccol = rows // 2, cols // 2
            r = int(min(rows, cols) * keep_ratio)
            mask = np.zeros_like(fshift)
            mask[crow - r : crow + r, ccol - r : ccol + r] = 1
            fshift = fshift * mask
            f_ishift = np.fft.ifftshift(fshift)
            img_back = np.fft.ifft2(f_ishift)
            img_back = np.abs(img_back)
            img_back = np.clip(img_back, 0, 255).astype(np.uint8)
            return img_back
        except Exception as e:
            logger.debug(f"FFT 去噪失败，回退原灰度图: {e}")
            return gray

    def preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        预处理：FFT 去噪 -> Canny 边缘 -> 二值化 + 形态学

        Returns:
            (clean_img, edge_mask, bin_img)
        """
        # 处理透明背景
        if image.shape[2] == 4:
            alpha = image[:, :, 3]
            bg = np.ones_like(image[:, :, :3], dtype=np.uint8) * 255
            img_rgb = image[:, :, :3]
            alpha_factor = alpha[:, :, np.newaxis].astype(np.float32) / 255.0
            clean_img = (bg * (1 - alpha_factor) + img_rgb * alpha_factor).astype(np.uint8)
        else:
            clean_img = image.copy()

        gray = cv2.cvtColor(clean_img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        gray = self.fft_denoise(gray, keep_ratio=0.08)
        # 自适应阈值二值化 + 轻度中值滤波
        bin_img = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 5
        )
        bin_img = cv2.medianBlur(bin_img, 3)

        # 边缘检测
        edges = cv2.Canny(bin_img, 50, 150)

        # 形态学膨胀：连接文字和主体
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        dilated = cv2.dilate(edges, kernel, iterations=2)

        return clean_img, dilated, bin_img

    def find_grid_lines(self, projection: np.ndarray, img_size: int) -> List[int]:
        """
        基于投影找到网格分隔线（波谷位置）

        Args:
            projection: 投影直方图
            img_size: 图像尺寸（用于边界）

        Returns:
            网格线位置列表
        """
        # 平滑处理减少噪点
        smooth = np.convolve(projection, np.ones(3)/3, mode='same')

        lines = [0]  # 起始边界
        is_gap = False
        gap_start = 0

        for i, val in enumerate(smooth):
            if val <= self.edge_threshold:  # 间隙
                if not is_gap:
                    is_gap = True
                    gap_start = i
            else:  # 内容
                if is_gap:
                    is_gap = False
                    gap_end = i
                    # 间隙足够大才认为是分隔线
                    if gap_end - gap_start >= self.min_gap:
                        mid = (gap_start + gap_end) // 2
                        lines.append(mid)

        lines.append(img_size)  # 结束边界
        return lines

    def validate_line(
        self,
        edges: np.ndarray,
        position: int,
        is_vertical: bool,
        max_edge_ratio: float = 0.35,
    ) -> bool:
        """
        验证一条线是否压到过多轮廓

        Args:
            edges: 边缘图
            position: 线的位置
            is_vertical: True=垂直线, False=水平线
            max_edge_ratio: 最大边缘像素比例

        Returns:
            True=线有效, False=压到过多轮廓
        """
        h, w = edges.shape

        if is_vertical:  # 垂直线
            if position < 0 or position >= w:
                return False
            edge_pixels = np.sum(edges[:, position] > 0)
            total_pixels = h
        else:  # 水平线
            if position < 0 or position >= h:
                return False
            edge_pixels = np.sum(edges[position, :] > 0)
            total_pixels = w

        # 计算边缘像素比例
        edge_ratio = edge_pixels / total_pixels if total_pixels > 0 else 1.0
        return edge_ratio <= max_edge_ratio

    def refine_boundary(self, edges: np.ndarray, position: int,
                       is_vertical: bool, search_range: int = 20) -> int:
        """
        精细调整边界位置，找到最清晰的分隔线

        Args:
            edges: 边缘图
            position: 初始位置
            is_vertical: True=垂直线, False=水平线
            search_range: 搜索范围

        Returns:
            调整后的位置
        """
        h, w = edges.shape
        best_pos = position
        min_edges = float('inf')

        # 双向搜索
        for offset in range(-search_range, search_range + 1):
            test_pos = position + offset

            if is_vertical:  # 垂直线
                if test_pos < 0 or test_pos >= w:
                    continue
                edge_count = np.sum(edges[:, test_pos] > 0)
            else:  # 水平线
                if test_pos < 0 or test_pos >= h:
                    continue
                edge_count = np.sum(edges[test_pos, :] > 0)

            if edge_count < min_edges:
                min_edges = edge_count
                best_pos = test_pos
                if edge_count <= 1:  # 足够清晰
                    break

        return best_pos

    def get_median_grid_size(self, boxes: List[Tuple[int, int, int, int]]) -> Tuple[float, float]:
        """
        计算网格的中位数尺寸

        Args:
            boxes: 边界框列表 [(x, y, w, h), ...]

        Returns:
            (median_width, median_height): 中位数宽度和高度
        """
        if not boxes:
            return 0, 0

        widths = [w for _, _, w, _ in boxes]
        heights = [h for _, _, _, h in boxes]

        median_w = np.median(widths)
        median_h = np.median(heights)

        return median_w, median_h

    def generate_boxes_from_grid_lines(self, edges: np.ndarray,
                                       row_lines: List[int],
                                       col_lines: List[int]) -> List[Tuple[int, int, int, int]]:
        """
        根据验证后的网格线生成边界框

        Args:
            edges: 边缘图
            row_lines: 行线位置列表
            col_lines: 列线位置列表

        Returns:
            边界框列表 [(x, y, w, h), ...]
        """
        h, w = edges.shape
        boxes = []

        # 验证并过滤行线
        valid_row_lines = []
        for line_pos in row_lines:
            if line_pos == 0 or line_pos == h:  # 保留边界
                valid_row_lines.append(line_pos)
            elif self.validate_line(edges, line_pos, is_vertical=False):
                valid_row_lines.append(line_pos)

        # 验证并过滤列线
        valid_col_lines = []
        for line_pos in col_lines:
            if line_pos == 0 or line_pos == w:  # 保留边界
                valid_col_lines.append(line_pos)
            elif self.validate_line(edges, line_pos, is_vertical=True):
                valid_col_lines.append(line_pos)

        # 根据网格线生成所有边界框
        for i in range(len(valid_row_lines) - 1):
            for j in range(len(valid_col_lines) - 1):
                y1 = valid_row_lines[i]
                y2 = valid_row_lines[i + 1]
                x1 = valid_col_lines[j]
                x2 = valid_col_lines[j + 1]

                # 检查尺寸是否足够大
                box_w = x2 - x1
                box_h = y2 - y1
                if box_w < 30 or box_h < 30:  # 跳过太小的格子
                    continue

                # 验证区域有足够内容（提高阈值）
                roi = edges[y1:y2, x1:x2]
                edge_density = np.sum(roi) / 255.0 / (box_w * box_h)  # 边缘密度
                if np.sum(roi) / 255.0 > 500 and edge_density > 0.02:  # 有足够内容且密度足够
                    boxes.append((x1, y1, box_w, box_h))

        return boxes

    def detect_grid(
        self,
        image: np.ndarray,
        debug: bool = False,
    ) -> List[Tuple[int, int, int, int]]:
        """
        检测网格并返回所有表情包的边界框

        Returns:
            [(x, y, w, h), ...]: 边界框列表
        """
        h, w = image.shape[:2]
        clean_img, edges, bin_img = self.preprocess(image)

        # 阶段1：投影法粗定位
        # 水平投影 -> 找行边界
        row_proj = np.sum(bin_img, axis=1) / 255.0
        row_lines = self.find_grid_lines(row_proj, h)

        if debug:
            logger.debug(f"检测到 {len(row_lines)-1} 行")

        boxes = []
        all_col_lines = set()  # 收集所有列线

        # 对每一行单独做垂直投影
        for i in range(len(row_lines) - 1):
            y1 = row_lines[i]
            y2 = row_lines[i + 1]

            # 跳过太小的行
            if y2 - y1 < 20:
                continue

            # 垂直投影 -> 找列边界
            row_mask_bin = bin_img[y1:y2, :]
            row_mask_edge = edges[y1:y2, :]
            if np.sum(row_mask_bin) < 500:  # 内容太少，跳过
                continue

            col_proj = np.sum(row_mask_bin, axis=0) / 255.0
            col_lines = self.find_grid_lines(col_proj, w)

            # 收集列线位置
            all_col_lines.update(col_lines)

            # 生成该行的所有格子
            for j in range(len(col_lines) - 1):
                x1 = col_lines[j]
                x2 = col_lines[j + 1]

                # 跳过太小的列
                if x2 - x1 < 30:
                    continue

                # 验证区域有足够内容
                box_w = x2 - x1
                box_h = y2 - y1
                roi_edge = row_mask_edge[:, x1:x2]
                roi_bin = row_mask_bin[:, x1:x2]
                edge_density = np.sum(roi_edge) / 255.0 / (box_w * box_h)
                if np.sum(roi_bin) / 255.0 > 500 and edge_density > 0.02:
                    boxes.append((x1, y1, box_w, box_h))

        # 阶段2：精细调整边界
        if debug:
            logger.debug(f"=== 阶段2：精细调整 ===")

        refined_boxes = []
        for x, y, w_box, h_box in boxes:
            # 计算搜索范围（基于框大小的20%）
            search_w = max(10, int(w_box * 0.2))
            search_h = max(10, int(h_box * 0.2))

            # 调整四条边
            new_x = self.refine_boundary(edges, x, True, search_w)
            new_x2 = self.refine_boundary(edges, x + w_box, True, search_w)
            new_y = self.refine_boundary(edges, y, False, search_h)
            new_y2 = self.refine_boundary(edges, y + h_box, False, search_h)

            # 确保边界有效
            new_w = max(20, new_x2 - new_x)
            new_h = max(20, new_y2 - new_y)

            refined_boxes.append((new_x, new_y, new_w, new_h))

        if debug:
            logger.debug(f"最终检测到 {len(refined_boxes)} 个表情包")

        # 阶段3：处理过大网格
        if len(refined_boxes) > 1:  # 至少有2个网格才能判断大小
            median_w, median_h = self.get_median_grid_size(refined_boxes)

            if debug:
                logger.debug(f"=== 阶段3：处理过大网格 ===")
                logger.debug(f"标准网格尺寸: {median_w:.0f} x {median_h:.0f}")

            # 检测过大网格并计算需要细分的网格线
            extra_row_lines = set()
            extra_col_lines = set()

            for box in refined_boxes:
                x, y, w_box, h_box = box

                # 检查是否过大（1.3倍阈值）
                is_too_wide = w_box > median_w * 1.3
                is_too_tall = h_box > median_h * 1.3

                if is_too_wide or is_too_tall:
                    if debug:
                        logger.debug(f"检测到过大网格 ({w_box}x{h_box})，添加细分网格线...")

                    # 添加细分的列线
                    if is_too_wide:
                        cols = max(1, round(w_box / median_w))
                        for i in range(1, cols):
                            new_col = int(x + i * w_box / cols)
                            extra_col_lines.add(new_col)

                    # 添加细分的行线
                    if is_too_tall:
                        rows = max(1, round(h_box / median_h))
                        for i in range(1, rows):
                            new_row = int(y + i * h_box / rows)
                            extra_row_lines.add(new_row)

            # 合并原有网格线和新增的细分线
            if extra_row_lines:
                row_lines = sorted(set(row_lines) | extra_row_lines)
                if debug:
                    logger.debug(f"添加了 {len(extra_row_lines)} 条行细分线")

            if extra_col_lines:
                all_col_lines.update(extra_col_lines)
                if debug:
                    logger.debug(f"添加了 {len(extra_col_lines)} 条列细分线")

        # 存储网格线信息供GUI使用
        self.last_row_lines = row_lines
        self.last_col_lines = sorted(list(all_col_lines))

        # 阶段4：根据最终网格线重新生成边界框
        if debug:
            logger.debug(f"=== 阶段4：根据网格线生成最终边界框 ===")

        final_boxes = self.generate_boxes_from_grid_lines(edges, row_lines, sorted(list(all_col_lines)))

        if debug:
            logger.debug(f"最终生成 {len(final_boxes)} 个表情包")

        return final_boxes

def split_image(
    image_path: str,
    rows: int = 6,
    cols: int = 4,
    output_dir: str | None = None,
    bboxes: list[dict[str, Any]] | None = None,
) -> list[str]:
    """
    使用 SmartMemeSplitter 智能切分图片

    Args:
        image_path: 源图片路径
        rows: 保留参数以兼容旧接口（新方法中不再使用）
        cols: 保留参数以兼容旧接口（新方法中不再使用）
        output_dir: 输出目录，如果不指定则使用插件数据目录下的 split_output
        bboxes: 保留参数以兼容旧接口（新方法中不再使用）

    Returns:
        List[str]: 切分后的图片文件路径列表，按顺序排列
    """
    try:
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

        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            logger.error(f"无法读取图像: {image_path}")
            return []

        # 使用 SmartMemeSplitter 进行智能切分
        splitter = SmartMemeSplitter(min_gap=5, edge_threshold=10)
        boxes = splitter.detect_grid(img, debug=True)

        if not boxes:
            logger.warning("智能切分未检测到网格")
            return []

        # 生成掩码预览（便于调试网格线）
        try:
            mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
            for x, y, w_box, h_box in boxes:
                cv2.rectangle(mask, (x, y), (x + w_box, y + h_box), 255, 2)
            mask_file = final_output_dir / f"{base_name}_mask.png"
            cv2.imwrite(str(mask_file), mask)
        except Exception as e:
            logger.debug(f"生成掩码预览失败: {e}")

        # 保存智能切分结果
        for idx, (x, y, w, h) in enumerate(boxes, 1):
            # 添加2像素padding
            pad = 2
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(img.shape[1], x + w + pad)
            y2 = min(img.shape[0], y + h + pad)

            crop = img[y1:y2, x1:x2]
            file_name = f"{base_name}_{idx:03d}.png"
            file_path = final_output_dir / file_name
            cv2.imwrite(str(file_path), crop)
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
