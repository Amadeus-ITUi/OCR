from __future__ import annotations


# 这份文件专门维护白色题板 ROI 检测参数，目标场景是：
# 电子显示屏中的白色题板 + 黑色题目 + 后续 OCR 识别。
#
# 推荐调参顺序：
# 1. 先调 min_roi_area_ratio：距离远一点就检测不到时优先看这个
# 2. 再调 edge_threshold：题板很明显但边缘响应不够时降低它
# 3. 再调 white_threshold：白底够亮却仍然漏检时略微降低它
# 4. 最后调 rectangle_ratio_tolerance / quadrilateral_ratio_tolerance
#    用于放宽透视、畸变或轮廓不够规整时的比例限制


DEFAULT_ROI_TUNING_VALUES: dict[str, int | float] = {
    # 白色阈值：越高越严格，只有更亮的区域才会被认为是题板候选。
    # 远距离、屏幕发灰、整体亮度偏低时，可以先尝试降到 185-195。
    "white_threshold": 200,

    # 白色连通域主路径阈值：默认沿用 white_threshold。
    "component_white_threshold": 200,

    # 白色连通域最小面积占比。
    "component_min_area_ratio": 0.03,

    # 最多评估前几个最大白色连通域。
    "component_max_candidates": 6,

    # 连通域面积 / 最小外接矩形面积，下限越高越像“实心白题板”。
    "component_fill_ratio_threshold": 0.72,

    # 白色连通域成功后的外扩 padding。
    "component_padding": 16,

    # 角点搜索时的候选扩展边距。
    "corner_search_margin": 18,

    # 边缘梯度阈值：越高越严格，只有更明显的边缘才会参与题板候选。
    # 远一点就检测不到，但题板边界肉眼很明显时，优先尝试降到 40-70。
    "edge_threshold": 80,

    # 最小 ROI 面积占比：题板在整张图里至少要占多大比例才会被接受。
    # 距离远导致题板变小、肉眼明显但总被漏掉时，优先尝试降到 0.03-0.06。
    "min_roi_area_ratio": 0.03,

    # 矩形路径比例容差：越大越宽松。
    # 题板轮廓像矩形，但总因为比例不准被拒绝时，尝试提高到 0.25-0.35。
    "rectangle_ratio_tolerance": 0.2,

    # 四边形路径比例容差：用于透视更明显、角点更歪时的放宽。
    # 倾斜拍摄、远距离、边缘不规整时，可尝试提高到 0.32-0.45。
    "quadrilateral_ratio_tolerance": 0.28,

    # 题板目标比例：当前场景固定为 16:9。
    "target_aspect_ratio": 16.0 / 9.0,

    # ROI 外扩 padding：找到题板后，向外稍微补一点边，避免裁太紧。
    "roi_padding": 12,

    # 透视拉正后的输出尺寸。
    "perspective_width": 1280,
    "perspective_height": 720,

    # OCR 输入图放大倍数。
    # 黑字偏细、OCR 容易漏笔画时可以适当提高；过大则会放大噪声。
    "scale_factor": 2.0,
}
