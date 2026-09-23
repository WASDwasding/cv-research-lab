"""把第 1 天的论文壳打成 PDF。不依赖 LaTeX，也不裁剪整包字体。

用法（在仓库根目录）:
    python paper/build_shell.py
输出:
    paper/shell.pdf
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

FONT = Path(r"C:\Windows\Fonts\simhei.ttf")
OUT = Path(__file__).resolve().parent / "shell.pdf"
PAGE = (1240, 1754)
MARGIN = 80

BLOCKS = [
    ("title", "Screen Damage Severity Estimation for Phones and Tablets"),
    ("meta", "论文壳  2026-09-22  第 1 天  cv-research-lab"),
    ("body", "Keywords: damage severity, ordinal classification, visual inspection"),
    ("head", "摘要 / Abstract"),
    ("body", "第 1 天只建壳。摘要以后写三档定义、对照结果、以及模型在哪里失效。本页不填任何实验数字。"),
    ("head", "1 引言 / Introduction"),
    ("body", "任务：已知是手机或平板屏幕，给一张屏幕图一个有序档次：无 / 轻 / 中重。不评后盖和边框。三条贡献主张："),
    ("body", "1. 分级定义。无 / 轻 / 中重有书面可见特征和边界例，见 experiments/label_protocol.md。"),
    ("body", "2. 对照。同一划分上比较整图分类、有序损失、先定位再分级。负结果保留。"),
    ("body", "3. 何时失效。失败类型和真实拍摄小集上的掉点写进正文。不把贡献写成提出新网络。"),
    ("head", "2 相关工作 / Related Work"),
    ("body", "待写三小节：表面缺陷、有序分类、成色与外观质检。"),
    ("head", "3 方法 / Method"),
    ("body", "3.1 标签定义。3.2 整图分类与有序损失。3.3 先定位再分级。"),
    ("head", "4 实验 / Experiments"),
    ("body", "4.1 数据与划分。4.2 指标与实现。4.3 主对照。4.4 定位再分级。4.5 消融与有序误差。4.6 失败案例与真实照片。"),
    ("head", "5 结论 / Conclusion"),
    ("body", "数字冻结后再写局限：反光、跨产品、标注主观、公开集映射不等于真实成色。"),
]


def wrap(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont, width: int) -> list[str]:
    lines: list[str] = []
    for paragraph in text.split("\n"):
        buf = ""
        for ch in paragraph:
            trial = buf + ch
            if draw.textlength(trial, font=font) <= width:
                buf = trial
            else:
                if buf:
                    lines.append(buf)
                buf = ch
        lines.append(buf)
    return lines


def main() -> None:
    fonts = {
        "title": ImageFont.truetype(str(FONT), 36),
        "meta": ImageFont.truetype(str(FONT), 24),
        "head": ImageFont.truetype(str(FONT), 28),
        "body": ImageFont.truetype(str(FONT), 22),
    }
    gaps = {"title": 18, "meta": 14, "head": 16, "body": 10}
    width, height = PAGE
    text_width = width - 2 * MARGIN
    pages: list[Image.Image] = []
    image = Image.new("RGB", PAGE, "white")
    draw = ImageDraw.Draw(image)
    y = MARGIN

    def new_page() -> None:
        nonlocal image, draw, y
        pages.append(image)
        image = Image.new("RGB", PAGE, "white")
        draw = ImageDraw.Draw(image)
        y = MARGIN

    for kind, text in BLOCKS:
        font = fonts[kind]
        gap = gaps[kind]
        if kind == "head":
            y += 12
        for line in wrap(draw, text, font, text_width):
            if y > height - MARGIN:
                new_page()
            draw.text((MARGIN, y), line, fill="black", font=font)
            y += font.size + gap
    pages.append(image)
    pages[0].save(
        OUT,
        "PDF",
        resolution=150.0,
        save_all=True,
        append_images=pages[1:],
    )
    print(OUT)


if __name__ == "__main__":
    main()
