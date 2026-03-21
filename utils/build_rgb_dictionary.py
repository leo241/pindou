import os
import json
import numpy as np
from PIL import Image
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont
from skimage.color import rgb2lab
from sklearn.cluster import KMeans
from collections import Counter
from skimage.color import deltaE_ciede2000
from scipy.spatial import KDTree
from PIL import ImageFilter
from tqdm import tqdm

# 读取拼豆色卡
with open("beads_palette_221_correct.json") as f:
    beads = json.load(f)

for k in beads:
    beads[k] = tuple(beads[k])

# -------------------------
# RGB -> LAB
# -------------------------
def rgb_to_lab(rgb):
    arr = np.array([[rgb]], dtype=float) / 255
    return rgb2lab(arr)[0][0]

# 预计算LAB色卡
palette_codes = []
palette_lab = []
for code, rgb in beads.items():
    palette_codes.append(code)
    palette_lab.append(rgb_to_lab(rgb))
palette_lab = np.array(palette_lab)
tree = KDTree(palette_lab)

# -------------------------
# 构建 LUT：RGB (0-255) -> bead_code
# -------------------------
LUT_FILE = "rgb_to_bead_lut.json"
rgb_lut = None

def build_rgb_lut():
    """构建 256x256x256 的 RGB 到 bead_code 的映射表"""
    print("正在构建 RGB 查找表... 这可能需要几分钟。")
    lut = {}
    total = 256 * 256 * 256
    count = 0
    for r in tqdm(range(256)):
        for g in range(256):
            for b in range(256):
                rgb = (r, g, b)
                lab = rgb_to_lab(rgb)
                # 使用和 nearest_bead 完全相同的逻辑
                dist, idx = tree.query(lab, k=8)
                best = None
                best_dist = 1e9
                for i in idx:
                    candidate_lab = palette_lab[i]
                    d = deltaE_ciede2000(
                        lab.reshape(1,1,3),
                        candidate_lab.reshape(1,1,3)
                    )[0][0]
                    if d < best_dist:
                        best_dist = d
                        best = palette_codes[i]
                lut[f"{r},{g},{b}"] = best
                count += 1
                if count % 1000000 == 0:
                    print(f"已处理 {count}/{total} ({100*count/total:.1f}%)")
    with open(LUT_FILE, "w") as f:
        json.dump(lut, f)
    print(f"LUT 已保存到 {LUT_FILE}")
    return lut

if __name__ == '__main__':
    lut = build_rgb_lut()