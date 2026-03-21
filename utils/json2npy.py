import json
import numpy as np
import sys
from time import time

LUT_JSON = "rgb_to_bead_lut.json"
LUT_NPY = "rgb_to_bead_lut.npy"

def main():
    print(f"正在从 {LUT_JSON} 加载 LUT...")
    t0 = time()
    with open(LUT_JSON, "r") as f:
        lut_dict = json.load(f)
    t1 = time()
    print(f"JSON 加载完成，耗时: {t1 - t0:.2f} 秒")
    print(f"字典大小: {len(lut_dict)} 条目")

    # 创建 (256, 256, 256) 的 object 数组
    print("正在构建 NumPy 数组...")
    lut_array = np.empty((256, 256, 256), dtype=object)

    count = 0
    for key, code in lut_dict.items():
        r, g, b = map(int, key.split(','))
        lut_array[r, g, b] = code
        count += 1
        if count % 2_000_000 == 0:
            print(f"已处理 {count} / {len(lut_dict)}")

    t2 = time()
    print(f"数组构建完成，耗时: {t2 - t1:.2f} 秒")

    # 保存为 .npy
    print(f"正在保存到 {LUT_NPY}...")
    np.save(LUT_NPY, lut_array)
    t3 = time()
    print(f"保存完成！总耗时: {t3 - t0:.2f} 秒")
    print(f"文件已生成: {LUT_NPY}")

if __name__ == "__main__":
    main()