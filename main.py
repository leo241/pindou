import json
from time import time
from utils.tools import load_lut_npy, generate_pattern, render_pattern, bead_stats, save_grid_to_json


if __name__ == "__main__":
    # 1.读取拼豆色卡
    with open("load/beads_palette_221_correct.json") as f:
        beads = json.load(f)
    for k in beads:
        beads[k] = tuple(beads[k])
    t0 = time()
    npy_file = "load/rgb_to_bead_lut.npy"
    rgb_lut_array = load_lut_npy(npy_file)
    t1 = time()

    # 2.设置超参数
    name = "fox"
    compress_colors = 30
    N = 100
    use_sharp = True
    open_mirror = True

    # 3.颜色查找
    grid = generate_pattern(f"input/{name}.jpg",
                            rgb_lut_array,
                            N=N, # 图纸大小 N*N
                            compress_colors=compress_colors, # 压缩色数
                            use_sharp=use_sharp, # 使用锐化
                            )
    save_grid_to_json(grid,f"grid/{name}.json")
    mirrored_grid = [row[::-1] for row in grid]
    save_grid_to_json(grid, f"grid/{name}_mirror.json")
    t2 = time()
    # 4.渲染图纸
    img = render_pattern(grid, beads)
    if open_mirror:
        img_mirror = render_pattern(mirrored_grid, beads)
    t3 = time()
    img.save(f"output/{name}.png")
    if open_mirror:
        img_mirror.save(f"output/{name}_mirror.png")
    # 5.日志打印，用户交互
    stats = bead_stats(grid)
    # print("颜色统计:")
    sorted_stats = sorted(stats.items(), key=lambda x: x[1], reverse=True)
    # for k, v in sorted_stats:
    #     print(k, v)
    print(f'\n1.加载npy：{round(t1-t0,2)}s')
    print(f'2.颜色查找：{round(t2 - t1,2)}s')
    print(f'3.图纸渲染：{round(t3 - t2,2)}s')
    print("\n 总色数:", len(stats))