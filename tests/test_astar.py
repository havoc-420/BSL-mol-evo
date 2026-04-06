#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A* 算法演示
============
A*（A-Star）是一种启发式图搜索算法，结合了 Dijkstra 算法的最优性
和贪心算法的速度，通过评估函数 f(n) = g(n) + h(n) 进行搜索。

- g(n)：从起点到当前节点的实际代价
- h(n)：从当前节点到终点的启发式估计代价
- f(n)：总估计代价，用于优先队列排序

运行方式：
    python tests/test_astar.py
"""

import heapq
import time
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


# ─────────────────────────────────────────────
# 节点类
# ─────────────────────────────────────────────

class Node:
    """搜索节点，封装位置、代价和父节点信息"""

    def __init__(self, pos, g=0.0, h=0.0, parent=None):
        """
        参数
        ----
        pos    : (row, col) 网格坐标
        g      : 从起点到本节点的实际代价
        h      : 从本节点到终点的启发估计代价
        parent : 父节点（用于路径回溯）
        """
        self.pos = pos
        self.g = g
        self.h = h
        self.parent = parent

    @property
    def f(self):
        """总估计代价 f = g + h"""
        return self.g + self.h

    # 优先队列比较：先按 f，相同时按 h（更接近终点优先）
    def __lt__(self, other):
        if self.f != other.f:
            return self.f < other.f
        return self.h < other.h

    def __eq__(self, other):
        return self.pos == other.pos

    def __hash__(self):
        return hash(self.pos)


# ─────────────────────────────────────────────
# 启发函数
# ─────────────────────────────────────────────

def heuristic_manhattan(a, b):
    """曼哈顿距离：适合 4 方向移动"""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def heuristic_euclidean(a, b):
    """欧几里得距离：适合 8 方向移动"""
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def heuristic_zero(_a, _b):
    """零启发：退化为 Dijkstra（仅用 g，保证最优但效率低）"""
    return 0.0


def heuristic_greedy(a, b, weight=1e6):
    """超大权重启发：近似贪心 BFS（仅用 h，速度快但可能不最优）"""
    return heuristic_manhattan(a, b) * weight


# ─────────────────────────────────────────────
# A* 核心搜索
# ─────────────────────────────────────────────

def astar(grid, start, end,
          heuristic=heuristic_manhattan,
          allow_diagonal=False):
    """
    在二维网格中执行 A* 搜索。

    参数
    ----
    grid           : 2D numpy 数组，0=通行，1=障碍物
    start          : (row, col) 起点
    end            : (row, col) 终点
    heuristic      : 启发函数 h(a, b) -> float
    allow_diagonal : 是否允许对角线移动（8方向）

    返回
    ----
    path           : 从起点到终点的坐标列表（找不到则为空列表）
    explored       : 已探索节点坐标列表（按探索顺序）
    """
    rows, cols = grid.shape

    # 移动方向：4方向或8方向
    if allow_diagonal:
        directions = [
            (-1, 0), (1, 0), (0, -1), (0, 1),   # 上下左右
            (-1, -1), (-1, 1), (1, -1), (1, 1),  # 斜向
        ]
        move_costs = [1.0, 1.0, 1.0, 1.0,
                      math.sqrt(2), math.sqrt(2), math.sqrt(2), math.sqrt(2)]
    else:
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        move_costs = [1.0, 1.0, 1.0, 1.0]

    # 初始化起点
    start_node = Node(start, g=0.0, h=heuristic(start, end))

    # open_list：优先队列，元素为 (f, node)
    open_list = []
    heapq.heappush(open_list, start_node)

    # 记录每个位置的最优 g 值（避免重复扩展）
    g_best = {start: 0.0}

    # closed_set：已完成扩展的节点
    closed_set = set()

    # 记录探索顺序（用于可视化）
    explored = []

    while open_list:
        current = heapq.heappop(open_list)

        # 跳过已经用更优代价处理过的节点
        if current.pos in closed_set:
            continue
        if current.g > g_best.get(current.pos, float('inf')):
            continue

        closed_set.add(current.pos)
        explored.append(current.pos)

        # 到达终点，回溯路径
        if current.pos == end:
            path = []
            node = current
            while node is not None:
                path.append(node.pos)
                node = node.parent
            return path[::-1], explored

        # 扩展邻居
        r, c = current.pos
        for (dr, dc), cost in zip(directions, move_costs):
            nr, nc = r + dr, c + dc

            # 边界检查
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
            # 障碍物检查
            if grid[nr, nc] == 1:
                continue
            # 已关闭节点跳过
            neighbor_pos = (nr, nc)
            if neighbor_pos in closed_set:
                continue

            new_g = current.g + cost

            # 只有更优路径才入队
            if new_g < g_best.get(neighbor_pos, float('inf')):
                g_best[neighbor_pos] = new_g
                h = heuristic(neighbor_pos, end)
                neighbor_node = Node(neighbor_pos, g=new_g, h=h, parent=current)
                heapq.heappush(open_list, neighbor_node)

    # 未找到路径
    return [], explored


# ─────────────────────────────────────────────
# 迷宫生成
# ─────────────────────────────────────────────

def make_maze(rows=20, cols=20, seed=42):
    """
    生成带障碍物的测试迷宫。
    0=通行，1=障碍物，保证起终点附近畅通。
    """
    rng = np.random.default_rng(seed)
    grid = np.zeros((rows, cols), dtype=int)

    # 随机放置障碍物（约 30% 密度）
    obstacle_mask = rng.random((rows, cols)) < 0.30
    grid[obstacle_mask] = 1

    # 确保起点和终点及其周围可通行
    for dr in range(-1, 2):
        for dc in range(-1, 2):
            r0, c0 = 1 + dr, 1 + dc
            r1, c1 = rows - 2 + dr, cols - 2 + dc
            if 0 <= r0 < rows and 0 <= c0 < cols:
                grid[r0, c0] = 0
            if 0 <= r1 < rows and 0 <= c1 < cols:
                grid[r1, c1] = 0

    return grid


def make_complex_maze(rows=25, cols=25):
    """
    手工设计的蛇形迷宫，包含交替横向挡墙（各留出口），
    保证 (0,0)→(rows-1,cols-1) 可达，用于对比算法探索差异。
    """
    grid = np.zeros((rows, cols), dtype=int)

    # 第1段横墙：从左延伸，右端留出口（最后4列空）
    for c in range(0, cols - 4):
        grid[5, c] = 1

    # 第2段横墙：从右延伸，左端留出口（前4列空）
    for c in range(4, cols):
        grid[11, c] = 1

    # 第3段横墙：从左延伸，右端留出口
    for c in range(0, cols - 4):
        grid[17, c] = 1

    # 第4段横墙：从右延伸，左端留出口
    for c in range(4, cols):
        grid[22, c] = 1

    return grid


# ─────────────────────────────────────────────
# 可视化
# ─────────────────────────────────────────────

def visualize_result(grid, path, explored, start, end,
                     title="A* 路径搜索", ax=None):
    """
    在 matplotlib axes 上可视化搜索结果。

    颜色方案
    --------
    白色  : 可通行区域
    灰色  : 障碍物
    浅蓝色: 已探索区域
    黄→红 : 最终路径（渐变）
    绿色  : 起点
    红色  : 终点
    """
    rows, cols = grid.shape
    standalone = ax is None
    if standalone:
        _fig, ax = plt.subplots(figsize=(8, 8))

    # 底图：白色=通行，深灰=障碍物
    img = np.ones((rows, cols, 3))  # RGB 全白
    obstacle_mask = grid == 1
    img[obstacle_mask] = [0.3, 0.3, 0.3]  # 深灰色障碍物

    # 绘制已探索节点（浅蓝色）
    for r, c in explored:
        if (r, c) != start and (r, c) != end:
            img[r, c] = [0.68, 0.85, 0.90]  # 浅蓝

    # 绘制路径（黄→红渐变）
    if path:
        n = len(path)
        cmap = plt.colormaps['autumn']
        for i, (r, c) in enumerate(path):
            color = cmap(i / max(n - 1, 1))[:3]
            img[r, c] = color

    # 起点（绿色）和终点（红色）
    img[start] = [0.0, 0.8, 0.0]
    img[end] = [0.9, 0.1, 0.1]

    ax.imshow(img, interpolation='nearest', origin='upper')

    # 标注起终点文字
    ax.text(start[1], start[0], 'S', ha='center', va='center',
            fontsize=9, fontweight='bold', color='white')
    ax.text(end[1], end[0], 'E', ha='center', va='center',
            fontsize=9, fontweight='bold', color='white')

    # 图例
    legend_elements = [
        Patch(facecolor=(0.68, 0.85, 0.90), label=f'已探索 ({len(explored)})'),
        Patch(facecolor=(1.0, 0.5, 0.0),    label=f'路径 ({len(path)})'),
        Patch(facecolor=(0.0, 0.8, 0.0),    label='起点'),
        Patch(facecolor=(0.9, 0.1, 0.1),    label='终点'),
        Patch(facecolor=(0.3, 0.3, 0.3),    label='障碍物'),
    ]
    ax.legend(handles=legend_elements, loc='upper right',
              fontsize=7, framealpha=0.85)

    ax.set_title(title, fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])

    if standalone:
        plt.tight_layout()
        plt.show()


# ─────────────────────────────────────────────
# Demo 1：基本 A* 演示
# ─────────────────────────────────────────────

def demo_basic():
    """基本 A* 演示：随机迷宫 + 曼哈顿启发 + 可视化"""
    print("\n" + "=" * 55)
    print("  Demo 1：基本 A* 路径搜索（随机迷宫）")
    print("=" * 55)

    grid = make_maze(rows=20, cols=20, seed=42)
    start = (1, 1)
    end = (18, 18)

    t0 = time.perf_counter()
    path, explored = astar(grid, start, end,
                           heuristic=heuristic_manhattan,
                           allow_diagonal=False)
    elapsed = (time.perf_counter() - t0) * 1000

    if path:
        print(f"  ✓ 找到路径！")
        print(f"    路径长度  : {len(path)} 步")
        print(f"    探索节点数: {len(explored)}")
        print(f"    耗时      : {elapsed:.3f} ms")
    else:
        print("  ✗ 未找到路径（迷宫不可达）")

    _fig, ax = plt.subplots(figsize=(7, 7))
    visualize_result(grid, path, explored, start, end,
                     title=f"A* 基本演示  路径={len(path)}步  探索={len(explored)}节点",
                     ax=ax)
    plt.tight_layout()
    plt.savefig("/tmp/astar_demo_basic.png", dpi=120)
    print("  图像已保存至 /tmp/astar_demo_basic.png")
    plt.show()

    return path, explored


# ─────────────────────────────────────────────
# Demo 2：算法对比
# ─────────────────────────────────────────────

def demo_compare():
    """
    对比三种搜索策略：
    - Dijkstra  (h=0，仅 g)
    - 贪心 BFS   (h 权重极大，仅 h)
    - A*         (g + h 曼哈顿)
    - A* 欧几里得 (g + h 欧几里得)
    """
    print("\n" + "=" * 55)
    print("  Demo 2：算法对比（复杂迷宫）")
    print("=" * 55)

    grid = make_complex_maze(rows=25, cols=25)
    start = (0, 0)
    end = (24, 24)

    algorithms = [
        ("Dijkstra\n(h=0)",        heuristic_zero,      False),
        ("贪心 BFS\n(仅h)",         heuristic_greedy,    False),
        ("A* 曼哈顿\n(g+h)",        heuristic_manhattan, False),
        ("A* 欧几里得\n(g+h, 8向)", heuristic_euclidean, True),
    ]

    results = []
    for name, h_func, diagonal in algorithms:
        t0 = time.perf_counter()
        path, explored = astar(grid, start, end,
                               heuristic=h_func,
                               allow_diagonal=diagonal)
        elapsed = (time.perf_counter() - t0) * 1000
        results.append((name, path, explored, elapsed))

        tag = name.replace('\n', ' ')
        status = "✓" if path else "✗"
        print(f"  {status} {tag:<20} "
              f"路径={len(path):>3}步  "
              f"探索={len(explored):>4}节点  "
              f"耗时={elapsed:.3f}ms")

    # 2×2 子图对比
    fig, axes = plt.subplots(2, 2, figsize=(13, 13))
    fig.suptitle("路径搜索算法对比", fontsize=14, fontweight='bold')

    for ax, (name, path, explored, elapsed) in zip(axes.flat, results):
        title = f"{name.replace(chr(10), ' ')}  路径={len(path)}  探索={len(explored)}"
        visualize_result(grid, path, explored, start, end,
                         title=title, ax=ax)

    plt.tight_layout()
    plt.savefig("/tmp/astar_demo_compare.png", dpi=120)
    print("\n  对比图已保存至 /tmp/astar_demo_compare.png")
    plt.show()

    return results


# ─────────────────────────────────────────────
# Demo 3：启发函数权重影响
# ─────────────────────────────────────────────

def demo_weight_effect():
    """
    展示启发函数权重 w 对 A* 的影响：
    f(n) = g(n) + w * h(n)
    w < 1 : 更保守，接近 Dijkstra，探索多但路径更优
    w = 1 : 标准 A*，最优且高效
    w > 1 : 加速搜索，但可能牺牲最优性（Weighted A*）
    """
    print("\n" + "=" * 55)
    print("  Demo 3：启发权重 w 对 A* 的影响")
    print("=" * 55)

    grid = make_maze(rows=30, cols=30, seed=7)
    start = (1, 1)
    end = (28, 28)
    weights = [0.5, 1.0, 2.0, 5.0]

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    fig.suptitle("Weighted A*：启发权重 w 的影响  f(n)=g(n)+w·h(n)",
                 fontsize=13, fontweight='bold')

    for ax, w in zip(axes, weights):
        def weighted_h(a, b, _w=w):
            return _w * heuristic_manhattan(a, b)

        t0 = time.perf_counter()
        path, explored = astar(grid, start, end,
                               heuristic=weighted_h,
                               allow_diagonal=False)
        elapsed = (time.perf_counter() - t0) * 1000

        tag = "最优" if w == 1.0 else ("保守" if w < 1.0 else "加速")
        title = f"w={w} ({tag})\n路径={len(path)}  探索={len(explored)}"
        visualize_result(grid, path, explored, start, end,
                         title=title, ax=ax)

        status = "✓" if path else "✗"
        print(f"  {status} w={w:<4}  路径={len(path):>3}步  "
              f"探索={len(explored):>4}节点  耗时={elapsed:.3f}ms")

    plt.tight_layout()
    plt.savefig("/tmp/astar_demo_weight.png", dpi=110)
    print("\n  权重对比图已保存至 /tmp/astar_demo_weight.png")
    plt.show()


# ─────────────────────────────────────────────
# 单元测试
# ─────────────────────────────────────────────

def run_tests():
    """运行基本单元测试，验证算法正确性"""
    print("\n" + "=" * 55)
    print("  单元测试")
    print("=" * 55)
    all_pass = True

    # ── 测试 1：简单 3×3 网格，无障碍物 ──
    grid = np.zeros((3, 3), dtype=int)
    path, _ = astar(grid, (0, 0), (2, 2),
                    heuristic=heuristic_manhattan,
                    allow_diagonal=False)
    ok = len(path) == 5 and path[0] == (0, 0) and path[-1] == (2, 2)
    print(f"  {'✓' if ok else '✗'} 测试1：3×3 无障碍 4方向  路径长={len(path)} (期望5)")
    all_pass &= ok

    # ── 测试 2：完全阻塞，应返回空路径 ──
    grid_blocked = np.zeros((5, 5), dtype=int)
    grid_blocked[0, 1] = 1
    grid_blocked[1, 0] = 1
    path2, _ = astar(grid_blocked, (0, 0), (4, 4),
                     heuristic=heuristic_manhattan)
    ok2 = len(path2) == 0
    print(f"  {'✓' if ok2 else '✗'} 测试2：起点被围堵  路径长={len(path2)} (期望0)")
    all_pass &= ok2

    # ── 测试 3：起点即终点 ──
    grid = np.zeros((5, 5), dtype=int)
    path3, _ = astar(grid, (2, 2), (2, 2),
                     heuristic=heuristic_manhattan)
    ok3 = len(path3) == 1 and path3[0] == (2, 2)
    print(f"  {'✓' if ok3 else '✗'} 测试3：起点=终点  路径长={len(path3)} (期望1)")
    all_pass &= ok3

    # ── 测试 4：A* 找到的路径与 Dijkstra 等长（最优性验证）──
    grid = make_maze(rows=15, cols=15, seed=99)
    start, end = (1, 1), (13, 13)
    path_astar, _ = astar(grid, start, end, heuristic=heuristic_manhattan)
    path_dijkstra, _ = astar(grid, start, end, heuristic=heuristic_zero)
    ok4 = len(path_astar) == len(path_dijkstra)
    print(f"  {'✓' if ok4 else '✗'} 测试4：A* 最优性  "
          f"A*={len(path_astar)} Dijkstra={len(path_dijkstra)} (应相等)")
    all_pass &= ok4

    # ── 测试 5：启发函数正确性 ──
    ok5a = heuristic_manhattan((0, 0), (3, 4)) == 7
    ok5b = abs(heuristic_euclidean((0, 0), (3, 4)) - 5.0) < 1e-9
    ok5 = ok5a and ok5b
    print(f"  {'✓' if ok5 else '✗'} 测试5：启发函数  "
          f"曼哈顿={(heuristic_manhattan((0,0),(3,4)))}(期望7)  "
          f"欧几里得={heuristic_euclidean((0,0),(3,4)):.4f}(期望5.0)")
    all_pass &= ok5

    # ── 测试 6：8方向路径不长于4方向 ──
    grid = make_maze(rows=15, cols=15, seed=55)
    path4dir, _ = astar(grid, (1, 1), (13, 13),
                        heuristic=heuristic_manhattan, allow_diagonal=False)
    path8dir, _ = astar(grid, (1, 1), (13, 13),
                        heuristic=heuristic_euclidean, allow_diagonal=True)
    ok6 = (not path8dir) or (not path4dir) or (len(path8dir) <= len(path4dir))
    print(f"  {'✓' if ok6 else '✗'} 测试6：8方向路径≤4方向  "
          f"8向={len(path8dir)} 4向={len(path4dir)}")
    all_pass &= ok6

    print(f"\n  {'所有测试通过 ✓' if all_pass else '部分测试失败 ✗'}")
    return all_pass


# ─────────────────────────────────────────────
# 入口
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════╗")
    print("║          A* 算法演示  (mol_evo / tests)          ║")
    print("╚══════════════════════════════════════════════════╝")

    # 单元测试
    tests_ok = run_tests()

    # Demo 演示
    demo_basic()
    demo_compare()
    demo_weight_effect()

    print("\n" + "=" * 55)
    if tests_ok:
        print("  ✓ 全部完成！")
    else:
        print("  ✗ 存在测试失败，请检查输出。")
    print("=" * 55)
