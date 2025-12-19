#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
createDataset.py
-----------------
Combine a hierarchical label spec (`label_hierarchy.json`) with raw annotations (`ann.csv`)
to produce a normalized dataset (`dataset.csv`).

Inputs:
1) label_hierarchy.json   # Nested dict tree, e.g. {"Root": {"Food": {"Grains": {"Wheat": {}}}}}
2) ann.csv                # Columns: id,text,leaf_labels (leaf_labels split by ';')

Output:
- dataset.csv  (UTF-8, quoted)
  columns: id, text, labels
  labels: union of every node along each leaf path (Root included); entries separated by ';'

Usage:
$ python createDataset.py --hier label_hierarchy.json --ann ann.csv --out dataset.csv
"""

import argparse
import csv
import json
import sys
import unicodedata
from collections import Counter, deque

# ----------------------------
# 小工具：全形 -> 半形，與空白正規化
# ----------------------------
FULL2HALF = str.maketrans({
    "；": ";",
    "＞": ">",
    "，": ",",
    "：": ":",
    "、": ",",
    "（": "(",
    "）": ")",
    "【": "[",
    "】": "]",
    "＼": "\\",
    "／": "/",
    "　": " ",  # 全形空白
})

def normalize_text(s: str) -> str:
    if s is None:
        return ""
    # 先用 NFKC 將全形字元（數字、英文字母等）轉半形，再套用自訂表
    s = unicodedata.normalize("NFKC", s)
    return s.strip().translate(FULL2HALF)

# ----------------------------
# 建樹：parent/children 索引
# ----------------------------
def walk_build_indices(node_name: str, subtree: dict, parent, children):
    """遞迴走訪 JSON 樹，建立 parent/children 映射。"""
    if node_name not in children:
        children[node_name] = []
    for child_name, child_sub in subtree.items():
        children[node_name].append(child_name)
        parent[child_name] = node_name
        walk_build_indices(child_name, child_sub, parent, children)

def path_to_root(node: str, parent: dict, root: str) -> list:
    """從節點上溯至 Root，回傳節點名稱列表（含 root 與原節點）。"""
    path = deque()
    cur = node
    while True:
        path.appendleft(cur)
        if cur == root:
            break
        if cur not in parent:
            raise ValueError(f"節點「{cur}」沒有父節點，請檢查階層樹是否從 Root 連通。")
        cur = parent[cur]
    return list(path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hier", required=True, help="label_hierarchy.json (中文鍵名)")
    ap.add_argument("--ann", required=True, help="ann.csv（id,text,leaf_labels）")
    ap.add_argument("--out", default="dataset.csv", help="輸出的 dataset.csv")
    args = ap.parse_args()

    # 讀取階層樹
    try:
        with open(args.hier, "r", encoding="utf-8") as f:
            H = json.load(f)
    except Exception as e:
        print(f"[ERROR] 讀取 {args.hier} 失敗：{e}", file=sys.stderr)
        sys.exit(1)

    ROOT = "Root"
    if ROOT not in H:
        print(f"[ERROR] 階層樹外層必須有 Root 作為唯一根節點。", file=sys.stderr)
        sys.exit(1)

    # 建 parent / children 查詢表
    parent = {}
    children = {}
    walk_build_indices(ROOT, H[ROOT], parent, children)

    # 讀 ann.csv 並生成 rows
    rows = []
    leaf_counts = Counter()
    try:
        with open(args.ann, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            required_cols = {"id", "text", "leaf_labels"}
            if not required_cols.issubset(set(reader.fieldnames or [])):
                raise ValueError(f"ann.csv 缺少必要欄位，需包含：{required_cols}，目前欄位：{reader.fieldnames}")

            for r in reader:
                rid = normalize_text(r.get("id", ""))
                text = r.get("text", "")
                # 全形轉半形 + 基本清洗
                text = normalize_text(text).replace("\r\n", " ").replace("\n", " ").strip()

                # 正規化 leaf_labels，支援誤用全形分號
                raw = normalize_text(r.get("leaf_labels", ""))
                # 以 ; 分割，去除空白
                leaf_names = [t.strip() for t in raw.split(";") if t.strip()]
                # 去除重複標籤，保留原順序
                seen_names = set()
                leaf_names = [n for n in leaf_names if not (n in seen_names or seen_names.add(n))]

                if not leaf_names:
                    # 若此列沒有標註，直接跳過或可視情況保留空集合
                    # 這裡選擇跳過，避免產生無效樣本
                    continue

                # 驗證節點存在
                for name in leaf_names:
                    if name != ROOT and (name not in parent and name not in children):
                        raise ValueError(f"標註的節點「{name}」不存在於階層樹中。")
                for name in leaf_names:
                    if name != ROOT and len(children.get(name, [])) == 0:
                        leaf_counts[name] += 1

                # 展開為 Root→…→leaf 的完整路徑（若標到中層也可）
                path_lists = []
                for name in leaf_names:
                    # 允許直接標 Root（雖不建議）
                    if name == ROOT:
                        path_lists.append([ROOT])
                    else:
                        # 若是非 Root 節點，需存在於 children 或 parent 索引中
                        # 找不到 parent 的情形，上面已檢查避免
                        # 這裡安全上溯
                        path_lists.append(path_to_root(name, parent, ROOT))


                # labels 欄位（節點集合：所有祖先 ∪ 葉；含 Root；以 ; 分隔）
                label_set = set()
                for p in path_lists:
                    # 確保 path 已含 Root
                    if p[0] != ROOT:
                        p = [ROOT] + p
                    label_set.update(p)

                # 轉成穩定順序（先按深度、後按字典序）
                def depth(n): 
                    return len(path_to_root(n, parent, ROOT)) if n != ROOT else 1

                labels_sorted = sorted(label_set, key=lambda x: (depth(x), x))
                labels_str = ";".join(labels_sorted)

                rows.append({
                    "id": rid,
                    "text": text,
                    "labels": labels_str
                })
    except Exception as e:
        print(f"[ERROR] 讀取/處理 {args.ann} 失敗：{e}", file=sys.stderr)
        sys.exit(1)

    # 寫出 dataset.csv（UTF-8, 逗號分隔, 雙引號包裹）
    try:
        with open(args.out, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["id","text","labels"], quoting=csv.QUOTE_ALL)
            writer.writeheader()
            for r in rows:
                writer.writerow(r)
        print(f"[OK] 已輸出 {args.out}（UTF-8, 逗號分隔, 雙引號包裹）。")
        print(f"[INFO] 總筆數：{len(rows)}")
        if leaf_counts:
            print("[INFO] Leaf node counts (descending):")
            for idx, (label, count) in enumerate(leaf_counts.most_common(), start=1):
                print(f"  {idx}. {label}: {count}")
    except Exception as e:
        print(f"[ERROR] 寫出 {args.out} 失敗：{e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
