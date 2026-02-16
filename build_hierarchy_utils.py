# build_hierarchy_utils.py
# Utilities to parse label_hierarchy.json and build:
# - label2id / id2label (global label indexing)
# - edges_parent_child (for M3 path hinge)
# - ancestors map (for M4 closure)
# - levels (depth per node), paths (root->...->node), level_sizes
# - helper to expand labels with ancestors and to build multi-hot Y
#
# This module aligns with thesis §3.3–§3.6: it supplies the structural
# information needed by the classifier losses (M3) and closure (M4).

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional, Iterable, Set, Any
from collections import defaultdict, deque
import json
import os

@dataclass
class HierarchyData:
    label2id: Dict[str, int]
    id2label: Dict[int, str]
    edges_parent_child: List[Tuple[int, int]]
    ancestors: Dict[int, List[int]]          # child_id -> sorted unique list of ancestor ids
    parents: Dict[int, List[int]]            # child_id -> parent ids (usually 1, but allow multi-parent)
    children: Dict[int, List[int]]           # parent_id -> child ids
    levels: Dict[int, int]                   # node depth: root=1
    paths: Dict[int, List[int]]              # node_id -> list of ids [root,...,node]
    path_strings: Dict[int, str]             # node_id -> "A > B > C"
    level_sizes: List[int]                   # [L1, L2, ..., L^D]
    num_labels: int

    def to_json(self, fp: str) -> None:
        obj = asdict(self)
        # keys are str/int; convert int keys to str for JSON safety in dicts
        obj["id2label"] = {str(k): v for k, v in self.id2label.items()}
        obj["ancestors"] = {str(k): v for k, v in self.ancestors.items()}
        obj["parents"] = {str(k): v for k, v in self.parents.items()}
        obj["children"] = {str(k): v for k, v in self.children.items()}
        obj["levels"] = {str(k): v for k, v in self.levels.items()}
        obj["paths"] = {str(k): v for k, v in self.paths.items()}
        obj["path_strings"] = {str(k): v for k, v in self.path_strings.items()}
        with open(fp, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)

    @staticmethod
    def from_json(fp: str) -> "HierarchyData":
        with open(fp, "r", encoding="utf-8") as f:
            obj = json.load(f)
        id2label = {int(k): v for k, v in obj["id2label"].items()}
        ancestors = {int(k): list(map(int, v)) for k, v in obj["ancestors"].items()}
        parents = {int(k): list(map(int, v)) for k, v in obj["parents"].items()}
        children = {int(k): list(map(int, v)) for k, v in obj["children"].items()}
        levels = {int(k): int(v) for k, v in obj["levels"].items()}
        paths = {int(k): list(map(int, v)) for k, v in obj["paths"].items()}
        path_strings = {int(k): v for k, v in obj["path_strings"].items()}
        return HierarchyData(
            label2id=obj["label2id"],
            id2label=id2label,
            edges_parent_child=[tuple(x) for x in obj["edges_parent_child"]],
            ancestors=ancestors,
            parents=parents,
            children=children,
            levels=levels,
            paths=paths,
            path_strings=path_strings,
            level_sizes=list(map(int, obj["level_sizes"])),
            num_labels=int(obj["num_labels"]),
        )


# -----------------------------
# Parsing helpers
# -----------------------------
def _is_leaf_container(x: Any) -> bool:
    """Return True if x is an iterable of leaf names, e.g., ["燕麥片", "糙米"]."""
    if isinstance(x, list):
        return all(isinstance(e, str) for e in x)
    return False

def _ensure_node(name: str, label2id: Dict[str, int], id2label: Dict[int, str]) -> int:
    if name not in label2id:
        nid = len(label2id)
        label2id[name] = nid
        id2label[nid] = name
    return label2id[name]

def _add_edge(p_id: int, c_id: int,
              parents: Dict[int, List[int]],
              children: Dict[int, List[int]],
              edges: List[Tuple[int, int]]) -> None:
    parents[c_id].append(p_id)
    children[p_id].append(c_id)
    edges.append((p_id, c_id))

def _dfs_build(root_name: str,
               node_payload: Any,
               label2id: Dict[str, int],
               id2label: Dict[int, str],
               parents: Dict[int, List[int]],
               children: Dict[int, List[int]],
               edges: List[Tuple[int, int]],
               ) -> None:
    """Recursively traverse a JSON subtree rooted at `root_name`."""
    root_id = _ensure_node(root_name, label2id, id2label)

    if node_payload is None:
        return
    # Case A: dict children
    if isinstance(node_payload, dict):
        for child_name, sub in node_payload.items():
            c_id = _ensure_node(child_name, label2id, id2label)
            _add_edge(root_id, c_id, parents, children, edges)
            _dfs_build(child_name, sub, label2id, id2label, parents, children, edges)
        return
    # Case B: list of leaf strings
    if _is_leaf_container(node_payload):
        for leaf_name in node_payload:
            c_id = _ensure_node(leaf_name, label2id, id2label)
            _add_edge(root_id, c_id, parents, children, edges)
        return
    # Case C: list of dict-like children (rare)
    if isinstance(node_payload, list):
        for elem in node_payload:
            # elem may be str (handled above), or dict {child: sub}
            if isinstance(elem, dict):
                for child_name, sub in elem.items():
                    c_id = _ensure_node(child_name, label2id, id2label)
                    _add_edge(root_id, c_id, parents, children, edges)
                    _dfs_build(child_name, sub, label2id, id2label, parents, children, edges)
            elif isinstance(elem, str):
                c_id = _ensure_node(elem, label2id, id2label)
                _add_edge(root_id, c_id, parents, children, edges)
        return
    # Fallback: unrecognized structure; ignore silently.

def parse_label_hierarchy(hierarchy_json: Dict[str, Any],
                          path_sep: str = " > ") -> HierarchyData:
    """
    Parse a potentially multi-root hierarchy JSON into a flat index + graph.
    Supports:
      { "食品": { "全穀雜糧類": ["燕麥片", ...], "豆魚蛋肉類": {...} }, "飲料": {...} }
    """
    label2id: Dict[str, int] = {}
    id2label: Dict[int, str] = {}
    parents: Dict[int, List[int]] = defaultdict(list)
    children: Dict[int, List[int]] = defaultdict(list)
    edges: List[Tuple[int, int]] = []

    # Traverse roots (forest allowed)
    for root_name, sub in hierarchy_json.items():
        _dfs_build(root_name, sub, label2id, id2label, parents, children, edges)

    # Compute levels (BFS from roots)
    indeg = {i: 0 for i in label2id.values()}
    for p, c in edges:
        indeg[c] += 1
    roots = [nid for nid, d in indeg.items() if d == 0]
    levels: Dict[int, int] = {}
    paths: Dict[int, List[int]] = {}

    q = deque()
    for r in roots:
        levels[r] = 1
        paths[r] = [r]
        q.append(r)

    while q:
        u = q.popleft()
        for v in children.get(u, []):
            # if multi-parent, set level as min(parent_level)+1
            lv = min(levels.get(p, 10**9) for p in parents.get(v, [u])) + 1
            if v not in levels or lv < levels[v]:
                levels[v] = lv
                # choose one parent path (first parent) for representative path
                parent_for_path = parents[v][0] if parents.get(v) else u
                paths[v] = paths[parent_for_path] + [v]
            q.append(v)

    # Compute ancestors by climbing parents (DAG-safe)
    ancestors: Dict[int, List[int]] = {}
    for node in label2id.values():
        seen: Set[int] = set()
        stack = list(parents.get(node, []))
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            stack.extend(parents.get(cur, []))
        ancestors[node] = sorted(seen)

    # Build path strings
    path_strings: Dict[int, str] = {
        nid: path_sep.join(id2label[i] for i in paths.get(nid, [nid]))
        for nid in label2id.values()
    }

    # Level sizes
    max_lvl = max(levels.values()) if levels else 0
    level_sizes = [0] * max_lvl
    for nid, lv in levels.items():
        level_sizes[lv - 1] += 1

    return HierarchyData(
        label2id=label2id,
        id2label=id2label,
        edges_parent_child=edges,
        ancestors=ancestors,
        parents=parents,
        children=children,
        levels=levels,
        paths=paths,
        path_strings=path_strings,
        level_sizes=level_sizes,
        num_labels=len(label2id),
    )

# -----------------------------
# Public APIs
# -----------------------------
def load_hierarchy_from_file(json_path: str) -> HierarchyData:
    with open(json_path, "r", encoding="utf-8") as f:
        hjson = json.load(f)
    return parse_label_hierarchy(hjson)

def expand_labels_with_ancestors(label_names: Iterable[str], label2id: Dict[str, int],
                                 ancestors: Dict[int, List[int]]) -> List[int]:
    """
    Given a set/list of label names (possibly leaf-only), return ids including all ancestors.
    """
    ids: Set[int] = set()
    for name in label_names:
        if name not in label2id:
            continue
        nid = label2id[name]
        ids.add(nid)
        for a in ancestors.get(nid, []):
            ids.add(a)
    return sorted(ids)

def build_multi_hot_Y(
    samples_labels: List[Iterable[str]],
    label2id: Dict[str, int],
    ancestors: Dict[int, List[int]],
    add_ancestors: bool = True,
) -> List[List[int]]:
    """
    Build a dense multi-hot matrix Y (as list-of-lists 0/1) for N samples × L labels.
    If add_ancestors=True, ancestors are auto-completed per sample (recommended for training).
    """
    L = len(label2id)
    Y = [[0] * L for _ in range(len(samples_labels))]
    for i, label_names in enumerate(samples_labels):
        if add_ancestors:
            ids = expand_labels_with_ancestors(label_names, label2id, ancestors)
        else:
            ids = [label2id[n] for n in label_names if n in label2id]
        for j in ids:
            Y[i][j] = 1
    return Y

def make_level_slices(levels: Dict[int, int]) -> List[List[int]]:
    """
    Return a list of lists: ids per level (1-based), useful to init M3 local heads.
    """
    max_lvl = max(levels.values()) if levels else 0
    per_level: List[List[int]] = [[] for _ in range(max_lvl)]
    for nid, lv in levels.items():
        per_level[lv - 1].append(nid)
    # stabilize order
    for l in range(max_lvl):
        per_level[l].sort()
    return per_level

# -----------------------------
# Minimal smoke test
# -----------------------------
if __name__ == "__main__":
    # Example JSON structure (multi-form supported)
    example = {
        "食品": {
            "全穀雜糧類": ["燕麥片", "糙米"],
            "豆魚蛋肉類": {"牛肉": None, "雞蛋": None},
            "醬料類": ["咖哩醬", "青醬"]
        },
        "飲料": {
            "茶飲": ["綠茶", "紅茶"],
            "咖啡": ["美式", "拿鐵"]
        }
    }
    hd = parse_label_hierarchy(example)
    print("num_labels:", hd.num_labels)
    print("level_sizes:", hd.level_sizes)
    print("edges (p->c):", hd.edges_parent_child[:8], " ...")
    print("sample path:", next(iter(hd.path_strings.values())))

    # Expand labels with ancestors
    labs = ["牛肉", "青醬"]
    ids = expand_labels_with_ancestors(labs, hd.label2id, hd.ancestors)
    print("expanded ids:", ids, "->", [hd.id2label[i] for i in ids])

    # Build Y for two samples
    Y = build_multi_hot_Y([["燕麥片"], ["美式", "拿鐵"]], hd.label2id, hd.ancestors, add_ancestors=True)
    print("Y[0] sum / Y[1] sum:", sum(Y[0]), sum(Y[1]))
