# coding=utf-8
"""Download and export the Web of Science dataset for this project."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.request import Request, urlopen

try:
    import requests
except ImportError:  # pragma: no cover - optional dependency
    requests = None

try:
    import datasets as hf_datasets
except ImportError:  # pragma: no cover - optional dependency
    hf_datasets = None


_CITATION = """\
@inproceedings{kowsari2017HDLTex,
title={HDLTex: Hierarchical Deep Learning for Text Classification},
author={Kowsari, Kamran and Brown, Donald E and Heidarysafa, Mojtaba and Jafari Meimandi, Kiana and and Gerber, Matthew S and Barnes, Laura E},
booktitle={Machine Learning and Applications (ICMLA), 2017 16th IEEE International Conference on},
year={2017},
organization={IEEE}
}
"""

_DESCRIPTION = """\
The Web Of Science (WOS) dataset is a collection of published paper abstracts.
It is distributed as three subsets: WOS-46985, WOS-11967 and WOS-5736.
"""

_DATA_URL = (
    "https://data.mendeley.com/public-files/datasets/9rw3vkcfy4/files/"
    "c9ea673d-5542-44c0-ab7b-f1311f7d61df/file_downloaded"
)
_DOWNLOAD_HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "*/*",
}
_CONFIG_NAMES = ("WOS5736", "WOS11967", "WOS46985")
_XLSX_NS = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _normalize_label_text(text: str) -> str:
    return " ".join((text or "").strip().split())


def _read_text_lines(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8") as f:
        return [line.rstrip("\r\n") for line in f]


def _read_int_lines(path: Path) -> List[int]:
    with path.open("r", encoding="utf-8") as f:
        return [int(line.strip()) for line in f if line.strip()]


def _download_with_requests(url: str, output_path: Path) -> None:
    assert requests is not None
    with requests.get(url, stream=True, timeout=120, headers=_DOWNLOAD_HEADERS) as r:
        r.raise_for_status()
        _ensure_parent(output_path)
        with output_path.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)


def _download_with_urllib(url: str, output_path: Path) -> None:
    req = Request(url, headers=_DOWNLOAD_HEADERS)
    with urlopen(req, timeout=120) as r:
        _ensure_parent(output_path)
        with output_path.open("wb") as f:
            shutil.copyfileobj(r, f)


def download_archive(
    output_dir: Path,
    force: bool = False,
    archive_path: Optional[Path] = None,
) -> Path:
    if archive_path is not None:
        archive_path = archive_path.expanduser().resolve()
        if not archive_path.exists():
            raise FileNotFoundError(f"Archive not found: {archive_path}")
        return archive_path

    target = output_dir / "raw" / "web_of_science.zip"
    if target.exists() and not force:
        print(f"[download] Reusing existing archive: {target}")
        return target

    print(f"[download] Downloading WOS archive to {target}")
    if requests is not None:
        _download_with_requests(_DATA_URL, target)
    else:
        _download_with_urllib(_DATA_URL, target)
    return target


def extract_archive(archive_path: Path, output_dir: Path, force: bool = False) -> Path:
    extract_dir = output_dir / "extracted"
    expected = extract_dir / "WOS46985" / "X.txt"
    if expected.exists() and not force:
        print(f"[extract] Reusing extracted files: {extract_dir}")
        return extract_dir

    if force and extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)

    print(f"[extract] Extracting {archive_path} -> {extract_dir}")
    with zipfile.ZipFile(archive_path) as zf:
        zf.extractall(extract_dir)
    return extract_dir


def _column_ref_to_index(cell_ref: str) -> int:
    letters = "".join(ch for ch in cell_ref if ch.isalpha()).upper()
    index = 0
    for ch in letters:
        index = index * 26 + (ord(ch) - ord("A") + 1)
    return max(index - 1, 0)


def _load_shared_strings(xlsx_zip: zipfile.ZipFile) -> List[str]:
    if "xl/sharedStrings.xml" not in xlsx_zip.namelist():
        return []

    root = ET.fromstring(xlsx_zip.read("xl/sharedStrings.xml"))
    shared: List[str] = []
    for si in root.findall("a:si", _XLSX_NS):
        parts = []
        for text_node in si.iterfind(".//a:t", _XLSX_NS):
            parts.append(text_node.text or "")
        shared.append("".join(parts))
    return shared


def _read_sheet_rows(xlsx_path: Path) -> List[List[str]]:
    with zipfile.ZipFile(xlsx_path) as xlsx_zip:
        shared_strings = _load_shared_strings(xlsx_zip)
        sheet_xml = ET.fromstring(xlsx_zip.read("xl/worksheets/sheet1.xml"))

    rows: List[List[str]] = []
    for row in sheet_xml.findall(".//a:sheetData/a:row", _XLSX_NS):
        values: List[str] = []
        for cell in row.findall("a:c", _XLSX_NS):
            idx = _column_ref_to_index(cell.attrib.get("r", "A1"))
            while len(values) <= idx:
                values.append("")

            cell_type = cell.attrib.get("t")
            value_node = cell.find("a:v", _XLSX_NS)
            if value_node is None:
                values[idx] = ""
            elif cell_type == "s":
                values[idx] = shared_strings[int(value_node.text)]
            else:
                values[idx] = value_node.text or ""
        rows.append(values)
    return rows


def read_metadata(metadata_path: Path) -> List[Dict[str, str]]:
    rows = _read_sheet_rows(metadata_path)
    if not rows:
        return []

    header = [cell.strip() for cell in rows[0]]
    records: List[Dict[str, str]] = []
    for row in rows[1:]:
        if not any(cell.strip() for cell in row):
            continue
        padded = row + [""] * max(0, len(header) - len(row))
        records.append({header[i]: padded[i] for i in range(len(header))})
    return records


def build_label_maps(
    metadata_rows: Sequence[Dict[str, str]]
) -> Tuple[
    Dict[int, str],
    Dict[int, str],
    Dict[int, str],
    List[Tuple[int, Counter]],
]:
    y1_to_domain_counts: Dict[int, Counter] = defaultdict(Counter)
    y_to_domain_counts: Dict[int, Counter] = defaultdict(Counter)
    y_to_area_counts: Dict[int, Counter] = defaultdict(Counter)

    for row in metadata_rows:
        y1 = int(row["Y1"])
        y = int(row["Y"])
        domain = _normalize_label_text(row["Domain"])
        area = _normalize_label_text(row["area"])
        y1_to_domain_counts[y1][domain] += 1
        y_to_domain_counts[y][domain] += 1
        y_to_area_counts[y][area] += 1

    y1_to_domain = {
        key: counter.most_common(1)[0][0]
        for key, counter in y1_to_domain_counts.items()
    }
    y_to_domain = {
        key: counter.most_common(1)[0][0]
        for key, counter in y_to_domain_counts.items()
    }
    y_to_area = {
        key: counter.most_common(1)[0][0]
        for key, counter in y_to_area_counts.items()
    }
    ambiguous_y = [
        (key, counter)
        for key, counter in sorted(y_to_area_counts.items())
        if len(counter) > 1
    ]
    return y1_to_domain, y_to_domain, y_to_area, ambiguous_y


def export_project_format(
    extracted_dir: Path,
    output_dir: Path,
    config_name: str,
    y1_to_domain: Dict[int, str],
    y_to_domain: Dict[int, str],
    y_to_area: Dict[int, str],
) -> Tuple[Path, Path]:
    if config_name not in _CONFIG_NAMES:
        raise ValueError(f"Unknown config: {config_name}")

    src_dir = extracted_dir / config_name
    x_path = src_dir / "X.txt"
    y_path = src_dir / "Y.txt"
    y1_path = src_dir / "YL1.txt"
    y2_path = src_dir / "YL2.txt"
    if not all(path.exists() for path in (x_path, y_path, y1_path, y2_path)):
        raise FileNotFoundError(f"Incomplete extracted data for {config_name}: {src_dir}")

    texts = _read_text_lines(x_path)
    labels = _read_int_lines(y_path)
    level1_ids = _read_int_lines(y1_path)
    level2_ids = _read_int_lines(y2_path)
    total = len(texts)
    if not (len(labels) == len(level1_ids) == len(level2_ids) == total):
        raise ValueError(f"Length mismatch while exporting {config_name}")

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{config_name}.csv"
    hierarchy_path = output_dir / f"{config_name}_hierarchy.json"
    hierarchy: Dict[str, Dict[str, Dict[str, dict]]] = {"Root": {}}

    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "text",
                "labels",
                "domain",
                "area",
                "label_id",
                "label_level_1_id",
                "label_level_2_id",
            ],
        )
        writer.writeheader()

        for text, label_id, level1_id, level2_id in zip(texts, labels, level1_ids, level2_ids):
            domain = y_to_domain.get(label_id) or y1_to_domain.get(level1_id) or f"Domain_{level1_id}"
            area = y_to_area.get(label_id) or f"Label_{label_id}"
            hierarchy["Root"].setdefault(domain, {})
            hierarchy["Root"][domain].setdefault(area, {})
            writer.writerow(
                {
                    "text": text,
                    "labels": f"Root;{domain};{area}",
                    "domain": domain,
                    "area": area,
                    "label_id": label_id,
                    "label_level_1_id": level1_id,
                    "label_level_2_id": level2_id,
                }
            )

    with hierarchy_path.open("w", encoding="utf-8") as f:
        json.dump(hierarchy, f, ensure_ascii=False, indent=2)

    print(f"[export] Wrote {csv_path}")
    print(f"[export] Wrote {hierarchy_path}")
    return csv_path, hierarchy_path


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download Web of Science into this project.")
    parser.add_argument(
        "--output-dir",
        default="dataset/web_of_science",
        help="Directory for raw and exported WOS files.",
    )
    parser.add_argument(
        "--config",
        choices=[*_CONFIG_NAMES, "all"],
        default="WOS46985",
        help="Subset to export into project CSV/JSON format.",
    )
    parser.add_argument(
        "--archive-path",
        default=None,
        help="Reuse an existing downloaded zip instead of downloading again.",
    )
    parser.add_argument(
        "--raw-only",
        action="store_true",
        help="Only download and extract the original archive.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Redownload and re-extract even if files already exist.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    archive_path = Path(args.archive_path) if args.archive_path else None

    archive = download_archive(output_dir, force=args.force, archive_path=archive_path)
    extracted_dir = extract_archive(archive, output_dir, force=args.force)

    if args.raw_only:
        print(f"[done] Raw WOS files are available in {extracted_dir}")
        return

    metadata_path = extracted_dir / "Meta-data" / "Data.xlsx"
    metadata_rows = read_metadata(metadata_path)
    y1_to_domain, y_to_domain, y_to_area, ambiguous_y = build_label_maps(metadata_rows)

    if ambiguous_y:
        print("[meta] Ambiguous Y -> area mappings detected; using the most frequent area name.")
        for y, counter in ambiguous_y:
            choices = ", ".join(f"{name} ({count})" for name, count in counter.most_common())
            print(f"  Y={y}: {choices}")

    configs = list(_CONFIG_NAMES) if args.config == "all" else [args.config]
    for config_name in configs:
        export_project_format(
            extracted_dir=extracted_dir,
            output_dir=output_dir,
            config_name=config_name,
            y1_to_domain=y1_to_domain,
            y_to_domain=y_to_domain,
            y_to_area=y_to_area,
        )

    print(f"[done] Project-ready files are available in {output_dir}")


if hf_datasets is not None:
    class WebOfScienceConfig(hf_datasets.BuilderConfig):
        """BuilderConfig for WebOfScience."""

        def __init__(self, **kwargs):
            super().__init__(version=hf_datasets.Version("6.0.0", ""), **kwargs)


    class WebOfScience(hf_datasets.GeneratorBasedBuilder):
        """Hugging Face datasets builder for Web of Science."""

        BUILDER_CONFIGS = [
            WebOfScienceConfig(
                name="WOS5736",
                description=(
                    "Web of Science Dataset WOS-5736: "
                    "5,736 documents with 11 categories including 3 parent categories."
                ),
            ),
            WebOfScienceConfig(
                name="WOS11967",
                description=(
                    "Web of Science Dataset WOS-11967: "
                    "11,967 documents with 35 categories including 7 parent categories."
                ),
            ),
            WebOfScienceConfig(
                name="WOS46985",
                description=(
                    "Web of Science Dataset WOS-46985: "
                    "46,985 documents with 134 categories including 7 parent categories."
                ),
            ),
        ]

        def _info(self):
            return hf_datasets.DatasetInfo(
                description=_DESCRIPTION + self.config.description,
                features=hf_datasets.Features(
                    {
                        "input_data": hf_datasets.Value("string"),
                        "label": hf_datasets.Value("int32"),
                        "label_level_1": hf_datasets.Value("int32"),
                        "label_level_2": hf_datasets.Value("int32"),
                    }
                ),
                supervised_keys=None,
                homepage="https://data.mendeley.com/datasets/9rw3vkcfy4/6",
                citation=_CITATION,
            )

        def _split_generators(self, dl_manager):
            dl_path = dl_manager.download_and_extract(_DATA_URL)
            return [
                hf_datasets.SplitGenerator(
                    name=hf_datasets.Split.TRAIN,
                    gen_kwargs={
                        "input_file": str(Path(dl_path) / self.config.name / "X.txt"),
                        "label_file": str(Path(dl_path) / self.config.name / "Y.txt"),
                        "label_level_1_file": str(Path(dl_path) / self.config.name / "YL1.txt"),
                        "label_level_2_file": str(Path(dl_path) / self.config.name / "YL2.txt"),
                    },
                )
            ]

        def _generate_examples(
            self,
            input_file: str,
            label_file: str,
            label_level_1_file: str,
            label_level_2_file: str,
        ):
            input_data = _read_text_lines(Path(input_file))
            label_data = _read_int_lines(Path(label_file))
            label_level_1_data = _read_int_lines(Path(label_level_1_file))
            label_level_2_data = _read_int_lines(Path(label_level_2_file))

            for i in range(len(input_data)):
                yield i, {
                    "input_data": input_data[i],
                    "label": label_data[i],
                    "label_level_1": label_level_1_data[i],
                    "label_level_2": label_level_2_data[i],
                }


if __name__ == "__main__":
    main()
