"""SHA256 checksums for regenerable data files."""

from __future__ import annotations

from pathlib import Path

from code.data.io_utils import read_json, sha256_file, write_json


def data_files(data_dir: Path) -> list[Path]:
    files: list[Path] = []
    for subdir in ("instances", "sft"):
        directory = data_dir / subdir
        if directory.is_dir():
            files.extend(path for path in directory.iterdir() if path.is_file())
    return sorted(files, key=lambda path: path.relative_to(data_dir).as_posix())


def compute(data_dir: Path) -> dict[str, str]:
    return {
        path.relative_to(data_dir).as_posix(): sha256_file(path)
        for path in data_files(data_dir)
    }


def write_checksums(data_dir: Path, path: Path) -> dict[str, str]:
    checksums = compute(data_dir)
    write_json(path, {"files": checksums})
    return checksums


def compare(expected_path: Path, actual: dict[str, str]) -> list[str]:
    expected = read_json(expected_path)["files"]
    problems: list[str] = []
    for name in sorted(set(expected) | set(actual)):
        if name not in expected:
            problems.append(f"unexpected file: {name}")
        elif name not in actual:
            problems.append(f"missing file: {name}")
        elif expected[name] != actual[name]:
            problems.append(
                f"hash mismatch: {name} (expected {expected[name][:12]}, "
                f"got {actual[name][:12]})"
            )
    return problems
