"""
Record Manager
---------------
Sits between the Page Manager (fixed-size pages, no concept of a "record")
and the Collection Manager (knows about vectors and metadata). Its job:
take variable-length encoded records and lay them out across pages as a
flat, append-only byte log, while keeping an in-memory directory mapping
record id -> (byte_offset, length) so lookups don't require scanning.

This intentionally does NOT implement a free-space manager or compaction --
deletes are tombstoned (removed from the directory, bytes left in place).
Reclaiming that space is exactly the kind of thing a real buffer-pool /
free-list layer would do, and it's called out as future work rather than
built here (see project README: "Explicitly cut").
"""
from __future__ import annotations

import json
import threading
from pathlib import Path

from atlasdb.storage.page_manager import PageManager


class RecordManager:
    def __init__(self, page_manager: PageManager, directory_path: str | Path):
        self.pages = page_manager
        self.directory_path = Path(directory_path)
        self._lock = threading.Lock()
        # id -> (byte_offset, length)
        self._directory: dict[str, tuple[int, int]] = {}
        self._write_cursor = 0  # next free byte offset in the flat log
        self._load_directory()

    # -- directory persistence -------------------------------------------------

    def _load_directory(self) -> None:
        if self.directory_path.exists():
            with open(self.directory_path, "r") as f:
                raw = json.load(f)
            if "directory" not in raw or "write_cursor" not in raw:
                raise ValueError("corrupt record directory file")
            self._directory = {k: tuple(v) for k, v in raw["directory"].items()}
            self._write_cursor = raw["write_cursor"]
        else:
            self.directory_path.parent.mkdir(parents=True, exist_ok=True)
            self._directory = {}
            self._write_cursor = 0

    def _flush_directory(self) -> None:
        with open(self.directory_path, "w") as f:
            json.dump({"directory": self._directory, "write_cursor": self._write_cursor}, f)

    # -- record I/O -------------------------------------------------------------

    def append(self, record_id: str, data: bytes) -> None:
        with self._lock:
            offset = self._write_cursor
            self._write_bytes(offset, data)
            self._directory[record_id] = (offset, len(data))
            self._write_cursor = offset + len(data)
            self._flush_directory()

    def read(self, record_id: str) -> bytes:
        with self._lock:
            if record_id not in self._directory:
                raise KeyError(record_id)
            offset, length = self._directory[record_id]
            return self._read_bytes(offset, length)

    def delete(self, record_id: str) -> None:
        with self._lock:
            if record_id not in self._directory:
                raise KeyError(record_id)
            del self._directory[record_id]
            self._flush_directory()

    def exists(self, record_id: str) -> bool:
        return record_id in self._directory

    def ids(self) -> list[str]:
        return list(self._directory.keys())

    def __len__(self) -> int:
        return len(self._directory)

    # -- flat log <-> page translation ------------------------------------------

    def _write_bytes(self, offset: int, data: bytes) -> None:
        page_size = self.pages.page_size
        end = offset + len(data)
        while self.pages.page_count * page_size < end:
            self.pages.allocate_page()

        pos = 0
        while pos < len(data):
            byte_pos = offset + pos
            page_id = byte_pos // page_size
            page_offset = byte_pos % page_size
            chunk = data[pos: pos + (page_size - page_offset)]

            page = bytearray(self.pages.read_page(page_id))
            page[page_offset: page_offset + len(chunk)] = chunk
            self.pages.write_page(page_id, bytes(page))
            pos += len(chunk)

    def _read_bytes(self, offset: int, length: int) -> bytes:
        page_size = self.pages.page_size
        out = bytearray()
        pos = 0
        while pos < length:
            byte_pos = offset + pos
            page_id = byte_pos // page_size
            page_offset = byte_pos % page_size
            chunk_len = min(page_size - page_offset, length - pos)
            page = self.pages.read_page(page_id)
            out += page[page_offset: page_offset + chunk_len]
            pos += chunk_len
        return bytes(out)
