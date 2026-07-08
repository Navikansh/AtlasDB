"""
Page Manager
------------
Lowest layer of the storage stack. Treats a binary file as a sequence of
fixed-size pages and exposes read/write/allocate by page id. Nothing above
this layer is allowed to touch file offsets directly -- that's the whole
point of the layering: record_manager.py maps logical records to
(page_id, offset) and never seeks into the file itself.
"""
from __future__ import annotations

import os
import struct
import threading
from pathlib import Path

PAGE_SIZE = 4096
HEADER_MAGIC = b"ATLASPG1"
HEADER_STRUCT = struct.Struct(">8sI")  # magic, page_count


class PageManager:
    """Fixed-size page storage backed by a single file.

    File layout:
        [ header page (PAGE_SIZE bytes) ][ page 0 ][ page 1 ] ...
    The header page stores a magic number + page count so we can validate
    the file and know how many pages exist without scanning.
    """

    def __init__(self, path: str | Path, page_size: int = PAGE_SIZE):
        self.path = Path(path)
        self.page_size = page_size
        self._lock = threading.Lock()
        self._page_count = 0
        self._init_file()

    def _init_file(self) -> None:
        if self.path.exists() and self.path.stat().st_size >= self.page_size:
            with open(self.path, "rb") as f:
                header = f.read(HEADER_STRUCT.size)
                magic, count = HEADER_STRUCT.unpack(header)
                if magic != HEADER_MAGIC:
                    raise ValueError(f"{self.path} is not a valid AtlasDB page file")
                self._page_count = count
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "wb") as f:
                header = HEADER_STRUCT.pack(HEADER_MAGIC, 0)
                f.write(header.ljust(self.page_size, b"\x00"))
            self._page_count = 0

    def _write_header(self, f) -> None:
        header = HEADER_STRUCT.pack(HEADER_MAGIC, self._page_count)
        f.seek(0)
        f.write(header.ljust(self.page_size, b"\x00"))

    def allocate_page(self) -> int:
        """Append a new zeroed page and return its page id."""
        with self._lock:
            page_id = self._page_count
            with open(self.path, "r+b") as f:
                f.seek(self._offset(page_id))
                f.write(b"\x00" * self.page_size)
                self._page_count += 1
                self._write_header(f)
            return page_id

    def read_page(self, page_id: int) -> bytes:
        if page_id < 0 or page_id >= self._page_count:
            raise IndexError(f"page {page_id} out of range (0..{self._page_count - 1})")
        with self._lock, open(self.path, "rb") as f:
            f.seek(self._offset(page_id))
            return f.read(self.page_size)

    def write_page(self, page_id: int, data: bytes) -> None:
        if len(data) > self.page_size:
            raise ValueError(f"data ({len(data)}B) exceeds page size ({self.page_size}B)")
        if page_id < 0 or page_id >= self._page_count:
            raise IndexError(f"page {page_id} out of range (0..{self._page_count - 1})")
        with self._lock, open(self.path, "r+b") as f:
            f.seek(self._offset(page_id))
            f.write(data.ljust(self.page_size, b"\x00"))

    def _offset(self, page_id: int) -> int:
        return self.page_size + page_id * self.page_size  # +1 for header page

    @property
    def page_count(self) -> int:
        return self._page_count
