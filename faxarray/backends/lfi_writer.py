"""Native writer for LFI files (the container format under FA).

The format is the same one parsed by :mod:`native_lfi`:

* one *header page* at the start of the file
* one or more *index sections* (each two pages: names + (length, position)
  pairs)
* the data articles, packed at word boundaries within the remaining pages
* the file is padded so its total length is a multiple of the page size

This writer keeps the layout simple: every index section is allocated at
the *start* of the file, before any data, so we never need to inject
``****************`` page-marker entries inside the index. Extra index
sections are pointed to from the header's ``ioffi`` table so the existing
:class:`LFIFile` reader picks them up.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import struct
import sys
from typing import Iterable, List, Sequence, Tuple

import numpy as np


_ART_NAME_LEN = 16
_END_INDEX = b"**FIN D'INDEX** "
_HOLE_INDEX = b"                "
_PAGE_INDEX = b"****************"


@dataclass
class _PendingArticle:
    name: bytes  # exactly _ART_NAME_LEN bytes, latin-1 padded with spaces
    payload: bytes  # already padded to a multiple of 8 bytes


def _ensure_name(raw: str) -> bytes:
    encoded = raw.encode("latin-1")
    if len(encoded) > _ART_NAME_LEN:
        raise ValueError(f"LFI article name too long ({len(encoded)} > {_ART_NAME_LEN}): {raw!r}")
    return encoded.ljust(_ART_NAME_LEN, b" ")


def _pad_to_words(payload: bytes) -> bytes:
    extra = (-len(payload)) % 8
    if extra:
        return payload + b"\x00" * extra
    return payload


class LFIWriter:
    """Build a fresh LFI file by appending named articles, then writing.

    Parameters
    ----------
    path : str
        Output file path.
    page_size_bytes : int, default 24576
        Physical article size in bytes. Must be a positive multiple of
        16. Default mirrors what real Météo-France FA files use.
    endian : str, default ">"
        ``">"`` for big-endian (matches existing FA samples) or ``"<"``
        for little-endian.
    """

    def __init__(
        self,
        path: str,
        page_size_bytes: int = 24576,
        endian: str = ">",
    ) -> None:
        if page_size_bytes <= 0 or page_size_bytes % 16 != 0:
            raise ValueError("LFI page size must be a positive multiple of 16 bytes")
        if endian not in (">", "<"):
            raise ValueError("endian must be '>' (big) or '<' (little)")
        self.path = Path(path)
        self.page_size = int(page_size_bytes)
        self.endian = endian
        self._articles: List[_PendingArticle] = []
        self._article_names_seen: set[bytes] = set()

    @property
    def page_words(self) -> int:
        return self.page_size // 8

    @property
    def entries_per_index(self) -> int:
        return self.page_size // _ART_NAME_LEN

    def add_article(self, name: str, payload: bytes | bytearray | memoryview) -> None:
        """Queue a new article. Names are padded/truncated to 16 bytes."""

        encoded_name = _ensure_name(name)
        if encoded_name in self._article_names_seen:
            raise ValueError(f"duplicate LFI article name: {name!r}")
        self._article_names_seen.add(encoded_name)
        self._articles.append(_PendingArticle(encoded_name, _pad_to_words(bytes(payload))))

    def write(self) -> None:
        """Materialise the file at :attr:`path`."""

        n_articles = len(self._articles)
        if n_articles == 0:
            raise ValueError("LFIWriter requires at least one article")

        # Need one entry slot per article + one for the **FIN D'INDEX** marker.
        n_indexes = max(1, (n_articles + 1 + self.entries_per_index - 1) // self.entries_per_index)
        ioffi_capacity = self.page_words - 22
        n_extra = n_indexes - 1
        if n_extra > ioffi_capacity:
            raise ValueError(
                f"too many articles for page size {self.page_size}: would need "
                f"{n_extra} extra index sections but header has room for {ioffi_capacity}"
            )

        # Compute byte positions.
        first_data_page = 1 + 2 * n_indexes  # 0-indexed
        first_data_byte = first_data_page * self.page_size

        # 1-indexed word positions for each article.
        cursor_words = first_data_byte // 8 + 1
        positions: List[int] = []
        for article in self._articles:
            positions.append(cursor_words)
            cursor_words += len(article.payload) // 8

        total_data_bytes = cursor_words * 8 - first_data_byte
        total_pages = first_data_page + (total_data_bytes + self.page_size - 1) // self.page_size
        if total_pages == first_data_page and n_articles > 0:
            # Defensive: ensure we still have at least one data page.
            total_pages = first_data_page + 1
        file_size = total_pages * self.page_size

        # Build the header page.
        header = bytearray(self.page_size)
        now = datetime.now()
        date_value = now.year * 10000 + now.month * 100 + now.day
        time_value = now.hour * 10000 + now.minute * 100 + now.second
        ilnal = min(len(a.payload) // 8 for a in self._articles)
        ilxal = max(len(a.payload) // 8 for a in self._articles)
        iltal = sum(len(a.payload) // 8 for a in self._articles)
        header_words = [
            self.page_words,        # ilpar (word 0)
            _ART_NAME_LEN,          # ilmna (1)
            0,                      # ifeam (2), clean close
            22,                     # illdo (3), header struct length
            total_pages,            # inaph (4)
            n_articles,             # inalo (5)
            ilnal,                  # ilnal (6)
            ilxal,                  # ilxal (7)
            iltal,                  # iltal (8)
            0,                      # inres (9)
            0,                      # inrec (10)
            0,                      # inrel (11)
            self.entries_per_index, # ixapi (12)
            date_value,             # idcre (13)
            time_value,             # ihcre (14)
            date_value,             # iddmg (15)
            time_value,             # ihdmg (16)
            date_value,             # idmng (17)
            time_value,             # ihmng (18)
            1,                      # inpir (19)
            0,                      # intru (20)
            total_pages,            # iaxpd (21)
        ]
        struct.pack_into(f"{self.endian}22q", header, 0, *header_words)

        # Extra index pointers live at the END of the ioffi[] tail, growing
        # backward: ioffi[ioffib-k] = (2k+2) for k=1..n_indexes-1.
        for k in range(1, n_indexes):
            extra_page_one_indexed = 2 * k + 2  # 1-indexed page where index k starts
            slot_word = self.page_words - k     # i.e. word[ioffib-k] = word[(page_words-22)-k+22]
            struct.pack_into(f"{self.endian}q", header, slot_word * 8, extra_page_one_indexed)

        # Build each index section.
        index_blobs: List[bytes] = []
        cursor = 0
        for section in range(n_indexes):
            names = bytearray(self.page_size)
            pairs = bytearray(self.page_size)
            for slot in range(self.entries_per_index):
                if cursor < n_articles:
                    article = self._articles[cursor]
                    names[slot * _ART_NAME_LEN : (slot + 1) * _ART_NAME_LEN] = article.name
                    struct.pack_into(
                        f"{self.endian}qq",
                        pairs,
                        slot * 16,
                        len(article.payload) // 8,
                        positions[cursor],
                    )
                    cursor += 1
                elif cursor == n_articles:
                    names[slot * _ART_NAME_LEN : (slot + 1) * _ART_NAME_LEN] = _END_INDEX
                    cursor += 1
                else:
                    # Extra slots get the end marker too. The reader stops at
                    # the first end marker but tolerates trailing ones.
                    names[slot * _ART_NAME_LEN : (slot + 1) * _ART_NAME_LEN] = _END_INDEX
            index_blobs.append(bytes(names))
            index_blobs.append(bytes(pairs))

        # Materialise the file in one buffer for simplicity. Files larger
        # than memory are not realistic for this writer (it is intended
        # for templates/conversions, not multi-GB ECMWF outputs).
        buffer = bytearray(file_size)
        buffer[: self.page_size] = bytes(header)
        cursor_byte = self.page_size
        for blob in index_blobs:
            buffer[cursor_byte : cursor_byte + self.page_size] = blob
            cursor_byte += self.page_size

        for article, position in zip(self._articles, positions):
            offset = (position - 1) * 8
            buffer[offset : offset + len(article.payload)] = article.payload

        self.path.write_bytes(bytes(buffer))


def write_lfi(
    path: str,
    articles: Sequence[Tuple[str, bytes]],
    page_size_bytes: int = 24576,
    endian: str = ">",
) -> None:
    """Convenience wrapper around :class:`LFIWriter`."""

    writer = LFIWriter(path, page_size_bytes=page_size_bytes, endian=endian)
    for name, payload in articles:
        writer.add_article(name, payload)
    writer.write()
