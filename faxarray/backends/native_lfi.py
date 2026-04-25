"""Minimal native reader for pure LFI containers.

The implementation follows the rootpack ``ifsaux/lfi_alt/lfi_alts.c`` layout:
an LFI file is an indexed container of 8-byte words. FA files store their
header records and field records as LFI articles.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import struct
from typing import Dict, Iterable, List, Optional

import numpy as np


_ART_NAME_LEN = 16
_END_INDEX = b"**FIN D'INDEX** "
_HOLE_INDEX = b"                "
_PAGE_INDEX = b"****************"
_SPECIAL_NAMES = {_END_INDEX, _HOLE_INDEX, _PAGE_INDEX}


@dataclass(frozen=True)
class LFIArticle:
    """Article descriptor from the LFI index."""

    name: str
    length_words: int
    position_words: int

    @property
    def length_bytes(self) -> int:
        return self.length_words * 8

    @property
    def offset_bytes(self) -> int:
        return (self.position_words - 1) * 8


class LFIFormatError(ValueError):
    """Raised when a file does not look like a supported pure LFI file."""


class LFIFile:
    """Read and update articles in a pure LFI file."""

    def __init__(self, path: str):
        self.path = Path(path)
        self.endian = self._detect_endian()
        self.word_dtype = np.dtype(f"{self.endian}i8")
        self.float64_dtype = np.dtype(f"{self.endian}f8")
        self.float32_dtype = np.dtype(f"{self.endian}f4")
        self.header_words: List[int] = []
        self.article_size_bytes = 0
        self.articles: List[LFIArticle] = []
        self._article_map: Dict[str, LFIArticle] = {}
        self._read_index()

    def _detect_endian(self) -> str:
        with self.path.open("rb") as fh:
            first16 = fh.read(16)
        if len(first16) != 16:
            raise LFIFormatError(f"{self.path} is too small to be an LFI file")

        second_native = struct.unpack("@q", first16[8:16])[0]
        if 0 < second_native <= 128:
            return "<"

        second_big = struct.unpack(">q", first16[8:16])[0]
        second_little = struct.unpack("<q", first16[8:16])[0]
        if 0 < second_big <= 128:
            return ">"
        if 0 < second_little <= 128:
            return "<"
        raise LFIFormatError(f"{self.path} does not look like a pure LFI file")

    def _unpack_words(self, data: bytes) -> List[int]:
        if len(data) % 8:
            raise LFIFormatError("LFI word data length is not a multiple of 8")
        return list(struct.unpack(f"{self.endian}{len(data) // 8}q", data))

    def _pack_words(self, words: Iterable[int]) -> bytes:
        values = list(words)
        return struct.pack(f"{self.endian}{len(values)}q", *values)

    def _read_words_at(self, offset: int, count: int) -> List[int]:
        with self.path.open("rb") as fh:
            fh.seek(offset)
            data = fh.read(count * 8)
        if len(data) != count * 8:
            raise LFIFormatError(f"short read at byte offset {offset}")
        return self._unpack_words(data)

    def _read_index(self) -> None:
        first_word = self._read_words_at(0, 1)[0]
        self.article_size_bytes = first_word * 8
        if self.article_size_bytes <= 0 or self.article_size_bytes % 8:
            raise LFIFormatError("invalid LFI physical article size")

        header_count = self.article_size_bytes // 8
        self.header_words = self._read_words_at(0, header_count)
        if len(self.header_words) < 23:
            raise LFIFormatError("invalid LFI header")

        illdo = self.header_words[3]
        if illdo <= 0 or illdo >= header_count:
            raise LFIFormatError("invalid LFI header length")

        ioffi = self.header_words[22:header_count]
        ioffib = header_count - illdo
        n_extra_indexes = 0
        while (
            n_extra_indexes < len(ioffi)
            and ioffib - 1 - n_extra_indexes >= 0
            and ioffi[ioffib - 1 - n_extra_indexes] != 0
        ):
            n_extra_indexes += 1

        articles: List[LFIArticle] = []
        entries_per_index = self.article_size_bytes // _ART_NAME_LEN
        words_per_index = self.article_size_bytes // 8

        with self.path.open("rb") as fh:
            for index_number in range(n_extra_indexes + 1):
                if index_number == 0:
                    offset = self.article_size_bytes
                else:
                    offset = (ioffi[ioffib - index_number] - 1) * self.article_size_bytes

                fh.seek(offset)
                names = fh.read(self.article_size_bytes)
                pair_bytes = fh.read(self.article_size_bytes)
                if len(names) != self.article_size_bytes or len(pair_bytes) != self.article_size_bytes:
                    raise LFIFormatError("short read while reading LFI index")

                pairs = self._unpack_words(pair_bytes)
                if len(pairs) != words_per_index:
                    raise LFIFormatError("invalid LFI index pair section")

                for item in range(entries_per_index):
                    raw_name = names[item * _ART_NAME_LEN : (item + 1) * _ART_NAME_LEN]
                    if raw_name == _END_INDEX:
                        self.articles = articles
                        self._article_map = {a.name: a for a in articles}
                        return
                    if raw_name in _SPECIAL_NAMES:
                        continue

                    name = raw_name.decode("latin-1").rstrip()
                    length = pairs[item * 2]
                    position = pairs[item * 2 + 1]
                    if length > 0 and position > 0:
                        articles.append(LFIArticle(name, int(length), int(position)))

        self.articles = articles
        self._article_map = {a.name: a for a in articles}

    def get_article(self, name: str) -> LFIArticle:
        try:
            return self._article_map[name]
        except KeyError as exc:
            raise KeyError(f"LFI article not found: {name}") from exc

    def read_article_bytes(self, name: str, max_words: Optional[int] = None) -> bytes:
        article = self.get_article(name)
        words = article.length_words if max_words is None else min(max_words, article.length_words)
        with self.path.open("rb") as fh:
            fh.seek(article.offset_bytes)
            data = fh.read(words * 8)
        if len(data) != words * 8:
            raise LFIFormatError(f"short read while reading article {name}")
        return data

    def read_article_words(self, name: str, max_words: Optional[int] = None) -> List[int]:
        return self._unpack_words(self.read_article_bytes(name, max_words=max_words))

    def write_article_bytes(self, name: str, data: bytes) -> None:
        article = self.get_article(name)
        if len(data) != article.length_bytes:
            raise ValueError(
                f"replacement for {name} is {len(data)} bytes, expected {article.length_bytes}"
            )
        with self.path.open("r+b") as fh:
            fh.seek(article.offset_bytes)
            fh.write(data)

    def list_fa_fields(self, include_misc: bool = False) -> List[str]:
        """Return FA field article names.

        By default the seven well-known FA header articles
        (``CADRE-*``, ``DATE-DES-DONNEES``, ``DATX-DES-DONNEES``) are
        excluded. Set ``include_misc=True`` to keep every non-header
        article in file order, including FULLPOS-style markers and any
        other Misc payload that lives next to the data fields.
        """

        skip = {
            "CADRE-DIMENSIONS",
            "CADRE-FRANKSCHMI",
            "CADRE-REDPOINPOL",
            "CADRE-SINLATITUD",
            "CADRE-FOCOHYBRID",
            "DATE-DES-DONNEES",
            "DATX-DES-DONNEES",
        }
        return [a.name for a in self.articles if a.name not in skip]
