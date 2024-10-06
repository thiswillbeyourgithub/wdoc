"""
source : https://api.python.langchain.com/en/latest/_modules/langchain/storage/file_system.html#LocalFileStore

This is basically the exact same code but with added compression
"""
import os
import re
import time
import zlib
from pathlib import Path
from typing import Iterator, List, Optional, Sequence, Tuple, Union
from functools import cache as memoize

from langchain_core.stores import ByteStore

from langchain.storage.exceptions import InvalidKeyException

HASH_REGEX = re.compile(r"^[a-zA-Z0-9_.\-/]+$")


class LocalFileStore(ByteStore):
    """BaseStore interface that works on the local file system.

    Examples:
        Create a LocalFileStore instance and perform operations on it:

        .. code-block:: python

            from langchain.storage import LocalFileStore

            # Instantiate the LocalFileStore with the root path
            file_store = LocalFileStore("/path/to/root")

            # Set values for keys
            file_store.mset([("key1", b"value1"), ("key2", b"value2")])

            # Get values for keys
            values = file_store.mget(["key1", "key2"])  # Returns [b"value1", b"value2"]

            # Delete keys
            file_store.mdelete(["key1"])

            # Iterate over keys
            for key in file_store.yield_keys():
                print(key)  # noqa: T201

    """
    __VERSION__ = "0.1"  # anticipate breaking cache change

    def __init__(
        self,
        root_path: Union[str, Path],
        *,
        chmod_file: Optional[int] = None,
        chmod_dir: Optional[int] = None,
        update_atime: bool = False,
        compress: Union[bool, int] = False,
        regex_check: bool = False,
    ) -> None:
        """Implement the BaseStore interface for the local file system.

        Args:
            root_path (Union[str, Path]): The root path of the file store. All keys are
                interpreted as paths relative to this root.
            chmod_file: (optional, defaults to `None`) If specified, sets permissions
                for newly created files, overriding the current `umask` if needed.
            chmod_dir: (optional, defaults to `None`) If specified, sets permissions
                for newly created dirs, overriding the current `umask` if needed.
            update_atime: (optional, defaults to `False`) If `True`, updates the
                filesystem access time (but not the modified time) when a file is read.
                This allows MRU/LRU cache policies to be implemented for filesystems
                where access time updates are disabled.
            compress: (optional, defaults to `False`) If an int, compress
                stored data, reducing speed but lowering size. The int given
                is the level of compression of zlib, so between -1 and 9, both
                included. If `True`, defaults to -1, like in zlib.
            regex_check: (bool, defaults to `False`) If True will check
                that keys indeed contain only characters expected in a hash.
        """
        self.chmod_file = chmod_file
        self.chmod_dir = chmod_dir
        self.update_atime = update_atime
        self.regex_check = regex_check

        if compress is True:
            compress = -1
        if isinstance(compress, int):
            assert compress >= -1 and compress <= 9, (
                "compress arg as int must be between -1 and 9, both "
                f"included. Not {compress}"
            )
        self.compress = compress

        rp = Path(root_path).absolute().resolve()
        rp = rp.parent / (rp.stem + f"_v{self.__VERSION__}" + rp.suffix)
        if self.compress:
            rp = rp.parent / (rp.stem + "_z" + rp.suffix)
        self.root_path = rp
        self._mkdir_for_store()


    @memoize
    def _check_key_regex(self, key: str) -> None:
        """memoized regex checker for if the key is indeed a hash and does
        not contain invalid characters"""
        if not re.match(HASH_REGEX, key):
            raise InvalidKeyException(f"Invalid characters in key (the key should be a hash): {key}")

    def _mkdir_for_store(self, dir: Path) -> None:
        """Makes a store directory path (including parents) with specified permissions

        This is needed because `Path.mkdir()` is restricted by the current `umask`,
        whereas the explicit `os.chmod()` used here is not.

        Args:
            dir: (Path) The store directory to make

        Returns:
            None
        """
        if not dir.exists():
            self._mkdir_for_store(dir.parent)
            dir.mkdir(exist_ok=True)
        if self.chmod_dir is not None:
            os.chmod(dir, self.chmod_dir)

    def mget(self, keys: Sequence[str]) -> List[Optional[bytes]]:
        """Get the values associated with the given keys.

        Args:
            keys: A sequence of keys.

        Returns:
            A sequence of optional values associated with the keys.
            If a key is not found, the corresponding value will be None.
        """
        values: List[Optional[bytes]] = []
        for key in keys:
            if self.regex_check:
                self._check_key_regex(key)
            full_path = self.root_path / key
            if full_path.exists():
                value = full_path.read_bytes()
                if self.compress:
                    value = zlib.decompress(value)
                values.append(value)
                if self.update_atime:
                    # update access time only; preserve modified time
                    os.utime(full_path, (time.time(), os.stat(full_path).st_mtime))
            else:
                values.append(None)
        return values


    def mset(self, key_value_pairs: Sequence[Tuple[str, bytes]]) -> None:
        """Set the values for the given keys.

        Args:
            key_value_pairs: A sequence of key-value pairs.

        Returns:
            None
        """
        for key, value in key_value_pairs:
            if self.regex_check:
                self._check_key_regex(key)
            full_path = self.root_path / key
            if self.compress:
                com_val = zlib.compress(value, level=self.compress)
                full_path.write_bytes(com_val)
            else:
                full_path.write_bytes(value)
            if self.chmod_file is not None:
                os.chmod(full_path, self.chmod_file)


    def mdelete(self, keys: Sequence[str]) -> None:
        """Delete the given keys and their associated values.

        Args:
            keys (Sequence[str]): A sequence of keys to delete.

        Returns:
            None
        """
        for key in keys:
            full_path = self.root_path / key
            if full_path.exists():
                full_path.unlink()


    def yield_keys(self, prefix: Optional[str] = None) -> Iterator[str]:
        """Get an iterator over keys that match the given prefix.

        Args:
            prefix (Optional[str]): The prefix to match.

        Returns:
            Iterator[str]: An iterator over keys that match the given prefix.
        """
        prefix_path = self.root_path / prefix if prefix else self.root_path
        for file in prefix_path.rglob("*"):
            if file.is_file():
                relative_path = file.relative_to(self.root_path)
                yield str(relative_path)
