"""
source : https://api.python.langchain.com/en/latest/_modules/langchain/storage/file_system.html#LocalFileStore

This is basically the exact same code but with added compression
"""
from pathlib import Path
from typing import Iterator, List, Optional, Sequence, Tuple, Union
import hashlib

from langchain_core.stores import ByteStore

from .fix_llm_caching import SQLiteCacheFixed

def hasher(text: str) -> str:
    """used to hash the text contant of each doc to cache the splitting and
    embeddings"""
    return hashlib.sha256(text.encode()).hexdigest()[:20]


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

    def __init__(
        self,
        database_path: Union[str, Path],
        *args,
        **kwargs,
    ) -> None:
        """Implement the BaseStore interface for the local file system.

        Args:
            database_path (Union[str, Path]): The path to the sqlite to use
            *args: All other args are ignored
            **kwargs: Ignored too
        """
        database_path = Path(database_path)
        if database_path.is_dir():
            database_path = database_path / "db.sqlite"
        self._sqlcache = SQLiteCacheFixed(
            database_path=database_path,
        )

    def mget(self, keys: Sequence[str]) -> List[Optional[bytes]]:
        """Get the values associated with the given keys.

        Args:
            keys: A sequence of keys.

        Returns:
            A sequence of optional values associated with the keys.
            If a key is not found, the corresponding value will be None.
        """
        values = [
            self._sqlcache.lookup(
                prompt=key,
                llm_string=""
            ) for key in keys
        ]
        # deleted keys are stored as 'None'
        for iv, v in enumerate(values):
            if isinstance(v, str) and v == "None":
                values[iv] = None
                with self._sqlcache.lock:
                    del self._sqlcache._cache[(keys[iv], "")]
        return values


    def mset(self, key_value_pairs: Sequence[Tuple[str, bytes]]) -> None:
        """Set the values for the given keys.

        Args:
            key_value_pairs: A sequence of key-value pairs.

        Returns:
            None
        """
        for key, value in key_value_pairs:
            self._sqlcache.update(
                prompt=key,
                llm_string="",
                return_val=value,
            )


    def mdelete(self, keys: Sequence[str]) -> None:
        """Delete the given keys and their associated values.

        Args:
            keys (Sequence[str]): A sequence of keys to delete.

        Returns:
            None
        """
        groups = []
        for key in keys:
            groups.append((key, "None"))
        self.mset(self.groups)


    def yield_keys(self, prefix: Optional[str] = None) -> Iterator[str]:
        """Get an iterator over keys that match the given prefix.

        Args:
            prefix (Optional[str]): The prefix to match.

        Returns:
            Iterator[str]: An iterator over keys that match the given prefix.
        """
        for k in self._sqlcache.__get_keys__():
            yield k
