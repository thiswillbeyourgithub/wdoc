"""
source : https://api.python.langchain.com/en/latest/_modules/langchain/storage/file_system.html#LocalFileStore

This is basically the exact same code but with added compression
"""

from pathlib import Path

from beartype.typing import Iterator, List, Optional, Sequence, Tuple, Union
from langchain_core.stores import ByteStore
from PersistDict import PersistDict


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
        expiration_days: int = None,
        verbose: bool = False,
        *args,
        **kwargs,
    ) -> None:
        """Implement the BaseStore interface for the local file system.

        Args:
            database_path (Union[str, Path]): The path to the sqlite to use
            expiration_days: int, embeddings older than this will get removed
            verbose: bool, default Fakle
            *args: All other args are ignored
            **kwargs: Ignored too
        """
        Path(database_path).parent.mkdir(exist_ok=True, parents=True)
        self.pdi = PersistDict(
            database_path=database_path,
            expiration_days=expiration_days,
            verbose=verbose,
        )

    def mget(self, keys: Sequence[str]) -> List[Optional[bytes]]:
        """Get the values associated with the given keys.

        Args:
            keys: A sequence of keys.

        Returns:
            A sequence of optional values associated with the keys.
            If a key is not found, the corresponding value will be None.
        """
        return [self.pdi[k] if k in self.pdi else None for k in keys]

    def mset(self, key_value_pairs: Sequence[Tuple[str, bytes]]) -> None:
        """Set the values for the given keys.

        Args:
            key_value_pairs: A sequence of key-value pairs.

        Returns:
            None
        """
        for k, v in key_value_pairs:
            self.pdi[k] = v

    def mdelete(self, keys: Sequence[str]) -> None:
        """Delete the given keys and their associated values.

        Args:
            keys (Sequence[str]): A sequence of keys to delete.

        Returns:
            None
        """
        for k in keys:
            if k in self.pdi:
                del self.pdi[k]

    def yield_keys(self, prefix: Optional[str] = None) -> Iterator[str]:
        """Get an iterator over keys that match the given prefix.

        Args:
            prefix (Optional[str]): The prefix to match.

        Returns:
            Iterator[str]: An iterator over keys that match the given prefix.
        """
        for k in self.pdi.keys():
            yield k
