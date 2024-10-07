"""
source : https://api.python.langchain.com/en/latest/_modules/langchain_community/cache.html#InMemoryCache

This workaround is to solve this: https://github.com/langchain-ai/langchain/issues/22389
Create a caching class that looks like it's just in memory but actually saves to sql

"""


import zlib
import sqlite3
import json
import pickle
from pathlib import Path, PosixPath
from typing import Union, Any, List
from threading import Lock

from langchain_core.caches import BaseCache

# use the same lock for each instance accessing the same db, as well as a
# global lock to create new locks
databases_locks = {"global": Lock()}
# also use the same cache
databases_caches = {}

SQLITE3_CHECK_SAME_THREAD=False

class SQLiteCacheFixed(BaseCache):
    """Cache that stores things in memory."""
    __VERSION__ = "0.4"

    def __init__(
        self,
        database_path: Union[str, PosixPath],
        ) -> None:
        dbp = Path(database_path)
        # add versioning to avoid trying to use non backward compatible version
        self.database_path = dbp.parent / (dbp.stem + f"_v{self.__VERSION__}" + dbp.suffix)

        self.lockkey = str(self.database_path.absolute().resolve())
        if self.lockkey not in databases_locks:
            with databases_locks["global"]:
                databases_locks[self.lockkey] = Lock()
                databases_caches[self.lockkey] = {}
        self.lock = databases_locks[self.lockkey]
        with self.lock:
            with databases_locks["global"]:
                self._cache = databases_caches[self.lockkey]

        # create db
        conn = sqlite3.connect(self.database_path, check_same_thread=SQLITE3_CHECK_SAME_THREAD)
        cursor = conn.cursor()
        try:
            with self.lock:
                cursor.execute("BEGIN")
                cursor.execute('''CREATE TABLE IF NOT EXISTS cache (
                                key TEXT PRIMARY KEY,
                                data BLOB
                                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                                )''')
                conn.commit()
        finally:
            conn.close()


    def lookup(self, prompt: str, llm_string: str) -> Any:
        """Look up based on prompt and llm_string."""
        key = json.dumps((prompt, llm_string))

        return self.__get_data__(key=key)

    def __get_data__(self, key: str) -> Any:
        "actual lookup through cache or db"
        # check the cache is still as expected
        with databases_locks["global"]:
            assert id(self._cache) == id(databases_caches[self.lockkey])

        # already cached
        if key in self._cache:
            return self._cache[key]

        # load the value from the db
        conn = sqlite3.connect(self.database_path, check_same_thread=SQLITE3_CHECK_SAME_THREAD)
        cursor = conn.cursor()
        try:
            with self.lock:
                cursor.execute("BEGIN")
                cursor.execute('SELECT data FROM cache WHERE key = ?', (key,))
                cursor.execute("UPDATE cache SET timestamp = CURRENT_TIMESTAMP WHERE key = ?", (key,))
                result = cursor.fetchone()

        finally:
            conn.close()
        if result:
            result = result[0]
        if result:
            result = pickle.loads(zlib.decompress(result))
            with self.lock:
                self._cache[key] = result
        return result

    def update(self, prompt: str, llm_string: str, return_val: Any) -> None:
        """Update cache based on prompt and llm_string."""
        key = json.dumps((prompt, llm_string))

        self.__set_data__(key=key, data=return_val)


    def __set_data__(self, key: str, data: Any) -> None:
        "actual code to set the data in the db then the cache"
        if key in self._cache and self._cache[key] == data:
            return

        conn = sqlite3.connect(self.database_path, check_same_thread=SQLITE3_CHECK_SAME_THREAD)
        cursor = conn.cursor()
        compressed = zlib.compress(pickle.dumps(data))
        try:
            with self.lock:
                cursor.execute("BEGIN")
                cursor.execute("INSERT OR REPLACE INTO cache (key, data) VALUES (?, ?)", (key, compressed))
                conn.commit()
                self._cache[key] = data
        finally:
            conn.close()


    def clear(self) -> None:
        """Clear cache."""
        with self.lock:
            self._cache.clear()


    async def alookup(self, prompt: str, llm_string: str) -> Any:
        """Look up based on prompt and llm_string."""
        return self.lookup(prompt, llm_string)


    async def aupdate(
        self, prompt: str, llm_string: str, return_val: Any
    ) -> None:
        """Update cache based on prompt and llm_string."""
        self.update(prompt, llm_string, return_val)


    async def aclear(self) -> None:
        """Clear cache."""
        self.clear()

    def __get_keys__(self) -> List[str]:
        "get the list of keys present in the db"
        # load the value from the db
        conn = sqlite3.connect(self.database_path, check_same_thread=SQLITE3_CHECK_SAME_THREAD)
        cursor = conn.cursor()
        try:
            with self.lock:
                cursor.execute("BEGIN")
                cursor.execute('SELECT key FROM cache')
                results = [row[0] if row else None for row in cursor.fetchall()]
        finally:
            conn.close()
        for r in results:
            yield r
