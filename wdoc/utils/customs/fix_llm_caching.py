"""
source : https://api.python.langchain.com/en/latest/_modules/langchain_community/cache.html#InMemoryCache

This workaround is to solve this: https://github.com/langchain-ai/langchain/issues/22389
Create a caching class that looks like it's just in memory but actually saves to sql

"""


import zlib
import sqlite3
import dill
from pathlib import Path, PosixPath
from typing import Union, Optional
from threading import Lock

from langchain_core.caches import RETURN_VAL_TYPE, BaseCache

# use the same lock for each instance accessing the same db, as well as a
# global lock to create new locks
databases_locks = {"global": Lock()}

SQLITE3_CHECK_SAME_THREAD=False

class SQLiteCacheFixed(BaseCache):
    """Cache that stores things in memory."""
    __VERSION__ = "0.1"

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
        self.lock = databases_locks[self.lockkey]

        if self.database_path.exists():
            self.clear()
        else:
            conn = sqlite3.connect(self.database_path, check_same_thread=SQLITE3_CHECK_SAME_THREAD)
            cursor = conn.cursor()
            with self.lock:
                cursor.execute('''CREATE TABLE IF NOT EXISTS saved_llm_calls
                                (id INTEGER PRIMARY KEY AUTOINCREMENT,
                                data TEXT)''')
            conn.close()
            self._cache = {}


    def lookup(self, prompt: str, llm_string: str) -> Optional[RETURN_VAL_TYPE]:
        """Look up based on prompt and llm_string."""
        key = (prompt, llm_string)
        if key in self._cache:
            return self._cache[key]


    def update(self, prompt: str, llm_string: str, return_val: RETURN_VAL_TYPE) -> None:
        """Update cache based on prompt and llm_string."""
        key = (prompt, llm_string)
        if key in self._cache and self._cache[key] == return_val:
            return
        self._cache[(prompt, llm_string)] = return_val
        data = zlib.compress(dill.dumps({"key": key, "value": return_val}))
        conn = sqlite3.connect(self.database_path, check_same_thread=SQLITE3_CHECK_SAME_THREAD)
        cursor = conn.cursor()
        with self.lock:
            cursor.execute("INSERT INTO saved_llm_calls (data) VALUES (?)", (data,))
            conn.commit()
        conn.close()


    def clear(self) -> None:
        """Clear cache."""
        conn = sqlite3.connect(self.database_path, check_same_thread=SQLITE3_CHECK_SAME_THREAD)
        cursor = conn.cursor()
        with self.lock:
            cursor.execute('''CREATE TABLE IF NOT EXISTS saved_llm_calls
                            (id INTEGER PRIMARY KEY AUTOINCREMENT,
                            data TEXT)''')
            cursor.execute("SELECT data FROM saved_llm_calls")
            conn.commit()
        rows = cursor.fetchall()
        conn.close()
        datas = [
            dill.loads(
                zlib.decompress(row[0])
            ) for row in rows
        ]
        self._cache = {
            d["key"]: d["value"]
            for d in datas
        }


    async def alookup(self, prompt: str, llm_string: str) -> Optional[RETURN_VAL_TYPE]:
        """Look up based on prompt and llm_string."""
        return self.lookup(prompt, llm_string)


    async def aupdate(
        self, prompt: str, llm_string: str, return_val: RETURN_VAL_TYPE
    ) -> None:
        """Update cache based on prompt and llm_string."""
        self.update(prompt, llm_string, return_val)


    async def aclear(self) -> None:
        """Clear cache."""
        self.clear()

