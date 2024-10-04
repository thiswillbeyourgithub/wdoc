"""
source : https://api.python.langchain.com/en/latest/_modules/langchain_community/cache.html#InMemoryCache

This workaround is to solve this: https://github.com/langchain-ai/langchain/issues/22389
Create a caching class that looks like it's just in memory but actually saves to sql

"""


import zlib
import sqlite3
import dill
from pathlib import Path, PosixPath
from typing import Union, Any, Optional
from threading import Lock

from langchain_core.caches import RETURN_VAL_TYPE, BaseCache

class SQLiteCacheFixed(BaseCache):
    """Cache that stores things in memory."""

    def __init__(
        self,
        database_path: Union[str, PosixPath],
        ) -> None:
        self.lock = Lock()
        self.database_path = Path(database_path)
        if database_path.exists():
            self.clear()
        else:
            conn = sqlite3.connect(self.database_path)
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
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()
        with self.lock:
            cursor.execute("INSERT INTO saved_llm_calls (data) VALUES (?)", (data,))
            conn.commit()
            conn.close()


    def clear(self, **kwargs: Any) -> None:
        """Clear cache."""
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()
        with self.lock:
            cursor.execute('''CREATE TABLE IF NOT EXISTS saved_llm_calls
                            (id INTEGER PRIMARY KEY AUTOINCREMENT,
                            data TEXT)''')
            cursor.execute("SELECT data FROM saved_llm_calls")
            rows = cursor.fetchall()
            conn.commit()
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


    async def aclear(self, **kwargs: Any) -> None:
        """Clear cache."""
        self.clear()

