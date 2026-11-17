import os 
import threading
from datetime import datetime, timezone

ANON_PER_IP = int(os.getenv("ANON_ANALYSES_PER_IP_DAY", 8))
ANON_GLOBAL = int(os.getenv("ANON_ANALYSES_GLOBAL_DAY", "600"))
FREE_PER_USER = int(os.getenv("FREE_ANALYSES_PER_USER_DAY", "30"))


class DailyCounter:
    def __init__(self):
        self._lock = threading.Lock()
        self._day = None
        self._counts = dict[str, int] = {}

    def hit(self, key: str, limit: int) -> bool:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        with self._lock:
            if self._day != today:
                self._day, self._counts = today, {}
            n = self._counts.get(key, 0)
            if n >= limit:
                return False
            self._counts[key] = n + 1
            return True


_counter = DailyCounter()


def allow_limited_analysis(uid, ip):
    if uid:
        return _counter.hit(f"user:{uid}", FREE_PER_USER), "DAILY_LIMIT_USER"
    if not _counter.hit(f"ip:{ip}", ANON_PER_IP):
        return False, "DAILY_LIMIT_ANON"
    if not _counter.hit("anon-global", ANON_GLOBAL):
        return False, "CAPACITY_REACHED"
    return True, ""