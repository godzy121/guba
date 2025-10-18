"""Utilities for downloading popularity rankings from Eastmoney.

The original script that motivated this module was a single file with a
mixture of global functions, print based logging and a ``RateLimiter``
implementation that never actually initialised because ``__init__`` was
misspelled.  As soon as the crawler started running the rate limiter was
silently bypassed, the API received a burst of requests and began to
return ``403`` responses.  Once Eastmoney starts rate limiting the cookie
rotation approach becomes increasingly ineffective and the crawler never
recovers.

This module keeps the original behaviour but tidies the implementation so
that the limiter is honoured and a couple of additional safeguards are in
place.  The intention is to provide an easy to consume Python API while
remaining usable as a small utility script.

为了方便在聚宽等无法轻易创建额外文件的研究环境里使用，模块最后
提供了 :func:`register_eastmoney_helpers`，可以直接把常用函数注入到
Notebook 的 ``globals()`` 中，复制粘贴即可使用。
"""

import logging
import os
import random
import re
import time
import uuid
from datetime import date as date_cls
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import requests

try:  # pandas is optional – only needed when CSV output is requested.
    import pandas as pd
except Exception:  # pragma: no cover - pandas is optional in tests.
    pd = None

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _strip_cookie(cookie: str) -> str:
    """Normalise and strip cookie strings."""

    return (cookie or "").strip()


def parse_cookie_str(cookie_str: str) -> Dict[str, str]:
    """Parse the Eastmoney cookie string into a dictionary."""

    out: Dict[str, str] = {}
    for chunk in (cookie_str or "").split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "=" not in chunk:
            continue
        key, value = chunk.split("=", 1)
        out[key.strip()] = value.strip()
    return out


def safe_float(value: Any) -> Optional[float]:
    """Best effort conversion from ``value`` to ``float``."""

    try:
        if value is None:
            return None
        if isinstance(value, str):
            value = value.replace(",", "")
        return float(value)
    except Exception:  # pragma: no cover - defensive.
        return None


def to_date_obj(value: Union[str, datetime, date_cls]) -> datetime:
    """Convert ``value`` to :class:`datetime`.

    Multiple input formats are accepted because the crawler usually reads
    from user provided configuration values.
    """

    if isinstance(value, datetime):
        return value
    if isinstance(value, date_cls):
        return datetime(value.year, value.month, value.day)

    text = str(value).strip()
    if re.fullmatch(r"\d{8}", text):
        return datetime.strptime(text, "%Y%m%d")
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
        return datetime.strptime(text, "%Y-%m-%d")

    zh = re.match(r"(\d{4})年(\d{1,2})月(\d{1,2})日", text)
    if zh:
        year, month, day = map(int, zh.groups())
        return datetime(year, month, day)

    raise ValueError(f"无法解析日期：{value!r}")


def date_zh(value: Union[str, datetime, date_cls]) -> str:
    dt = to_date_obj(value)
    return f"{dt.year}年{dt.month}月{dt.day}日"


def date_dash(value: Union[str, datetime, date_cls]) -> str:
    return to_date_obj(value).strftime("%Y-%m-%d")


# ---------------------------------------------------------------------------
# rate limiting
# ---------------------------------------------------------------------------


class RateLimiter(object):
    """Token bucket style rate limiter with jitter.

    聚宽等平台通常运行在较老的 Python 版本（例如 3.6），因此不能依赖
    ``dataclasses``。这个实现保留了原有功能，同时兼容旧版本。
    """

    def __init__(self, min_interval=1.0, jitter=0.4):
        if min_interval < 0:
            raise ValueError("min_interval must be non-negative")
        if jitter < 0:
            raise ValueError("jitter must be non-negative")

        self.min_interval = float(min_interval)
        self.jitter = float(jitter)
        self._last = 0.0

    def wait(self):
        now = time.time()
        gap = now - self._last
        wait_for = self.min_interval - gap
        if wait_for > 0:
            jitter = random.uniform(0, self.jitter)
            time.sleep(wait_for + jitter)
            self._last = time.time()
        else:
            self._last = now


# ---------------------------------------------------------------------------
# Eastmoney client
# ---------------------------------------------------------------------------


class EastmoneyClient:
    """HTTP client for the Eastmoney popularity ranking endpoint."""

    URL = "https://np-tjxg-g.eastmoney.com/api/smart-tag/stock/v3/pw/search-code"

    def __init__(
        self,
        cookie: str,
        user_agent: Optional[str] = None,
        timeout: Tuple[float, float] = (6.0, 15.0),
        proxies: Optional[Dict[str, str]] = None,
        limiter: Optional[RateLimiter] = None,
        max_retries: int = 3,
        backoff_base: float = 1.8,
    ) -> None:
        self.cookie = _strip_cookie(cookie)
        self.ua = user_agent or (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.1 Safari/537.36"
        )
        self.timeout = timeout
        self.proxies = proxies
        self.session = requests.Session()
        self.limiter = limiter or RateLimiter(1.1, 0.5)
        self.max_retries = max(1, int(max_retries))
        self.backoff_base = max(1.0, float(backoff_base))

        if "qgqp_b_id" not in parse_cookie_str(self.cookie):
            raise ValueError("Cookie 中缺少 qgqp_b_id，请从浏览器完整复制。")

    # ------------------------------------------------------------------
    # request helpers
    # ------------------------------------------------------------------

    def _headers(self) -> Dict[str, str]:
        return {
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            "Content-Type": "application/json",
            "Origin": "https://xuangu.eastmoney.com",
            "Referer": "https://xuangu.eastmoney.com/",
            "User-Agent": self.ua,
            "Cookie": self.cookie,
            "Host": "np-tjxg-g.eastmoney.com",
            "Pragma": "no-cache",
            "Cache-Control": "no-cache",
        }

    @staticmethod
    def _request_id() -> str:
        return uuid.uuid4().hex

    @staticmethod
    def _timestamp_us() -> str:
        return str(int(time.time() * 1_000_000))

    def _fingerprint(self) -> str:
        cookie_dict = parse_cookie_str(self.cookie)
        return cookie_dict.get("qgqp_b_id") or "0000000000000000"

    def _post_json(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        attempt = 0
        while True:
            attempt += 1
            self.limiter.wait()
            try:
                response = self.session.post(
                    self.URL,
                    json=payload,
                    headers=self._headers(),
                    timeout=self.timeout,
                    proxies=self.proxies,
                )

                if response.status_code in {403, 429}:
                    raise requests.HTTPError(
                        f"{response.status_code} Throttled/Forbidden",
                        response=response,
                    )

                if response.status_code >= 500:
                    raise requests.HTTPError(
                        f"{response.status_code} Server Error", response=response
                    )

                response.raise_for_status()
                return response.json()
            except Exception as exc:  # pragma: no cover - network failure path.
                if attempt >= self.max_retries:
                    raise

                backoff = (self.backoff_base**attempt) + random.uniform(0, 1.0)
                LOGGER.warning(
                    "POST failed (%s/%s): %s – sleeping %.1fs",
                    attempt,
                    self.max_retries,
                    exc,
                    backoff,
                )
                time.sleep(backoff)

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def fetch_heat_one_day(
        self,
        day: Union[str, datetime, date_cls],
        page_size: int = 200,
        max_pages: Optional[int] = None,
        as_dataframe: bool = True,
    ) -> Union[List[Dict[str, Any]], "pd.DataFrame"]:
        zh = date_zh(day)
        dash = date_dash(day)
        keyword = f"{zh}股吧人气榜排名；不要退市股;不要退市股;不要北交所"

        results: List[Dict[str, Any]] = []
        page_no = 1
        total = None

        while True:
            payload = {
                "keyWord": keyword,
                "pageSize": int(page_size),
                "pageNo": int(page_no),
                "biz": "web_ai_select_stocks",
                "client": "web",
                "dxInfo": [],
                "gids": [],
                "matchWord": "",
                "needCorrect": True,
                "ownSelectAll": False,
                "removedConditionIdList": [],
                "shareToGuba": False,
                "requestId": self._request_id(),
                "timestamp": self._timestamp_us(),
                "fingerprint": self._fingerprint(),
            }

            response_json = self._post_json(payload)
            data = (response_json or {}).get("data", {})
            result = data.get("result", {}) if isinstance(data, dict) else {}
            if total is None:
                total = result.get("total")

            rows = result.get("dataList") or []
            key_rank = f"POPULARITY_LIST{{{dash}}}"

            for row in rows:
                rank_value = row.get(key_rank)
                rank = int(rank_value) if str(rank_value).isdigit() else None
                results.append(
                    {
                        "date": dash,
                        "code": row.get("SECURITY_CODE"),
                        "name": row.get("SECURITY_SHORT_NAME"),
                        "rank": rank,
                    }
                )

            if not rows:
                break
            if total is not None and len(results) >= int(total):
                break
            if max_pages and page_no >= max_pages:
                break

            page_no += 1

        if as_dataframe and pd is not None:
            df = pd.DataFrame(results)
            if not df.empty and "rank" in df.columns:
                df = df.sort_values(["rank", "code"], na_position="last")
            return df.reset_index(drop=True)

        return results


# ---------------------------------------------------------------------------
# Crawl helpers
# ---------------------------------------------------------------------------


def _ensure_out_dir(path: Union[str, os.PathLike[str]]) -> Path:
    out_dir = Path(path)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def crawl_em_heat_by_range(
    cookie: str,
    start_date: Union[str, datetime, date_cls],
    end_date: Union[str, datetime, date_cls],
    out_dir: str = "./em_heat_daily",
    top_k: int = 300,
    page_size: int = 200,
    overwrite: bool = False,
    min_gap: float = 1.2,
    max_gap: float = 3.0,
    fail_log_path: Optional[str] = None,
    proxies: Optional[Dict[str, str]] = None,
) -> None:
    """Fetch popularity rankings for the inclusive date range.

    The crawler is intentionally conservative with its pacing to avoid
    triggering rate limiting on the upstream API.
    """

    out_dir_path = _ensure_out_dir(out_dir)
    if fail_log_path is None:
        fail_log_path = str(out_dir_path / "failed_dates.log")

    jitter = max(0.0, max_gap - min_gap)
    limiter = RateLimiter(min_interval=min_gap, jitter=jitter)
    client = EastmoneyClient(cookie=cookie, proxies=proxies, limiter=limiter)

    start = to_date_obj(start_date)
    end = to_date_obj(end_date)

    def append_fail(date_str: str) -> None:
        with open(fail_log_path, "a", encoding="utf-8") as handle:
            handle.write(date_str + "\n")

    current = start
    while current <= end:
        current_dash = current.strftime("%Y-%m-%d")
        csv_path = out_dir_path / f"eastmoney_heat{current.strftime('%Y%m%d')}.csv"

        if csv_path.exists() and not overwrite:
            LOGGER.info("[SKIP] %s 已存在", current_dash)
        else:
            LOGGER.info("[FETCH] %s ...", current_dash)
            try:
                df = client.fetch_heat_one_day(
                    current_dash,
                    page_size=page_size,
                    as_dataframe=True,
                )

                if pd is None:
                    raise RuntimeError("需要 pandas 才能保存 CSV，请先安装 pandas")

                if top_k and "rank" in df.columns:
                    df = df.sort_values("rank").head(top_k).reset_index(drop=True)

                df.to_csv(csv_path, index=False, encoding="utf-8-sig")
                LOGGER.info("[OK] %s -> %s (%s rows)", current_dash, csv_path.name, len(df))
            except Exception as exc:  # pragma: no cover - network / IO failure.
                LOGGER.error("[FAIL] %s -> %s", current_dash, exc)
                append_fail(current_dash)

            sleep_for = random.uniform(min_gap, max_gap)
            time.sleep(sleep_for)

        current += timedelta(days=1)


def retry_failed_dates_em(
    cookie: str,
    out_dir: str = "./em_heat_daily",
    top_k: int = 300,
    page_size: int = 200,
    fail_log_path: Optional[str] = None,
    overwrite: bool = True,
    proxies: Optional[Dict[str, str]] = None,
) -> None:
    out_dir_path = _ensure_out_dir(out_dir)
    if fail_log_path is None:
        fail_log_path = str(out_dir_path / "failed_dates.log")

    if not os.path.exists(fail_log_path):
        LOGGER.info("没有失败日期需要重试。")
        return

    limiter = RateLimiter(min_interval=1.2, jitter=1.8)
    client = EastmoneyClient(cookie=cookie, proxies=proxies, limiter=limiter)

    with open(fail_log_path, "r", encoding="utf-8") as handle:
        dates = [line.strip() for line in handle if line.strip()]

    if not dates:
        LOGGER.info("失败列表为空。")
        return

    remain: List[str] = []
    for dash in dates:
        csv_path = out_dir_path / f"eastmoney_heat{dash.replace('-', '')}.csv"

        if csv_path.exists() and not overwrite:
            LOGGER.info("[SKIP] %s 已存在", dash)
            continue

        LOGGER.info("[RETRY] %s ...", dash)
        try:
            df = client.fetch_heat_one_day(dash, page_size=page_size, as_dataframe=True)

            if pd is None:
                raise RuntimeError("需要 pandas 保存 CSV，请安装 pandas")

            if top_k and "rank" in df.columns:
                df = df.sort_values("rank").head(top_k).reset_index(drop=True)

            df.to_csv(csv_path, index=False, encoding="utf-8-sig")
            LOGGER.info("[OK] %s -> %s (%s rows)", dash, csv_path.name, len(df))
        except Exception as exc:  # pragma: no cover
            LOGGER.error("[FAIL] %s -> %s", dash, exc)
            remain.append(dash)

        time.sleep(random.uniform(1.2, 3.0))

    with open(fail_log_path, "w", encoding="utf-8") as handle:
        for dash in remain:
            handle.write(dash + "\n")


def register_eastmoney_helpers(namespace=None, prefix=""):
    """Inject helpers into ``namespace`` for notebook style workflows.

    Example::

        register_eastmoney_helpers(globals())
        crawl_em_heat_by_range(...)

    Parameters
    ----------
    namespace:
        ``dict``-like object to receive the helpers.  Defaults to the caller's
        global namespace.
    prefix:
        Optional string prefix applied to the exported names.  For example,
        ``prefix="em_"`` will export ``em_crawl_em_heat_by_range`` etc.
    """

    if namespace is None:
        import inspect

        frame = inspect.currentframe()
        try:
            namespace = frame.f_back.f_globals if frame and frame.f_back else globals()
        finally:
            del frame

    exports = {
        "RateLimiter": RateLimiter,
        "EastmoneyClient": EastmoneyClient,
        "crawl_em_heat_by_range": crawl_em_heat_by_range,
        "retry_failed_dates_em": retry_failed_dates_em,
        "parse_cookie_str": parse_cookie_str,
        "safe_float": safe_float,
        "to_date_obj": to_date_obj,
        "date_zh": date_zh,
        "date_dash": date_dash,
    }

    for name, value in exports.items():
        namespace[prefix + name] = value

    return exports


def _load_cookie_from_env() -> Optional[str]:
    return os.environ.get("EASTMONEY_COOKIE")


def _load_config_from_env() -> Dict[str, Any]:
    return {
        "cookie": _load_cookie_from_env(),
        "start_date": os.environ.get("EM_START_DATE"),
        "end_date": os.environ.get("EM_END_DATE"),
        "out_dir": os.environ.get("EM_OUT_DIR", "./dc_heat"),
    }


def main(args: Optional[Iterable[str]] = None) -> None:  # pragma: no cover - CLI glue
    import argparse

    parser = argparse.ArgumentParser(description="Fetch Eastmoney heat rankings")
    parser.add_argument("cookie", nargs="?", help="Eastmoney cookie")
    parser.add_argument("start_date", help="Start date (YYYY-MM-DD)")
    parser.add_argument("end_date", help="End date (YYYY-MM-DD)")
    parser.add_argument("--out-dir", default="./em_heat_daily", help="Output directory")
    parser.add_argument("--top-k", type=int, default=300, help="Number of rows to keep")
    parser.add_argument("--page-size", type=int, default=200, help="Page size")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing CSVs")
    parser.add_argument("--min-gap", type=float, default=1.2, help="Minimum gap between days")
    parser.add_argument("--max-gap", type=float, default=3.0, help="Maximum gap between days")
    parser.add_argument(
        "--fail-log", default=None, help="Path to failed dates log (defaults inside out dir)"
    )

    env_config = _load_config_from_env()
    parsed = parser.parse_args(args=args)

    cookie = parsed.cookie or env_config.get("cookie")
    if not cookie:
        raise SystemExit("Missing cookie – pass as argument or set EASTMONEY_COOKIE")

    crawl_em_heat_by_range(
        cookie=cookie,
        start_date=parsed.start_date,
        end_date=parsed.end_date,
        out_dir=parsed.out_dir,
        top_k=parsed.top_k,
        page_size=parsed.page_size,
        overwrite=parsed.overwrite,
        min_gap=parsed.min_gap,
        max_gap=parsed.max_gap,
        fail_log_path=parsed.fail_log,
    )


if __name__ == "__main__":  # pragma: no cover - manual execution only.
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()

