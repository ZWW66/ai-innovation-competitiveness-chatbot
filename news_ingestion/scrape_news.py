"""Collect AI news from RSS/Atom feeds and save a deduplicated CSV.

Run from the project root:
    python -m news_ingestion.scrape_news

The scraper prefers full text embedded in feeds, fetches article pages only when
needed, retries transient HTTP errors, extracts pages concurrently, filters broad
feeds for AI relevance, and incrementally merges with the newest prior CSV.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import re
import threading
import time
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import feedparser
import pandas as pd
import requests
import trafilatura
from bs4 import BeautifulSoup
from newspaper import Article
from readability import Document
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
LOGGER = logging.getLogger(__name__)
USER_AGENT = "AI-News-Research-Bot/1.0"
TRACKING_PARAMS = {
    "fbclid", "gclid", "mc_cid", "mc_eid", "ref", "source", "spm",
    "utm_campaign", "utm_content", "utm_medium", "utm_source", "utm_term",
}
AI_TERMS = {
    "ai", "artificial intelligence", "machine learning", "deep learning",
    "foundation model", "large language model", "llm", "generative ai",
    "genai", "neural network", "computer vision", "robotics", "agentic",
    "chatbot", "inference", "training compute", "ai accelerator", "gpu",
    "semiconductor", "chip export", "data center", "datacenter", "openai",
    "anthropic", "deepmind", "hugging face", "nvidia", "copilot",
}


@dataclass(frozen=True)
class FeedSource:
    name: str
    url: str
    category: str
    filter_for_ai: bool = False


FEED_SOURCES = (
    FeedSource("OpenAI", "https://openai.com/news/rss.xml", "lab"),
    FeedSource("Google AI", "https://blog.google/technology/ai/rss/", "lab"),
    FeedSource("Google DeepMind", "https://deepmind.google/blog/rss.xml", "research"),
    FeedSource("Google Research", "https://research.google/blog/rss/", "research", True),
    FeedSource("AWS Machine Learning", "https://aws.amazon.com/blogs/machine-learning/feed/", "cloud"),
    FeedSource("Apple Machine Learning", "https://machinelearning.apple.com/rss.xml", "research"),
    FeedSource("Mozilla AI", "https://blog.mozilla.ai/rss/", "open-source"),
    FeedSource("Microsoft Research", "https://www.microsoft.com/en-us/research/feed/", "research", True),
    FeedSource("Hugging Face", "https://huggingface.co/blog/feed.xml", "open-source"),
    FeedSource("NVIDIA Blog", "https://blogs.nvidia.com/feed/", "compute", True),
    FeedSource("NVIDIA Developer", "https://developer.nvidia.com/blog/feed/", "compute", True),
    FeedSource("MIT News AI", "https://news.mit.edu/rss/topic/artificial-intelligence2", "research"),
    FeedSource("MIT Technology Review AI", "https://www.technologyreview.com/topic/artificial-intelligence/feed/", "analysis"),
    FeedSource("TechCrunch AI", "https://techcrunch.com/category/artificial-intelligence/feed/", "business"),
    FeedSource("VentureBeat AI", "https://venturebeat.com/category/ai/feed/", "business"),
    FeedSource("The Verge AI", "https://www.theverge.com/rss/ai-artificial-intelligence/index.xml", "industry"),
    FeedSource("WIRED AI", "https://www.wired.com/feed/tag/ai/latest/rss", "industry"),
    FeedSource("Ars Technica AI", "https://arstechnica.com/ai/feed/", "industry"),
    FeedSource("MarkTechPost", "https://www.marktechpost.com/feed/", "research-news"),
    FeedSource("AI News", "https://www.artificialintelligence-news.com/feed/", "industry"),
    FeedSource("Last Week in AI", "https://lastweekin.ai/feed", "analysis"),
    FeedSource("Berkeley AI Research", "https://bair.berkeley.edu/blog/feed.xml", "research"),
    FeedSource("arXiv cs.AI", "https://export.arxiv.org/rss/cs.AI", "papers"),
    FeedSource("arXiv cs.LG", "https://export.arxiv.org/rss/cs.LG", "papers"),
    FeedSource("ScienceDaily AI", "https://www.sciencedaily.com/rss/computers_math/artificial_intelligence.xml", "research-news"),
    FeedSource("Cloudflare AI", "https://blog.cloudflare.com/tag/ai/rss/", "compute"),
    FeedSource("Semiconductor Engineering", "https://semiengineering.com/feed/", "chips", True),
    FeedSource("Intel Newsroom", "https://newsroom.intel.com/feed", "chips", True),
    FeedSource("IEEE Spectrum AI", "https://spectrum.ieee.org/feeds/topic/artificial-intelligence.rss", "engineering"),
    FeedSource("NIST News", "https://www.nist.gov/news-events/news/rss.xml", "policy", True),
    FeedSource("Georgetown CSET", "https://cset.georgetown.edu/feed/", "competitiveness"),
    FeedSource("EU Digital Strategy", "https://digital-strategy.ec.europa.eu/en/rss.xml", "policy", True),
    FeedSource("GOV.UK AI News", "https://www.gov.uk/search/news-and-communications.atom?keywords=artificial+intelligence", "policy"),
    FeedSource("The Decoder", "https://the-decoder.com/feed/", "industry"),
    FeedSource("AI as Normal Technology", "https://www.normaltech.ai/feed", "policy-analysis"),
)

_thread_local = threading.local()
_host_locks: dict[str, threading.Lock] = {}
_host_last_request: dict[str, float] = {}
_host_guard = threading.Lock()


def get_session() -> requests.Session:
    session = getattr(_thread_local, "session", None)
    if session is None:
        retry = Retry(
            total=1,
            connect=1,
            read=1,
            backoff_factor=0.2,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset({"GET"}),
            respect_retry_after_header=True,
        )
        session = requests.Session()
        session.headers.update({"User-Agent": USER_AGENT, "Accept-Language": "en-US,en;q=0.8"})
        session.mount("https://", HTTPAdapter(max_retries=retry, pool_connections=20, pool_maxsize=20))
        session.mount("http://", HTTPAdapter(max_retries=retry, pool_connections=20, pool_maxsize=20))
        _thread_local.session = session
    return session


def normalize_url(url: str) -> str:
    """Canonicalize a URL for deterministic link-level deduplication."""
    if not url:
        return ""
    parsed = urlparse(url.strip())
    query = [(k, v) for k, v in parse_qsl(parsed.query, keep_blank_values=True)
             if k.lower() not in TRACKING_PARAMS]
    path = re.sub(r"/{2,}", "/", parsed.path or "/")
    if path != "/":
        path = path.rstrip("/")
    return urlunparse((parsed.scheme.lower(), parsed.netloc.lower(), path, "", urlencode(query), ""))


def normalized_title(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (title or "").lower()).strip()


def strip_html(value: str) -> str:
    if not value:
        return ""
    return re.sub(r"\s+", " ", BeautifulSoup(value, "lxml").get_text(" ", strip=True)).strip()


def is_ai_relevant(title: str, description: str = "") -> bool:
    haystack = f" {normalized_title(title)} {normalized_title(description)} "
    return any(re.search(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", haystack)
               for term in AI_TERMS)


def record_quality(record: dict[str, Any]) -> tuple[int, int]:
    return (len(str(record.get("text", ""))), len(str(record.get("description", ""))))


def deduplicate_records(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deduplicate by canonical URL, then normalized title; keep richer copy."""
    by_key: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    title_to_key: dict[str, str] = {}
    for original in records:
        record = dict(original)
        canonical = normalize_url(str(record.get("link", "")))
        if canonical:
            record["link"] = canonical
        title_key = normalized_title(str(record.get("title", "")))
        key = canonical or f"title:{title_key}"
        existing_key = title_to_key.get(title_key) if title_key else None
        if existing_key:
            key = existing_key
        if key not in by_key:
            by_key[key] = record
            order.append(key)
            if title_key:
                title_to_key[title_key] = key
        elif record_quality(record) > record_quality(by_key[key]):
            by_key[key] = record
    return [by_key[key] for key in order]


def merge_records(existing: Iterable[dict[str, Any]], fresh: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return deduplicate_records([*existing, *fresh])


def _throttled_get(url: str, timeout: int, min_host_interval: float) -> requests.Response:
    host = urlparse(url).netloc.lower()
    with _host_guard:
        lock = _host_locks.setdefault(host, threading.Lock())
    with lock:
        elapsed = time.monotonic() - _host_last_request.get(host, 0.0)
        if elapsed < min_host_interval:
            time.sleep(min_host_interval - elapsed)
        response = get_session().get(url, timeout=timeout, allow_redirects=True)
        _host_last_request[host] = time.monotonic()
    response.raise_for_status()
    return response


def extract_article_text(url: str, timeout: int = 8, min_host_interval: float = 0.2) -> str:
    """Fetch once and try several extraction algorithms without duplicate downloads."""
    try:
        response = _throttled_get(url, timeout, min_host_interval)
        html = response.text or ""
        if not html:
            return ""

        text = trafilatura.extract(
            html, url=response.url, include_comments=False, include_images=False,
            include_tables=False, favor_precision=True,
        )
        if text and len(text.split()) >= 50:
            return re.sub(r"\s+", " ", text).strip()

        try:
            summary = Document(html).summary(html_partial=True)
            text = strip_html(summary)
            if len(text.split()) >= 50:
                return text
        except Exception as exc:  # noqa: BLE001 - optional extractor fallback
            LOGGER.debug("Readability extraction failed for %s: %s", url, exc)

        try:
            article = Article(response.url)
            article.set_html(html)
            article.parse()
            if article.text and len(article.text.split()) >= 50:
                return re.sub(r"\s+", " ", article.text).strip()
        except Exception as exc:  # noqa: BLE001 - optional extractor fallback
            LOGGER.debug("Newspaper extraction failed for %s: %s", url, exc)

        soup = BeautifulSoup(html, "lxml")
        for tag in soup(["script", "style", "noscript", "nav", "header", "footer", "aside"]):
            tag.decompose()
        return re.sub(r"\s+", " ", " ".join(p.get_text(" ", strip=True) for p in soup.find_all("p"))).strip()
    except requests.RequestException as exc:
        LOGGER.debug("Article request failed %s: %s", url, exc)
        return ""


def entry_value(entry: Any, key: str, default: str = "") -> str:
    value = entry.get(key, default) if hasattr(entry, "get") else getattr(entry, key, default)
    return str(value or "").strip()


def feed_embedded_text(entry: Any) -> str:
    parts: list[str] = []
    for item in entry.get("content", []) if hasattr(entry, "get") else []:
        if isinstance(item, dict) and item.get("value"):
            parts.append(strip_html(str(item["value"])))
    summary = entry_value(entry, "summary") or entry_value(entry, "description")
    if summary:
        parts.append(strip_html(summary))
    return max(parts, key=len, default="")


def parse_feed(source: FeedSource, max_items: int, timeout: int) -> list[dict[str, Any]]:
    response = _throttled_get(source.url, timeout, 0.1)
    parsed = feedparser.parse(response.content)
    if parsed.bozo and not parsed.entries:
        raise ValueError(f"invalid feed: {parsed.get('bozo_exception', 'unknown parse error')}")
    records: list[dict[str, Any]] = []
    for entry in parsed.entries[:max_items]:
        title = entry_value(entry, "title")
        link = normalize_url(entry_value(entry, "link"))
        description = strip_html(entry_value(entry, "summary") or entry_value(entry, "description"))
        if not title or not link:
            continue
        if source.filter_for_ai and not is_ai_relevant(title, description):
            continue
        embedded = feed_embedded_text(entry)
        authors = entry_value(entry, "author")
        if not authors and entry.get("authors"):
            authors = ", ".join(str(a.get("name", "")) for a in entry.authors if a.get("name"))
        records.append({
            "source": source.name,
            "category": source.category,
            "title": title,
            "link": link,
            "description": description,
            "published": entry_value(entry, "published") or entry_value(entry, "updated"),
            "entry_id": entry_value(entry, "id") or hashlib.sha1(link.encode()).hexdigest(),
            "authors": authors,
            "domain": urlparse(link).netloc,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "text": embedded,
        })
    return records


def scrape_news(
    sources: Iterable[FeedSource] = FEED_SOURCES,
    max_items_per_source: int = 50,
    workers: int = 12,
    timeout: int = 8,
    fetch_full_text: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Fetch feeds and article bodies, returning records and per-source failures."""
    records: list[dict[str, Any]] = []
    failures: dict[str, str] = {}
    source_list = list(sources)
    with ThreadPoolExecutor(max_workers=min(workers, len(source_list) or 1)) as pool:
        futures = {pool.submit(parse_feed, source, max_items_per_source, timeout): source for source in source_list}
        for future in as_completed(futures):
            source = futures[future]
            try:
                batch = future.result()
                records.extend(batch)
                LOGGER.info("%-28s %4d feed items", source.name, len(batch))
            except Exception as exc:  # noqa: BLE001 - isolate per-feed failures
                failures[source.name] = str(exc)
                LOGGER.warning("Feed failed: %s (%s)", source.name, exc)

    records = deduplicate_records(records)
    if fetch_full_text:
        missing = [r for r in records if len(str(r.get("text", "")).split()) < 80]
        LOGGER.info("Fetching full text for %d/%d records", len(missing), len(records))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(extract_article_text, r["link"], timeout): r for r in missing}
            for future in as_completed(futures):
                record = futures[future]
                text = future.result()
                if text:
                    record["text"] = text
    return records, failures


def latest_csv(data_dir: Path = DATA_DIR) -> Path | None:
    files = sorted(data_dir.glob("news_articles_combined_*.csv"))
    return files[-1] if files else None


def load_existing(path: Path | None) -> list[dict[str, Any]]:
    if not path or not path.exists():
        return []
    return pd.read_csv(path).fillna("").to_dict("records")


def save_records(records: list[dict[str, Any]], data_dir: Path = DATA_DIR) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output = data_dir / f"news_articles_combined_{timestamp}.csv"
    pd.DataFrame(records).to_csv(output, index=False, encoding="utf-8-sig")
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-per-source", type=int, default=50)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--timeout", type=int, default=8)
    parser.add_argument("--full-text", action="store_true", help="Fetch missing article bodies (slower)")
    parser.add_argument("--no-merge", action="store_true", help="Do not merge the newest existing CSV")
    parser.add_argument("--output-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    return parser


def main() -> int:
    args = build_parser().parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s %(message)s")
    fresh, failures = scrape_news(
        max_items_per_source=args.max_per_source,
        workers=args.workers,
        timeout=args.timeout,
        fetch_full_text=args.full_text,
    )
    previous = [] if args.no_merge else load_existing(latest_csv(args.output_dir))
    combined = merge_records(previous, fresh)
    output = save_records(combined, args.output_dir)
    print(f"Collected {len(fresh)} unique articles; saved {len(combined)} total to {output}")
    if failures:
        print(f"Feeds with errors ({len(failures)}): " + ", ".join(sorted(failures)))
    return 0 if fresh else 1


if __name__ == "__main__":
    raise SystemExit(main())
