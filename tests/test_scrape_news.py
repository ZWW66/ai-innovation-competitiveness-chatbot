import unittest

from news_ingestion.scrape_news import (
    build_parser,
    deduplicate_records,
    is_ai_relevant,
    merge_records,
    normalize_url,
)


class ScraperUtilityTests(unittest.TestCase):
    def test_cli_defaults_prioritize_fast_feed_only_scraping(self):
        args = build_parser().parse_args([])
        self.assertEqual(args.timeout, 8)
        self.assertFalse(args.full_text)
        self.assertEqual(args.workers, 12)

    def test_full_text_is_explicit_opt_in(self):
        args = build_parser().parse_args(["--full-text"])
        self.assertTrue(args.full_text)

    def test_normalize_url_removes_tracking_and_fragment(self):
        url = "https://Example.com/news/ai/?utm_source=rss&ref=home&id=7#section"
        self.assertEqual(normalize_url(url), "https://example.com/news/ai?id=7")

    def test_ai_relevance_accepts_ai_terms_in_title_or_description(self):
        self.assertTrue(is_ai_relevant("New foundation model released", "Benchmark results"))
        self.assertTrue(is_ai_relevant("New chip export rules", "Restrictions target AI accelerators"))

    def test_ai_relevance_rejects_unrelated_general_news(self):
        self.assertFalse(is_ai_relevant("Local football team wins", "Match report and scores"))

    def test_deduplicate_records_prefers_record_with_more_text(self):
        short = {"link": "https://example.com/a?utm_source=x", "title": "AI news", "text": "short"}
        long = {"link": "https://example.com/a", "title": "AI news", "text": "a much longer article body"}
        result = deduplicate_records([short, long])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["text"], long["text"])

    def test_deduplicate_records_uses_title_when_links_differ(self):
        records = [
            {"link": "https://wire.example/a", "title": "OpenAI launches a new model", "text": "first"},
            {"link": "https://syndication.example/b", "title": "OpenAI launches a new model!", "text": "longer syndicated copy"},
        ]
        result = deduplicate_records(records)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["text"], "longer syndicated copy")

    def test_merge_records_keeps_existing_and_adds_only_new_items(self):
        existing = [{"link": "https://example.com/a", "title": "A", "text": "old"}]
        fresh = [
            {"link": "https://example.com/a?utm_campaign=rss", "title": "A", "text": "new longer text"},
            {"link": "https://example.com/b", "title": "B", "text": "brand new"},
        ]
        merged = merge_records(existing, fresh)
        self.assertEqual(len(merged), 2)
        self.assertEqual(merged[0]["text"], "new longer text")


if __name__ == "__main__":
    unittest.main()
