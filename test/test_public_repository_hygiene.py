"""Offline checks for public documentation and visible project branding."""

from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]


class PublicRepositoryHygieneTests(unittest.TestCase):
    def test_documentation_avoids_personal_home_paths(self):
        files = [ROOT / "README.md", *(ROOT / "docs").rglob("*.md")]
        pattern = re.compile(r"/Users/[^\s/]+/|[A-Za-z]:[\\/]Users[\\/][^\s\\/]+")
        for path in files:
            with self.subTest(path=str(path.relative_to(ROOT))):
                self.assertIsNone(pattern.search(path.read_text(encoding="utf-8")))

    def test_architecture_resource_links_resolve(self):
        path = ROOT / "docs/architecture/ARCHITECTURE_DOCS_INDEX.md"
        for target in re.findall(r"\]\(([^)#]+)(?:#[^)]*)?\)", path.read_text(encoding="utf-8")):
            if "://" not in target:
                with self.subTest(target=target):
                    self.assertTrue((path.parent / target).exists())

    def test_visible_brand_is_knowflow(self):
        paths = ["web/index.html", "web/src/components/AuthModal.vue",
                 "web/src/components/MessageBubble.vue", "web/src/components/MessageList.vue",
                 "web/src/components/SideBar.vue"]
        for name in paths:
            with self.subTest(path=name):
                text = (ROOT / name).read_text(encoding="utf-8")
                self.assertIn("KnowFlow", text)
                self.assertNotIn("AGI-saber", text)

    def test_unrelated_screenshot_is_not_published(self):
        self.assertFalse((ROOT / "docs/code-tutor/image.png").exists())


if __name__ == "__main__":
    unittest.main()
