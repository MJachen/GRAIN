from pathlib import Path
import tempfile
import unittest

from grain.data.paths import PathPolicyError, RepositoryPathPolicy


class PathPolicyTests(unittest.TestCase):
    def test_legacy_source_cannot_be_a_write_target(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repository = root / "GRAIN_official"
            source = root / "attention"
            repository.mkdir()
            source.mkdir()
            policy = RepositoryPathPolicy.create(repository, source)
            with self.assertRaises(PathPolicyError):
                policy.resolve_write(source / "data" / "result.csv")

    def test_only_official_artifact_roots_are_writable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repository = root / "GRAIN_official"
            source = root / "attention"
            repository.mkdir()
            source.mkdir()
            policy = RepositoryPathPolicy.create(repository, source)
            self.assertEqual(
                policy.resolve_write("artifacts/report.json"),
                (repository / "artifacts" / "report.json").resolve(),
            )
            with self.assertRaises(PathPolicyError):
                policy.resolve_write("unexpected/result.json")


if __name__ == "__main__":
    unittest.main()

