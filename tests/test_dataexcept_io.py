"""Offline tests for DataExcept at the cache and analyst CSV boundaries."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from dataexcept import DataLoadingError, FileReadError, FileWriteError

from ptgdp import config, fetch, import_content


class DatasetIOErrorsTest(unittest.TestCase):
    """Keep the path and original cause when source I/O fails."""

    def test_missing_import_content_csv(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "missing.csv"
            with self.assertRaises(FileReadError) as caught:
                import_content.load_import_content(path)

            self.assertEqual(caught.exception.path, str(path))
            self.assertIsInstance(caught.exception.original, FileNotFoundError)
            self.assertIs(caught.exception.__cause__, caught.exception.original)

    def test_invalid_import_content_csv_parser(self) -> None:
        path = Path("broken.csv")
        failure = pd.errors.ParserError("bad CSV row")
        with patch.object(import_content.pd, "read_csv", side_effect=failure):
            with self.assertRaises(DataLoadingError) as caught:
                import_content.load_import_content(path)

        self.assertEqual(caught.exception.source, str(path))
        self.assertIs(caught.exception.original, failure)
        self.assertIs(caught.exception.__cause__, failure)

    def test_invalid_import_content_columns_still_raise_value_error(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "shares.csv"
            path.write_text("component,wrong_column\ngfcf,0.34\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "import_share"):
                import_content.load_import_content(path)

    def test_valid_import_content_overrides_one_share(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "shares.csv"
            path.write_text("component,import_share\ngfcf,0.42\n", encoding="utf-8")
            shares = import_content.load_import_content(path)

        self.assertEqual(shares["gfcf"], 0.42)
        self.assertEqual(
            shares["exports"], import_content.DEFAULT_IMPORT_CONTENT["exports"]
        )

    def test_cache_round_trip_does_not_fetch_twice(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            frame = pd.DataFrame(
                {"GDP": [1.0, 2.0]}, index=pd.period_range("2020Q1", periods=2, freq="Q")
            )
            with (
                patch.object(config, "DATA_DIR", directory),
                patch.object(fetch, "fetch_unit", return_value=frame) as fetch_unit,
            ):
                first = fetch.load(config.UNIT_CP)
                second = fetch.load(config.UNIT_CP)

            fetch_unit.assert_called_once_with(config.UNIT_CP)
            pd.testing.assert_frame_equal(first, frame)
            pd.testing.assert_frame_equal(second, frame)

    def test_cache_read_failure_does_not_trigger_fetch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            cache = directory / f"namq_10_gdp_{config.GEO}_{config.UNIT_CP}.parquet"
            cache.touch()
            failure = PermissionError("cache unreadable")
            with (
                patch.object(config, "DATA_DIR", directory),
                patch.object(fetch.pd, "read_parquet", side_effect=failure),
                patch.object(fetch, "fetch_unit") as fetch_unit,
            ):
                with self.assertRaises(FileReadError) as caught:
                    fetch.load(config.UNIT_CP)

            fetch_unit.assert_not_called()
            self.assertEqual(caught.exception.path, str(cache))
            self.assertIs(caught.exception.original, failure)
            self.assertIs(caught.exception.__cause__, failure)

    def test_cache_decode_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            cache = directory / f"namq_10_gdp_{config.GEO}_{config.UNIT_CP}.parquet"
            cache.touch()
            failure = ValueError("invalid parquet")
            with (
                patch.object(config, "DATA_DIR", directory),
                patch.object(fetch.pd, "read_parquet", side_effect=failure),
            ):
                with self.assertRaises(DataLoadingError) as caught:
                    fetch.load(config.UNIT_CP)

            self.assertEqual(caught.exception.source, str(cache))
            self.assertIs(caught.exception.original, failure)
            self.assertIs(caught.exception.__cause__, failure)

    def test_cache_write_failure_does_not_change_fetched_data(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            cache = directory / f"namq_10_gdp_{config.GEO}_{config.UNIT_CP}.parquet"
            frame = pd.DataFrame(
                {"GDP": [1.0]}, index=pd.period_range("2020Q1", periods=1, freq="Q")
            )
            failure = PermissionError("cache unwritable")
            with (
                patch.object(config, "DATA_DIR", directory),
                patch.object(fetch, "fetch_unit", return_value=frame),
                patch.object(pd.DataFrame, "to_parquet", side_effect=failure),
            ):
                with self.assertRaises(FileWriteError) as caught:
                    fetch.load(config.UNIT_CP)

            self.assertEqual(caught.exception.path, str(cache))
            self.assertIs(caught.exception.original, failure)
            self.assertIs(caught.exception.__cause__, failure)
            self.assertIsInstance(frame.index, pd.PeriodIndex)


if __name__ == "__main__":
    unittest.main()
