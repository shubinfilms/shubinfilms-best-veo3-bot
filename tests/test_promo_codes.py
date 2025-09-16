import os
import tempfile
import unittest
from unittest.mock import patch

from promo_codes import (
    DEFAULT_PROMO_CODES,
    iter_sorted_promo_codes,
    load_promo_codes,
    normalize_promo_code,
)


class PromoCodesTestCase(unittest.TestCase):
    def test_normalize_promo_code_removes_spaces_and_case(self) -> None:
        self.assertEqual(normalize_promo_code(" weLcome 50 "), "WELCOME50")
        self.assertEqual(normalize_promo_code("free\u200b10"), "FREE10")
        self.assertEqual(normalize_promo_code(""), "")

    def test_load_promo_codes_merges_sources(self) -> None:
        defaults = {"WELCOME50": 50, "FREE10": 10}
        with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8") as tmp:
            tmp.write("extra = 30\nwelcome50 = 70\ninvalid=oops\nother: 15\n")
            file_path = tmp.name

        try:
            with patch.dict(
                os.environ,
                {
                    "PROMO_CODES_FILE": file_path,
                    "PROMO_CODES": "bonus25=25",
                    "PROMO_CODES_JSON": "",
                },
                clear=False,
            ):
                codes = load_promo_codes(defaults)
        finally:
            os.unlink(file_path)

        self.assertEqual(codes["WELCOME50"], 70)  # overridden by file
        self.assertEqual(codes["FREE10"], 10)     # from defaults
        self.assertEqual(codes["EXTRA"], 30)      # from file
        self.assertEqual(codes["OTHER"], 15)      # from file with ':'
        self.assertEqual(codes["BONUS25"], 25)    # from env variable
        self.assertNotIn("INVALID", codes)        # invalid amount is skipped

    def test_load_promo_codes_honours_json_env(self) -> None:
        with patch.dict(
            os.environ,
            {
                "PROMO_CODES_JSON": '{"bonus50": 80, "friends150": 150}',
                "PROMO_CODES": "",
                "PROMO_CODES_FILE": "",
            },
            clear=False,
        ):
            codes = load_promo_codes(DEFAULT_PROMO_CODES)

        self.assertEqual(codes["BONUS50"], 80)
        self.assertEqual(codes["FRIENDS150"], 150)
        self.assertIn(("BONUS50", 80), iter_sorted_promo_codes(codes))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()

