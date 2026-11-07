import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np

from vampirebci import (
    ExtractionError,
    _safe_extract_zip,
    decode_cell,
    extract_arrays,
    find_edfs,
)


def synthetic_data(target=9, selected=9):
    n = 100
    sb = np.zeros(n, dtype=int)
    dr = np.zeros(n, dtype=int)
    ct = np.zeros(n, dtype=int)
    st = np.zeros(n, dtype=int)
    sb[[10, 20]] = 1
    dr[30:] = 1
    ct[5:30] = target
    st[25:30] = selected
    return {
        "StimulusBegin": sb,
        "DisplayResults": dr,
        "CurrentTarget": ct,
        "SelectedTarget": st,
    }


class ExtractionTests(unittest.TestCase):
    def test_decode_cell_boundaries(self):
        self.assertEqual(decode_cell(1), (1, 1))
        self.assertEqual(decode_cell(8), (1, 8))
        self.assertEqual(decode_cell(9), (2, 1))
        self.assertEqual(decode_cell(64), (8, 8))

    def test_decode_rejects_invalid_cell(self):
        with self.assertRaises(ExtractionError):
            decode_cell(65)

    def test_decode_rejects_non_integer_cell_ids_without_truncating(self):
        for value in (1.5, 9.9, True, "9"):
            with self.subTest(value=value):
                with self.assertRaises(ExtractionError):
                    decode_cell(value)

    def test_extracts_trial_and_correctness(self):
        frame = extract_arrays(
            synthetic_data(), 10,
            filename="M_05_SE001_ADdiff_Test01.edf",
        )
        self.assertEqual(len(frame), 1)
        self.assertEqual(int(frame.loc[0, "Y"]), 1)
        self.assertAlmostEqual(float(frame.loc[0, "L_s"]), 2.0)

    def test_fractional_selected_target_is_rejected_not_truncated(self):
        data = synthetic_data()
        data["SelectedTarget"] = data["SelectedTarget"].astype(float)
        data["SelectedTarget"][25] = 9.5
        with self.assertRaisesRegex(ExtractionError, "SelectedTarget=.*integer cell ID"):
            extract_arrays(data, 10, filename="M_05_SE001_ADdiff_Test01.edf")

    def test_fractional_current_target_is_rejected_not_truncated(self):
        data = synthetic_data()
        data["CurrentTarget"] = data["CurrentTarget"].astype(float)
        data["CurrentTarget"][29] = 9.5
        with self.assertRaisesRegex(ExtractionError, "CurrentTarget=.*integer cell ID"):
            extract_arrays(data, 10, filename="M_05_SE001_ADdiff_Test01.edf")

    def test_missing_target_is_rejected_not_scored_incorrect(self):
        data = synthetic_data()
        data["CurrentTarget"][:] = 0
        with self.assertRaisesRegex(ExtractionError, "no nonzero target"):
            extract_arrays(data, 10, filename="M_05_SE001_ADdiff_Test01.edf")

    def test_multiple_selection_events_are_rejected(self):
        data = synthetic_data()
        data["SelectedTarget"][22] = 9
        with self.assertRaisesRegex(ExtractionError, "exactly one selection"):
            extract_arrays(data, 10, filename="M_05_SE001_ADdiff_Test01.edf")

    def test_channel_length_mismatch_is_rejected(self):
        data = synthetic_data()
        data["CurrentTarget"] = data["CurrentTarget"][:-1]
        with self.assertRaisesRegex(ExtractionError, "different sample counts"):
            extract_arrays(data, 10, filename="M_05_SE001_ADdiff_Test01.edf")

    def test_zip_path_traversal_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / "bad.zip"
            with zipfile.ZipFile(archive, "w") as zf:
                zf.writestr("../escaped.txt", "not allowed")
            with self.assertRaisesRegex(ExtractionError, "unsafe ZIP member"):
                _safe_extract_zip(archive, root)

    def test_find_edfs_does_not_extract_archives_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with zipfile.ZipFile(root / "data.zip", "w") as zf:
                zf.writestr("inside.edf", "placeholder")
            self.assertEqual(find_edfs(root), [])


if __name__ == "__main__":
    unittest.main()
