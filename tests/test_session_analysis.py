"""Tests for session logging and analysis (run: python -m unittest discover tests)."""

import tempfile
import unittest
from pathlib import Path

from src.session import analyse
from src.session.session_logger import SessionLogger


def writeScriptedSession(path):
    """120 s at 10 Hz with known events:

    40-55 s   looking away (inattentive, left pose, gaze off-screen)
    80-90 s   no face (inattentive)
    100-105 s phone (inattentive)
    markers at 60 s and 100 s
    """

    with SessionLogger(path) as logger:
        for i in range(1200):
            t = i / 10

            away = 40 <= t < 55
            noFace = 80 <= t < 90
            phone = 100 <= t < 105
            inattentive = away or noFace or phone

            logger.log(
                t=t,
                attention=int(inattentive),
                faces=0 if noFace else 1,
                phone=phone,
                pose=None if noFace else ("left" if away else "forward"),
                gaze=None if noFace else (0.5, 0.5),
                offScreen=away,
                marker="section 2" if i == 600 else "section 3" if i == 1000 else ""
            )
        logger.finish()


class SessionAnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.csv = Path(cls.directory.name) / "session_test.csv"
        writeScriptedSession(cls.csv)
        cls.data = analyse.loadSession(cls.csv)
        cls.result = analyse.analyse(cls.data, window=30, threshold=0.7)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def test_overall_attention(self):
        overall = self.result["overall"]
        self.assertAlmostEqual(overall["seconds"], 120, delta=0.5)
        self.assertAlmostEqual(overall["attention"], 0.75, delta=0.01)

    def test_causes(self):
        causes = self.result["overall"]["causes"]
        self.assertAlmostEqual(causes["looking away"], 15, delta=0.5)
        self.assertAlmostEqual(causes["no face"], 10, delta=0.5)
        self.assertAlmostEqual(causes["phone"], 5, delta=0.5)

    def test_periods(self):
        periods = self.result["periods"]
        self.assertEqual(
            [p["cause"] for p in periods],
            ["looking away", "no face", "phone"]
        )
        self.assertAlmostEqual(periods[0]["start"], 40, delta=0.2)
        self.assertAlmostEqual(periods[0]["end"], 55, delta=0.2)

    def test_short_periods_are_ignored(self):
        periods = analyse.findInattentivePeriods(
            self.data,
            analyse.sampleDurations(self.data["t"]),
            minSeconds=8
        )
        self.assertEqual(len(periods), 2)

    def test_sections(self):
        sections = self.result["sections"]
        self.assertEqual(
            [s["name"] for s in sections],
            ["start", "section 2", "section 3"]
        )
        # 0-60 s has 15 s away; 60-100 has 10 s no face; 100-120 has 5 s phone
        self.assertAlmostEqual(sections[0]["attention"], 0.75, delta=0.01)
        self.assertAlmostEqual(sections[1]["attention"], 0.75, delta=0.01)
        self.assertAlmostEqual(sections[2]["attention"], 0.75, delta=0.01)

    def test_windows_flag_the_worst_period(self):
        windows = self.result["windows"]
        self.assertEqual(len(windows), 4)
        # 30-60 s contains the 15 s of looking away, so attention is 50%
        self.assertAlmostEqual(windows[1]["attention"], 0.5, delta=0.02)
        self.assertAlmostEqual(windows[0]["attention"], 1.0, delta=0.01)

    def test_gaze_off_screen_and_grid(self):
        # 110 s have gaze data; 15 s of those are off-screen
        self.assertAlmostEqual(
            self.result["overall"]["off_screen"], 15 / 110, delta=0.01
        )
        self.assertAlmostEqual(self.result["grid"][1][1], 1.0, delta=0.01)

    def test_report_and_chart(self):
        report = analyse.formatReport(self.result)
        self.assertIn("looking away", report)
        self.assertIn("LOW", report)

        chart = Path(self.directory.name) / "chart.png"
        self.assertTrue(analyse.saveTimeline(self.data, self.result, chart))
        self.assertTrue(chart.exists())

    def test_missing_values_are_handled(self):
        path = Path(self.directory.name) / "sparse.csv"
        with SessionLogger(path) as logger:
            for i in range(20):
                logger.log(t=i / 10, attention=None if i < 10 else 0)
        result = analyse.analyse(analyse.loadSession(path))
        self.assertAlmostEqual(result["overall"]["attention"], 1.0)
        self.assertIsNone(result["overall"]["off_screen"])
        self.assertIsNone(result["grid"])


if __name__ == "__main__":
    unittest.main()
