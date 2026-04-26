from __future__ import annotations

import os
import sys
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from extractor.ingestor.extract_file_cli import run_extract


class TestConcurrentRuntime(unittest.TestCase):
    def test_real_extract_runtime_compare(self) -> None:
        if os.getenv("RUN_REAL_RUNTIME_BENCH", "0") != "1":
            self.skipTest("未开启真实基准测试，设置 RUN_REAL_RUNTIME_BENCH=1 后再运行")

        file_rel = os.getenv(
            "BENCH_FILE",
            "files/AIGC时代的学术出版伦理：风险挑战与治理路径_葛建平.pdf",
        )
        repeats = max(1, int(os.getenv("BENCH_REPEATS", "2")))
        max_chunks = max(1, int(os.getenv("BENCH_MAX_CHUNKS", "6")))

        serial_costs = []
        parallel_costs = []

        for i in range(repeats):
            t1 = time.perf_counter()
            rc1 = run_extract(
                SimpleNamespace(
                    file=file_rel,
                    max_chunks=max_chunks,
                    section="runtime-bench",
                    time_version="2026",
                    output_dir="extractor/ingestor/output",
                    log_dir="extractor/ingestor/log",
                    workers=1,
                )
            )
            serial_costs.append(time.perf_counter() - t1)
            self.assertEqual(rc1, 0, f"workers=1 第{i+1}次失败")

            t4 = time.perf_counter()
            rc4 = run_extract(
                SimpleNamespace(
                    file=file_rel,
                    max_chunks=max_chunks,
                    section="runtime-bench",
                    time_version="2026",
                    output_dir="extractor/ingestor/output",
                    log_dir="extractor/ingestor/log",
                    workers=4,
                )
            )
            parallel_costs.append(time.perf_counter() - t4)
            self.assertEqual(rc4, 0, f"workers=4 第{i+1}次失败")

        serial_avg = sum(serial_costs) / len(serial_costs)
        parallel_avg = sum(parallel_costs) / len(parallel_costs)
        speedup = serial_avg / parallel_avg if parallel_avg > 0 else 0.0

        print(
            f"serial_avg={serial_avg:.2f}s, "
            f"parallel_avg={parallel_avg:.2f}s, "
            f"speedup={speedup:.2f}x"
        )

        self.assertGreater(serial_avg, 0.0)
        self.assertGreater(parallel_avg, 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
