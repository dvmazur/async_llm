import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
from robot_sim.candidate_nav.hardware_profile import choose_devices, settings, discover, PROFILES


class HardwareProfileTests(unittest.TestCase):
    def device(self, index=0, **kw):
        return dict(
            dict(
                index=index,
                uuid=f"gpu-{index}",
                name="NVIDIA H100",
                capability=9.0,
                total_mib=81559,
                free_mib=81000,
                busy=False,
            ),
            **kw,
        )

    def test_auto_chooses_one_free_h100_and_never_shares_busy_gpu(self):
        a = self.device(0)
        a["busy"] = True
        chosen = choose_devices("h100", "auto", [a, self.device(1)])
        self.assertEqual([d["index"] for d in chosen], [1])
        with self.assertRaises(RuntimeError):
            choose_devices("h100", "0", [a, self.device(1)])

    def test_busy_or_small_or_wrong_arch_cards_are_rejected(self):
        for change in (
            {"busy": True},
            {"free_mib": 59000},
            {"capability": 8.0},
            {"total_mib": 40000},
        ):
            d = self.device()
            d.update(change)
            with self.subTest(change=change), self.assertRaises(RuntimeError):
                choose_devices("h100", "auto", [d])

    def test_legacy_retains_two_card_memory_requirement(self):
        a = self.device(0)
        a.update(name="RTX 3090", total_mib=24576, free_mib=23000, capability=8.6)
        b = dict(a, index=1)
        self.assertEqual(len(choose_devices("legacy", "auto", [a, b])), 2)
        b["free_mib"] = 21000
        with self.assertRaises(RuntimeError):
            choose_devices("legacy", "auto", [a, b])

    def test_defaults_and_h100_configuration(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self.assertEqual(settings(root)["profile"], "legacy")
            folder = root / "robot_sim/.runtime"
            folder.mkdir(parents=True)
            (folder / "live_model.json").write_text(json.dumps({"profile": "h100", "gpu": "auto"}))
            self.assertEqual(settings(root)["session"], "rover-live-qwen-h100")
        self.assertEqual(PROFILES["h100"]["tp_size"], 1)
        self.assertEqual(PROFILES["legacy"]["prefill_rows"], 256)

    def test_discovery_is_nvml_cli_only_and_does_not_import_torch(self):
        results = [
            SimpleNamespace(stdout="0, GPU-a, NVIDIA H100, 78000, 81559, 9.0\n"),
            SimpleNamespace(stdout="GPU-a, 1234\n"),
        ]
        with patch("subprocess.run", side_effect=results) as call:
            result = discover()
        self.assertTrue(result[0]["busy"])
        self.assertEqual(call.call_count, 2)


    def test_sharing_requires_explicit_authorized_id_and_enough_free_memory(self):
        busy=self.device(0,busy=True,free_mib=55000)
        self.assertEqual(choose_devices('h100','0',[busy],shared_gpu_ids=(0,))[0]['index'],0)
        for ids,free in [((1,),55000),((0,),44000),((),55000)]:
            with self.assertRaises(RuntimeError):
                choose_devices('h100','0',[dict(busy,free_mib=free)],shared_gpu_ids=ids)
        with self.assertRaises(ValueError):choose_devices('h100','0',[busy],shared_gpu_ids=(False,))

if __name__ == "__main__":
    unittest.main()
