import os
import unittest
from pathlib import Path

from core.config import CACHE_DIR, DATA_DIR, MODEL_CACHE_DIR, ROOT, TEMP_DIR


class StoragePolicyTests(unittest.TestCase):
    def test_python_runtime_storage_stays_below_project_data_by_default(self):
        expected_root = (ROOT / "data").resolve()
        self.assertEqual(DATA_DIR, expected_root)
        for path in (CACHE_DIR, MODEL_CACHE_DIR, TEMP_DIR):
            self.assertTrue(path.is_relative_to(expected_root), path)

    def test_project_process_cache_environment_uses_project_storage(self):
        for variable in (
            "TEMP", "TMP", "TMPDIR", "XDG_CACHE_HOME", "PIP_CACHE_DIR",
            "HF_HOME", "SENTENCE_TRANSFORMERS_HOME", "TORCH_HOME",
        ):
            value = Path(os.environ[variable]).resolve()
            self.assertTrue(value.is_relative_to(DATA_DIR), f"{variable}={value}")

    def test_desktop_prefers_d_drive_and_build_scripts_share_storage_bootstrap(self):
        rust = (ROOT / "desktop" / "src-tauri" / "src" / "lib.rs").read_text(encoding="utf-8")
        self.assertIn('let drive = PathBuf::from(', rust)
        self.assertIn('KnowledgeGarden', rust)
        self.assertIn('GARDEN_DATA_DIR', rust)
        for relative in (
            "bootstrap.ps1",
            "desktop/build_sidecar.ps1",
            "desktop/build_desktop.ps1",
            "desktop/prepare_bundled_components.ps1",
            "desktop/prepare_bilibili_runtime.ps1",
        ):
            source = (ROOT / relative).read_text(encoding="utf-8")
            self.assertIn("project_storage.ps1", source, relative)

    def test_desktop_uses_owned_dynamic_sidecar_and_persistent_diagnostics(self):
        rust = (ROOT / "desktop" / "src-tauri" / "src" / "lib.rs").read_text(encoding="utf-8")
        frontend = (ROOT / "desktop" / "src" / "main.js").read_text(encoding="utf-8")
        config = (ROOT / "desktop" / "src-tauri" / "tauri.conf.json").read_text(encoding="utf-8")
        self.assertIn("for port in 8765..8795", rust)
        self.assertIn('log_dir.join("sidecar.log")', rust)
        self.assertIn("append_sidecar_diagnostic", rust)
        self.assertIn("sidecar process spawned; waiting for health check", rust)
        self.assertIn("GARDEN_DESKTOP_INSTANCE_ID", rust)
        self.assertIn("GARDEN_DESKTOP_PARENT_PID", rust)
        self.assertIn('env("PYTHONUNBUFFERED", "1")', rust)
        self.assertIn('env("PYTHONIOENCODING", "utf-8")', rust)
        self.assertIn('invoke("backend_url")', frontend)
        self.assertIn("status.desktop_instance !== desktopInstanceId", frontend)
        self.assertIn("gardenStartupTimeoutSeconds = 180", frontend)
        self.assertIn("首次启动正在解压并初始化", frontend)
        self.assertIn("http://127.0.0.1:*", config)

    def test_desktop_sidecar_exits_when_its_owner_disappears(self):
        app_source = (ROOT / "app.py").read_text(encoding="utf-8")
        self.assertIn("def start_desktop_parent_watchdog", app_source)
        self.assertIn('os.getenv("GARDEN_DESKTOP_PARENT_PID"', app_source)
        self.assertIn("os._exit(0)", app_source)

    def test_desktop_reuses_an_existing_tracememo_api(self):
        rust = (ROOT / "desktop" / "src-tauri" / "src" / "lib.rs").read_text(encoding="utf-8")
        availability_check = rust.index("if tracememo_api_available()")
        process_spawn = rust.index("let mut command = Command::new(executable)")
        self.assertLess(availability_check, process_spawn)
        self.assertIn('"127.0.0.1:6131"', rust)
        self.assertIn("TcpStream::connect_timeout", rust)
        self.assertIn("return None", rust[availability_check:process_spawn])

    def test_desktop_starts_tracememo_in_background_tray_mode(self):
        rust = (ROOT / "desktop" / "src-tauri" / "src" / "lib.rs").read_text(encoding="utf-8")
        self.assertIn('.arg("--tray")', rust)
        self.assertIn('.env("WXE_TRAY", "1")', rust)
        self.assertIn('log_dir.join("tracememo.log")', rust)
        self.assertIn(".stdin(Stdio::null())", rust)
        self.assertIn("command.creation_flags(0x08000000)", rust)


if __name__ == "__main__":
    unittest.main()
