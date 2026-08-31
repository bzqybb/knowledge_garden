use std::fs::{self, OpenOptions};
use std::io::Write;
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::Mutex;
use std::time::Duration;
use tauri::{Manager, RunEvent, State};
use tauri_plugin_shell::process::{CommandChild, CommandEvent};
use tauri_plugin_shell::ShellExt;

const DESKTOP_INSTANCE_ID: &str = "knowledge-garden-desktop-v2";
const PUBLIC_BETA_CLOUD_URL: &str = match option_env!("GARDEN_BETA_CLOUD_URL") {
    Some(url) => url,
    None => "",
};

struct RuntimeState {
    garden: Mutex<Option<CommandChild>>,
    tracememo: Mutex<Option<Child>>,
    backend_url: String,
}

#[tauri::command]
fn backend_url(state: State<'_, RuntimeState>) -> String {
    state.backend_url.clone()
}

fn available_backend_port() -> Result<u16, String> {
    for port in 8765..8795 {
        if TcpListener::bind(("127.0.0.1", port)).is_ok() {
            return Ok(port);
        }
    }
    Err("找不到可用的本地服务端口（8765-8794 均被占用）".to_string())
}

fn write_sidecar_events(
    mut events: tauri::async_runtime::Receiver<CommandEvent>,
    log_path: PathBuf,
) {
    tauri::async_runtime::spawn(async move {
        let mut log = match OpenOptions::new().create(true).append(true).open(log_path) {
            Ok(file) => file,
            Err(_) => return,
        };
        while let Some(event) = events.recv().await {
            let line = match event {
                CommandEvent::Stdout(bytes) => String::from_utf8_lossy(&bytes).into_owned(),
                CommandEvent::Stderr(bytes) => String::from_utf8_lossy(&bytes).into_owned(),
                CommandEvent::Error(error) => format!("sidecar error: {error}"),
                CommandEvent::Terminated(payload) => format!("sidecar terminated: {payload:?}"),
                _ => continue,
            };
            let _ = writeln!(log, "{}", line.trim_end());
            let _ = log.flush();
        }
    });
}

fn append_sidecar_diagnostic(log_path: &Path, message: &str) {
    if let Ok(mut log) = OpenOptions::new().create(true).append(true).open(log_path) {
        let _ = writeln!(log, "{message}");
        let _ = log.flush();
    }
}

fn bundled_resource_root(resource_dir: &Path, name: &str) -> PathBuf {
    let prefixed = resource_dir.join("resources").join(name);
    if prefixed.exists() {
        prefixed
    } else {
        resource_dir.join(name)
    }
}

#[cfg(target_os = "windows")]
fn registered_tracememo_executable() -> Option<PathBuf> {
    use std::os::windows::process::CommandExt;

    let mut command = Command::new("reg.exe");
    command
        .args([
            "query",
            r"HKCU\Software\Microsoft\Windows\CurrentVersion\Uninstall",
            "/s",
            "/f",
            "TraceMemo",
            "/d",
        ])
        .creation_flags(0x08000000);
    let output = command.output().ok()?;
    let stdout = String::from_utf8_lossy(&output.stdout);
    stdout.lines().find_map(|line| {
        let (_, value) = line.split_once("REG_SZ")?;
        let value = value.trim();
        let uninstall = if let Some(quoted) = value.strip_prefix('"') {
            quoted.split('"').next().unwrap_or(quoted)
        } else {
            value.split(" /").next().unwrap_or(value)
        };
        let path = PathBuf::from(uninstall);
        let file_name = path.file_name()?.to_string_lossy();
        if file_name.eq_ignore_ascii_case("Uninstall TraceMemo.exe") {
            let candidate = path.parent()?.join("TraceMemo.exe");
            candidate.is_file().then_some(candidate)
        } else if file_name.eq_ignore_ascii_case("TraceMemo.exe") && path.is_file() {
            Some(path)
        } else {
            None
        }
    })
}

#[cfg(not(target_os = "windows"))]
fn registered_tracememo_executable() -> Option<PathBuf> {
    None
}

fn tracememo_executable(resource_dir: &Path) -> Option<PathBuf> {
    let configured = std::env::var_os("GARDEN_TRACEMEMO_EXE").map(PathBuf::from);
    let registered = registered_tracememo_executable();
    let local_app_data = std::env::var_os("LOCALAPPDATA").map(PathBuf::from);
    let program_files = std::env::var_os("ProgramFiles").map(PathBuf::from);
    configured
        .into_iter()
        .chain(registered)
        .chain([
            resource_dir.join("resources/tracememo/TraceMemo.exe"),
            resource_dir.join("tracememo/TraceMemo.exe"),
        ])
        .chain(local_app_data.iter().flat_map(|root| [
            root.join("Programs/TraceMemo/TraceMemo.exe"),
            root.join("Programs/tracememo/TraceMemo.exe"),
            root.join("TraceMemo/TraceMemo.exe"),
        ]))
        .chain(program_files.iter().map(|root| root.join("TraceMemo/TraceMemo.exe")))
        .find(|candidate| candidate.is_file())
}

fn tracememo_api_available() -> bool {
    let Ok(address) = "127.0.0.1:6131".parse::<SocketAddr>() else {
        return false;
    };
    TcpStream::connect_timeout(&address, Duration::from_millis(400)).is_ok()
}

fn start_tracememo(resource_dir: &Path, data_dir: &Path) -> Option<Child> {
    // TraceMemo exposes one machine-local API on port 6131. Reuse an existing
    // healthy instance instead of opening a second Electron process, which
    // otherwise fails its API startup and competes for the same disk cache.
    if tracememo_api_available() {
        return None;
    }
    let mut executable = tracememo_executable(resource_dir);
    if executable.is_none() {
        let installers = [
            resource_dir.join("resources/tracememo/TraceMemo-setup.exe"),
            resource_dir.join("tracememo/TraceMemo-setup.exe"),
        ];
        if let Some(installer) = installers.iter().find(|candidate| candidate.is_file()) {
            let mut install_command = Command::new(installer);
            install_command
                .arg("/S")
                .stdin(Stdio::null())
                .stdout(Stdio::null())
                .stderr(Stdio::null());
            #[cfg(target_os = "windows")]
            {
                use std::os::windows::process::CommandExt;
                install_command.creation_flags(0x08000000);
            }
            let _ = install_command.status();
            executable = tracememo_executable(resource_dir);
        }
    }
    let executable = executable?;
    let mut command = Command::new(executable);
    let temp_dir = data_dir.join("tmp").join("tracememo");
    let log_dir = data_dir.join("logs");
    let _ = fs::create_dir_all(&temp_dir);
    let _ = fs::create_dir_all(&log_dir);
    let trace_log = OpenOptions::new()
        .create(true)
        .append(true)
        .open(log_dir.join("tracememo.log"))
        .ok();
    let trace_error_log = trace_log.as_ref().and_then(|log| log.try_clone().ok());
    command
        .arg("--tray")
        .env("WXE_TRAY", "1")
        .env("TEMP", &temp_dir)
        .env("TMP", &temp_dir)
        .env("TMPDIR", &temp_dir)
        .stdin(Stdio::null());
    if let Some(log) = trace_log {
        command.stdout(Stdio::from(log));
    } else {
        command.stdout(Stdio::null());
    }
    if let Some(log) = trace_error_log {
        command.stderr(Stdio::from(log));
    } else {
        command.stderr(Stdio::null());
    }
    #[cfg(target_os = "windows")]
    {
        use std::os::windows::process::CommandExt;
        command.creation_flags(0x08000000);
    }
    command.spawn().ok()
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let app = tauri::Builder::default()
        .plugin(tauri_plugin_process::init())
        .plugin(tauri_plugin_shell::init())
        .invoke_handler(tauri::generate_handler![backend_url])
        .setup(|app| {
            #[cfg(desktop)]
            app.handle().plugin(tauri_plugin_updater::Builder::new().build())?;
            let configured_data_dir = std::env::var_os("GARDEN_DATA_DIR").map(PathBuf::from);
            #[cfg(target_os = "windows")]
            let d_drive_data_dir = {
                let drive = PathBuf::from(r"D:\");
                drive.exists().then(|| drive.join("KnowledgeGarden").join("data"))
            };
            #[cfg(not(target_os = "windows"))]
            let d_drive_data_dir: Option<PathBuf> = None;
            let data_dir = configured_data_dir
                .or(d_drive_data_dir)
                .unwrap_or(app.path().app_data_dir()?);
            let resource_dir = app.path().resource_dir()?;
            let bilibili_root = bundled_resource_root(&resource_dir, "bilibili");
            fs::create_dir_all(&data_dir)?;
            let temp_dir = data_dir.join("tmp");
            let cache_dir = data_dir.join("cache");
            let model_dir = data_dir.join("models");
            let log_dir = data_dir.join("logs");
            fs::create_dir_all(&temp_dir)?;
            fs::create_dir_all(&cache_dir)?;
            fs::create_dir_all(&model_dir)?;
            fs::create_dir_all(&log_dir)?;
            let backend_port = available_backend_port().map_err(std::io::Error::other)?;
            let backend_port_arg = backend_port.to_string();
            let backend_url = format!("http://127.0.0.1:{backend_port}/");
            let sidecar = app
                .shell()
                .sidecar("knowledge-garden-sidecar")?
                .args(["--host", "127.0.0.1", "--port", &backend_port_arg])
                .env("GARDEN_DATA_DIR", data_dir.to_string_lossy().to_string())
                .env("GARDEN_TEMP_DIR", temp_dir.to_string_lossy().to_string())
                .env("GARDEN_CACHE_DIR", cache_dir.to_string_lossy().to_string())
                .env("GARDEN_MODEL_CACHE_DIR", model_dir.to_string_lossy().to_string())
                .env("TEMP", temp_dir.to_string_lossy().to_string())
                .env("TMP", temp_dir.to_string_lossy().to_string())
                .env("TMPDIR", temp_dir.to_string_lossy().to_string())
                .env("PYTHONUNBUFFERED", "1")
                .env("PYTHONIOENCODING", "utf-8")
                .env("XDG_CACHE_HOME", cache_dir.to_string_lossy().to_string())
                .env("HF_HOME", model_dir.join("huggingface").to_string_lossy().to_string())
                .env("PIP_CACHE_DIR", cache_dir.join("pip").to_string_lossy().to_string())
                .env(
                    "GARDEN_NODE_EXE",
                    bilibili_root.join("node/node.exe").to_string_lossy().to_string(),
                )
                .env(
                    "GARDEN_BILIBILI_MCP_ROOT",
                    bilibili_root
                        .join("runtime")
                        .to_string_lossy()
                        .to_string(),
                )
                .env("GARDEN_DESKTOP_INSTANCE_ID", DESKTOP_INSTANCE_ID)
                .env("GARDEN_DESKTOP_PARENT_PID", std::process::id().to_string())
                .env("GARDEN_RELEASE_VERSION", concat!("desktop-", env!("CARGO_PKG_VERSION")))
                .env("GARDEN_BETA_MODE", "true")
                .env("GARDEN_BETA_CLOUD_URL", PUBLIC_BETA_CLOUD_URL)
                .env("GARDEN_BETA_MODEL", "glm-5.2")
                .env("GARDEN_DISABLE_SAVED_API_KEY", "1")
                .env("GARDEN_AUTH_REQUIRED", "false");
            let sidecar_log = log_dir.join("sidecar.log");
            append_sidecar_diagnostic(
                &sidecar_log,
                &format!(
                    "desktop={} launching sidecar at {}",
                    env!("CARGO_PKG_VERSION"),
                    backend_url
                ),
            );
            let (events, child) = sidecar.spawn()?;
            append_sidecar_diagnostic(&sidecar_log, "sidecar process spawned; waiting for health check");
            write_sidecar_events(events, sidecar_log);
            let tracememo = start_tracememo(&resource_dir, &data_dir);
            app.manage(RuntimeState {
                garden: Mutex::new(Some(child)),
                tracememo: Mutex::new(tracememo),
                backend_url,
            });
            Ok(())
        })
        .build(tauri::generate_context!())
        .expect("failed to build Knowledge Garden desktop companion");

    app.run(|handle, event| {
        if matches!(event, RunEvent::ExitRequested { .. }) {
            if let Some(state) = handle.try_state::<RuntimeState>() {
                if let Ok(mut guard) = state.garden.lock() {
                    if let Some(child) = guard.take() {
                        let _ = child.kill();
                    }
                }
                if let Ok(mut guard) = state.tracememo.lock() {
                    if let Some(mut child) = guard.take() {
                        let _ = child.kill();
                    }
                }
            }
        }
    });
}
