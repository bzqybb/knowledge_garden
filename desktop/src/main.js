import "./styles.css";
import { getVersion } from "@tauri-apps/api/app";
import { invoke } from "@tauri-apps/api/core";
import { check } from "@tauri-apps/plugin-updater";
import { relaunch } from "@tauri-apps/plugin-process";

const desktopInstanceId = "knowledge-garden-desktop-v2";
const gardenStartupTimeoutSeconds = 180;
let localGarden = "";
const $ = selector => document.querySelector(selector);
let pendingUpdate = null;

async function waitForGarden(attempt = 0) {
  try {
    const response = await fetch(`${localGarden}api/auth/status`, { cache: "no-store" });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const status = await response.json();
    if (status.desktop_instance && status.desktop_instance !== desktopInstanceId) {
      throw new Error("检测到的不是本次桌面伴侣启动的知识服务");
    }
    if (!("required" in status) || !("beta_required" in status)) {
      throw new Error("本地服务状态响应不完整");
    }
    $("#server-state").textContent = "本地园丁已就绪";
    $("#open-garden").disabled = false;
    return;
  } catch (error) {
    if (attempt < 45) {
      $("#server-state").textContent = "正在启动本地园丁…";
    } else if (attempt < gardenStartupTimeoutSeconds) {
      $("#server-state").textContent = "首次启动正在解压并初始化，通常需要 1–3 分钟，请稍候…";
    } else {
      $("#server-state").textContent = `本地园丁启动超时：${error?.message || error}；诊断日志：D:\\KnowledgeGarden\\data\\logs\\sidecar.log`;
    }
    if (attempt < gardenStartupTimeoutSeconds) {
      setTimeout(() => waitForGarden(attempt + 1), 1000);
    }
  }
}

async function checkForUpdate({ silent = false } = {}) {
  try {
    pendingUpdate = await check({ timeout: 30000 });
    if (!pendingUpdate) {
      if (!silent) $("#version-label").textContent = `当前 ${await getVersion()}，已是最新版`;
      return;
    }
    $("#update-title").textContent = `发现新版本 ${pendingUpdate.version}`;
    $("#update-notes").textContent = pendingUpdate.body || "包含稳定性与连接器改进。";
    $("#update-panel").hidden = false;
  } catch (error) {
    if (!silent) $("#version-label").textContent = `更新检查失败：${error}`;
  }
}

$("#open-garden").addEventListener("click", () => {
  $("#setup-panel").hidden = true;
  const frame = $("#garden-frame");
  frame.src = localGarden;
  frame.hidden = false;
});
$("#check-update").addEventListener("click", () => checkForUpdate());
$("#dismiss-update").addEventListener("click", () => { $("#update-panel").hidden = true; });
$("#install-update").addEventListener("click", async () => {
  if (!pendingUpdate) return;
  $("#install-update").disabled = true;
  $("#update-title").textContent = `正在安装 ${pendingUpdate.version}…`;
  try { await pendingUpdate.downloadAndInstall(); await relaunch(); }
  catch (error) { $("#update-notes").textContent = `安装失败：${error}`; $("#install-update").disabled = false; }
});

getVersion().then(version => { $("#version-label").textContent = `版本 ${version}`; });
invoke("backend_url").then(url => {
  localGarden = String(url || "");
  return waitForGarden();
}).catch(error => {
  $("#server-state").textContent = `本地服务启动失败：${error}`;
});
setTimeout(() => checkForUpdate({ silent: true }), 2500);
