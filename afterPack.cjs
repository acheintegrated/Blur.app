// build/afterPack.cjs — v11 (auto xattr clear for installed app)
// - FIX: EPERM on numpy dirs by replacing hand-rolled recursion with fs.cpSync (symlink-aware)
// - FIX: use lstatSync; preserve symlinks; pre-clean destination (rm -rf) before copy
// - ADD: dual venv mode (BLUR_VENV_MODE=dual) or single (BLUR_VENV_NAME=...)
// - ADD: strict mode (STRICT_AFTERPACK=1) to fail on missing assets
// - ADD: quarantine scrub for .app (xattr -dr com.apple.quarantine)
// - ADD: AUTO clear quarantine in /Applications if app exists there
// - ADD: chmod +x for bin/* in venvs + core/bin
// - SAFE: filters out __pycache__ and *.pyc when copying
//
// ENV knobs:
//   BLUR_VENV_MODE=dual                      → expects blur_env-darwin-x64 and blur_env-darwin-arm64
//   BLUR_VENV_NAME=blur_env-darwin-arm64     → single venv name (overrides dual)
//   STRICT_AFTERPACK=1                        → throw on any missing asset
//   BLUR_BUNDLE_ROOT=/abs/path                → source root override (expects /core under it)
//   BLUR_BUNDLE_BLURCHIVE=/abs/path          → blurchive source override
//   BLUR_REFRESH_BLURCHIVE=1                 → force re-sync blurchive even if exists

const fs = require("fs");
const path = require("path");
const { execSync, spawnSync } = require("child_process");

const exists = (p) => { try { return fs.existsSync(p); } catch { return false; } };
const isDir  = (p) => { try { return fs.lstatSync(p).isDirectory(); } catch { return false; } };
const statSafe = (p) => { try { return fs.lstatSync(p); } catch { return null; } };
const ensureDir = (p) => { try { fs.mkdirSync(p, { recursive: true }); } catch {} };

function rmrf(p) {
  if (!exists(p)) return;
  try { fs.rmSync(p, { recursive: true, force: true, maxRetries: 2 }); } catch {}
}

function clearQuarantine(p) {
  if (!exists(p)) return;
  try {
    console.log(`[afterPack] Clearing quarantine: ${p}`);
    execSync(`xattr -cr "${p}"`, { stdio: "inherit" });
  } catch (e) {
    console.warn(`[afterPack] Failed to clear quarantine on ${p}: ${e.message}`);
  }
}

function chmodExecRecursive(dir) {
  if (!isDir(dir)) return;
  const stack = [dir];
  while (stack.length) {
    const cur = stack.pop();
    for (const name of fs.readdirSync(cur)) {
      const p = path.join(cur, name);
      const st = statSafe(p); if (!st) continue;
      if (st.isDirectory()) {
        stack.push(p);
      } else {
        // only mark likely executables
        if (/\/bin\/[^/]+$/.test(p) || /(\.sh|\.bin|^activate)$/.test(name)) {
          try { fs.chmodSync(p, 0o755); } catch {}
        }
      }
    }
  }
}

// Prefer fs.cpSync (Node ≥16.7). Fallback to `cp -a`.
function copyTree(src, dst) {
  if (!exists(src)) return false;
  rmrf(dst);
  ensureDir(path.dirname(dst));

  if (fs.cpSync) {
    // keep symlinks; don't dereference; filter pycache junk
    fs.cpSync(src, dst, {
      recursive: true,
      force: true,
      dereference: false,
      errorOnExist: false,
      filter: (p) => {
        const base = path.basename(p);
        if (base === "__pycache__") return false;
        if (base.endsWith(".pyc") || base.endsWith(".pyo")) return false;
        return true;
      },
    });
    return true;
  }

  // fallback for older Node: cp -a preserves links, perms, times
  const res = spawnSync("cp", ["-a", src + "/.", dst], { stdio: "inherit" });
  if (res.status !== 0) throw new Error(`cp -a failed: ${src} -> ${dst}`);
  return true;
}

function resolveAppResources(context) {
  const appName = context?.packager?.appInfo?.productFilename || "Blur";
  const appOutDir = context?.appOutDir || process.cwd();
  const appPath = path.join(appOutDir, `${appName}.app`);
  const resDir = exists(appPath)
    ? path.join(appPath, "Contents", "Resources")
    : path.join(appOutDir, "resources");
  ensureDir(resDir);
  try { fs.writeFileSync(path.join(resDir, "BLUR_PACKAGED"), "1"); } catch {}
  return { appName, appOutDir, appPath, resDir };
}

module.exports = async (context) => {
  const strict = process.env.STRICT_AFTERPACK === "1";
  const refreshBlurchive = process.env.BLUR_REFRESH_BLURCHIVE === "1";
  const dualVenv = process.env.BLUR_VENV_MODE === "dual";
  const singleVenv = process.env.BLUR_VENV_NAME || null;

  if (!dualVenv && !singleVenv && strict) {
    throw new Error("[afterPack] must set BLUR_VENV_NAME (single) or BLUR_VENV_MODE=dual when STRICT_AFTERPACK=1");
  }

  const { appName, appOutDir, appPath, resDir } = resolveAppResources(context);
  const projectDir = context?.packager?.projectDir || process.cwd();

  const resCore      = path.join(resDir, "core");
  const resCoreBin   = path.join(resCore, "bin");
  const resBlurchive = path.join(resCore, "ouinet", "blurchive", "ecosystem");
  const dstChunks    = path.join(resBlurchive, "knowledge_chunks.jsonl");

  const envRoot      = process.env.BLUR_BUNDLE_ROOT;
  const envCore      = envRoot ? path.join(envRoot, "core") : null;
  const envBlurchive = process.env.BLUR_BUNDLE_BLURCHIVE || null;

  const repoCoreA      = path.join(projectDir, "core");
  const repoCoreB      = path.join(projectDir, "electron", "backend");
  const repoBlurchiveA = path.join(repoCoreA, "ouinet", "blurchive", "ecosystem");
  const repoBlurchiveB = path.join(repoCoreB, "ouinet", "blurchive", "ecosystem");

  const srcCore =
    (envCore && exists(envCore)) ? envCore :
    (exists(repoCoreA))          ? repoCoreA :
    (exists(repoCoreB))          ? repoCoreB : null;

  const srcBlurchive =
    (envBlurchive && exists(envBlurchive)) ? envBlurchive :
    (exists(repoBlurchiveA))               ? repoBlurchiveA :
    (exists(repoBlurchiveB))               ? repoBlurchiveB : null;

  // 1) Ensure core in Resources/core
  if (!exists(path.join(resCore, "convo_chat_core.py")) && srcCore) {
    console.log(`[afterPack] core missing → copying from: ${srcCore}`);
    copyTree(srcCore, resCore);
  } else if (!exists(path.join(resCore, "convo_chat_core.py"))) {
    const msg = "[afterPack] ❌ core missing and no source found";
    if (strict) throw new Error(msg);
    console.warn(msg);
  }

  // 2) Ensure blurchive (optional, warn if missing chunks)
  if (srcBlurchive) {
    const needSync = refreshBlurchive || !exists(resBlurchive) || !exists(dstChunks);
    if (needSync) {
      console.log(`[afterPack] syncing blurchive → ${resBlurchive}`);
      copyTree(srcBlurchive, resBlurchive);
    }
  }
  if (exists(resBlurchive) && !exists(dstChunks)) {
    const msg = `[afterPack] ⚠ blurchive present but knowledge_chunks.jsonl missing: ${dstChunks}`;
    if (strict) throw new Error(msg);
    console.warn(msg);
  }

  // 3) Venvs: dual or single
  const venvs = dualVenv
    ? ["blur_env-darwin-x64", "blur_env-darwin-arm64"]
    : (singleVenv ? [singleVenv] : []);

  const missing = [];
  for (const v of venvs) {
    const srcV = path.join(projectDir, v);
    const dstV = path.join(resDir, v);
    
    console.log(`[afterPack] Processing venv: ${v}`);
    
    if (!exists(dstV)) {
      if (exists(srcV)) {
        console.log(`[afterPack] copying venv → ${v}`);
        try {
          copyTree(srcV, dstV);
        } catch (e) {
          // surface a clearer error for numpy/symlink weirdness
          const msg = `[afterPack] failed to copy ${v}: ${e?.message || e}`;
          if (strict) throw new Error(msg);
          console.warn(msg);
        }
      } else {
        const msg = `[afterPack] ❌ Source venv not found: ${srcV}`;
        console.error(msg);
        missing.push(msg);
      }
    }
    
    const pythonExe = path.join(dstV, "bin", "python3");
    if (!exists(pythonExe)) {
      const msg = `[afterPack] ❌ Python not found in venv: ${pythonExe}`;
      console.error(msg);
      missing.push(msg);
    }
    
    chmodExecRecursive(path.join(dstV, "bin"));
  }

  // 4) perms for core/bin
  chmodExecRecursive(resCoreBin);

  // 5) quarantine clear (macOS)
  if (process.platform === "darwin") {
    // Clear quarantine on build output
    if (exists(appPath)) {
      clearQuarantine(appPath);
    }
    
    // ALSO clear quarantine in /Applications if app exists there
    const installedAppPath = path.join("/Applications", `${appName}.app`);
    if (exists(installedAppPath)) {
      console.log(`[afterPack] Detected installed app, clearing quarantine: ${installedAppPath}`);
      clearQuarantine(installedAppPath);
    }
  }

  if (missing.length) {
    const msg = `[afterPack] missing assets:\n - ${missing.join("\n - ")}`;
    if (strict) throw new Error(msg);
    console.warn(msg);
  }

  console.log("[afterPack] ✅ finalized Resources at:", resDir);
};