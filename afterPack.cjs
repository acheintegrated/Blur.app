// build/afterPack.cjs — v13 (FIX symlinks by dereferencing)
// - FIX: Set `dereference: true` in fs.cpSync to resolve symlinks like `python3`.
// - FIX: Use `cp -aL` in fallback to ensure symlinks are followed.
// - REVERT: Venv copying is now handled here again, not in package.json's extraResources.

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
        if (/\/bin\/[^/]+$/.test(p) || /(\.sh|\.bin|^activate)$/.test(name)) {
          try { fs.chmodSync(p, 0o755); } catch {}
        }
      }
    }
  }
}

// **KEY CHANGE HERE**: This function now resolves symlinks.
function copyTree(src, dst) {
  if (!exists(src)) return false;
  rmrf(dst);
  ensureDir(path.dirname(dst));

  if (fs.cpSync) {
    fs.cpSync(src, dst, {
      recursive: true,
      force: true,
      dereference: true, // <-- The fix! This copies file content, not the link.
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

  // Fallback for older Node: `cp -aL` follows all symlinks.
  const res = spawnSync("cp", ["-aL", src + "/.", dst], { stdio: "inherit" });
  if (res.status !== 0) throw new Error(`cp -aL failed: ${src} -> ${dst}`);
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
  const dualVenv = process.env.BLUR_VENV_MODE === "dual";
  const singleVenv = process.env.BLUR_VENV_NAME || null;

  if (!dualVenv && !singleVenv && strict) {
    throw new Error("[afterPack] must set BLUR_VENV_NAME (single) or BLUR_VENV_MODE=dual");
  }

  const { appName, appOutDir, appPath, resDir } = resolveAppResources(context);
  const projectDir = context?.packager?.projectDir || process.cwd();

  const resCoreBin   = path.join(resDir, "core", "bin");

  // Venvs: Dual or single copy logic is now back in this script.
  const venvs = dualVenv
    ? ["blur_env-darwin-x64", "blur_env-darwin-arm64"]
    : (singleVenv ? [singleVenv] : []);

  const missing = [];
  for (const v of venvs) {
    const srcV = path.join(projectDir, v);
    const dstV = path.join(resDir, v);
    
    console.log(`[afterPack] Processing venv: ${v}`);
    
    if (exists(srcV)) {
        console.log(`[afterPack] Copying and dereferencing venv → ${v}`);
        copyTree(srcV, dstV);
    } else {
        const msg = `[afterPack] ❌ Source venv not found: ${srcV}`;
        console.error(msg);
        missing.push(msg);
        continue;
    }
    
    const pythonExe = path.join(dstV, "bin", "python3");
    if (!exists(pythonExe)) {
      const msg = `[afterPack] ❌ Python not found in packaged venv: ${pythonExe}`;
      console.error(msg);
      missing.push(msg);
    }
    
    chmodExecRecursive(path.join(dstV, "bin"));
  }

  // Perms for core/bin
  chmodExecRecursive(resCoreBin);

  // Quarantine clear (macOS)
  if (process.platform === "darwin") {
    if (exists(appPath)) {
      clearQuarantine(appPath);
    }
    const installedAppPath = path.join("/Applications", `${appName}.app`);
    if (exists(installedAppPath)) {
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