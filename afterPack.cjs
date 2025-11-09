// build/afterPack.cjs — minimal, loud, Blur-style.
// Goal: copy venv(s) into .app/Contents/Resources, no magic, no rsync bullshit.

const fs = require("fs");
const path = require("path");
const { execSync, spawnSync } = require("child_process");

const exists    = (p) => { try { return fs.existsSync(p); } catch { return false; } };
const isDir     = (p) => { try { return fs.lstatSync(p).isDirectory(); } catch { return false; } };
const statSafe  = (p) => { try { return fs.lstatSync(p); } catch { return null; } };
const ensureDir = (p) => { try { fs.mkdirSync(p, { recursive: true }); } catch {} };

function rmrf(p) {
  if (!exists(p)) return;
  try {
    fs.rmSync(p, { recursive: true, force: true, maxRetries: 2 });
  } catch (e) {
    console.error("[afterPack] rmrf failed:", p, String(e));
  }
}

// ----- ATTR/PERM -----

function clearQuarantine(p) {
  if (!exists(p) || process.platform !== "darwin") return;
  try {
    console.log(`[afterPack] xattr -cr ${p}`);
    execSync(`xattr -cr "${p}"`, { stdio: "ignore" });
  } catch (e) {
    console.warn("[afterPack] clearQuarantine failed:", String(e));
  }
}

function scrubAclsAndFlags(p) {
  if (!exists(p) || process.platform !== "darwin") return;
  try { execSync(`chmod -RN "${p}"`, { stdio: "ignore" }); } catch {}
  try { execSync(`chflags -R nouchg,noschg "${p}"`, { stdio: "ignore" }); } catch {}
  try { execSync(`chmod -R u+rwX "${p}"`, { stdio: "ignore" }); } catch {}
}

function makeWritableDir(p) {
  ensureDir(p);
  if (process.platform !== "darwin") return;
  const parts = p.split(path.sep);
  for (let i = parts.length; i > 1; i--) {
    const seg = parts.slice(0, i).join(path.sep);
    scrubAclsAndFlags(seg);
  }
}

// ----- COPY: DITTO-ONLY (we trust this) -----

function copyTree(src, dst) {
  if (!exists(src)) {
    const msg = `[afterPack] copyTree: src missing: ${src}`;
    console.error(msg);
    throw new Error(msg);
  }

  console.log(`[afterPack] copyTree (ditto-only): ${src} -> ${dst}`);

  rmrf(dst);
  makeWritableDir(path.dirname(dst));

  const args = ["--noqtn", src, dst];
  console.log("[afterPack] ditto", args.join(" "));
  const r = spawnSync("ditto", args, { stdio: "inherit" });

  if (r.status !== 0) {
    throw new Error(`ditto failed (${r.status}) ${src} -> ${dst}`);
  }

  console.log(`[afterPack] ditto OK ${dst}`);
}

// ----- PERMISSIONS -----

function chmodExecRecursive(dir) {
  if (!isDir(dir)) return;
  const stack = [dir];
  while (stack.length) {
    const cur = stack.pop();
    for (const name of fs.readdirSync(cur)) {
      const p = path.join(cur, name);
      const st = statSafe(p);
      if (!st) continue;
      if (st.isDirectory()) {
        stack.push(p);
      } else if (
        /\/bin\/[^/]+$/.test(p) ||
        /(\.sh|\.bin|^activate)$/.test(name)
      ) {
        try {
          fs.chmodSync(p, 0o755);
        } catch (e) {
          console.warn("[afterPack] chmod +x failed:", p, String(e));
        }
      }
    }
  }
}

// ----- RES PATH -----

function resolveAppResources(context) {
  const appName =
    context?.packager?.appInfo?.productFilename || "Blur";
  const appOutDir = context?.appOutDir || process.cwd();
  const appPath = path.join(appOutDir, `${appName}.app`);
  const resDir = exists(appPath)
    ? path.join(appPath, "Contents", "Resources")
    : path.join(appOutDir, "resources");

  ensureDir(resDir);

  try {
    fs.writeFileSync(path.join(resDir, "BLUR_PACKAGED"), "1");
  } catch (e) {
    console.warn("[afterPack] failed to write BLUR_PACKAGED:", String(e));
  }

  console.log("[afterPack] appOutDir:", appOutDir);
  console.log("[afterPack] appPath:", appPath, "exists?", exists(appPath));
  console.log("[afterPack] resDir:", resDir, "exists?", exists(resDir));

  return { appName, appOutDir, appPath, resDir };
}

// ----- MAIN HOOK -----

module.exports = async (context) => {
  const strict   = process.env.STRICT_AFTERPACK === "1";
  const dualVenv = process.env.BLUR_VENV_MODE === "dual";
  const single   = process.env.BLUR_VENV_NAME || null;

  console.log("[afterPack] strict =", strict);
  console.log("[afterPack] BLUR_VENV_MODE =", dualVenv ? "dual" : "single");
  console.log("[afterPack] BLUR_VENV_NAME =", single);

  if (!dualVenv && !single && strict) {
    throw new Error(
      "[afterPack] must set BLUR_VENV_NAME (single) or BLUR_VENV_MODE=dual"
    );
  }

  const { appName, appOutDir, appPath, resDir } = resolveAppResources(context);
  const projectDir = context?.packager?.projectDir || process.cwd();
  const resCoreBin = path.join(resDir, "core", "bin");

  if (process.platform === "darwin") {
    clearQuarantine(appPath);
    scrubAclsAndFlags(appPath);
    makeWritableDir(path.join(appPath, "Contents"));
    makeWritableDir(resDir);
  }

  const venvs = dualVenv
    ? ["blur_env-darwin-x64", "blur_env-darwin-arm64"]
    : (single ? [single] : []);

  const missing = [];

  for (const v of venvs) {
    const srcV = path.join(projectDir, v);
    const dstV = path.join(resDir, v);

    console.log("[afterPack] Processing venv:", v);
    console.log("[afterPack]   srcV:", srcV, "exists?", exists(srcV));
    console.log("[afterPack]   dstV:", dstV);

    if (!exists(srcV)) {
      const msg = `[afterPack] ❌ Source venv not found: ${srcV}`;
      console.error(msg);
      missing.push(msg);
      continue;
    }

    if (process.platform === "darwin") {
      clearQuarantine(srcV);
      scrubAclsAndFlags(srcV);
    }

    try {
      copyTree(srcV, dstV);
    } catch (e) {
      const msg = `[afterPack] ❌ venv copy failed (${v}): ${String(e)}`;
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

  chmodExecRecursive(resCoreBin);

  if (process.platform === "darwin") {
    clearQuarantine(appPath);
    const installedAppPath = path.join("/Applications", `${appName}.app`);
    if (exists(installedAppPath)) clearQuarantine(installedAppPath);
  }

  if (missing.length) {
    const msg = `[afterPack] missing assets:\n - ${missing.join("\n - ")}`;
    if (strict) throw new Error(msg);
    console.warn(msg);
  }

  console.log("[afterPack] ✅ finalized Resources at:", resDir);
};
