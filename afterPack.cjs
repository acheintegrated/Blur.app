// build/afterPack.cjs — v14.2 (DEST SCRUB + SAFE FALLBACKS)
const fs = require("fs");
const path = require("path");
const { execSync, spawnSync } = require("child_process");

const exists    = (p) => { try { return fs.existsSync(p); } catch { return false; } };
const isDir     = (p) => { try { return fs.lstatSync(p).isDirectory(); } catch { return false; } };
const statSafe  = (p) => { try { return fs.lstatSync(p); } catch { return null; } };
const ensureDir = (p) => { try { fs.mkdirSync(p, { recursive: true }); } catch {} };

function rmrf(p) {
  if (exists(p)) try { fs.rmSync(p, { recursive: true, force: true, maxRetries: 2 }); } catch {}
}

// ----- ATTR/PERM FIXERS -----
function clearQuarantine(p) {
  if (!exists(p) || process.platform !== "darwin") return;
  try { console.log(`[afterPack] Clearing quarantine: ${p}`); execSync(`xattr -cr "${p}"`, { stdio: "ignore" }); } catch {}
}

function scrubAclsAndFlags(p) {
  if (!exists(p) || process.platform !== "darwin") return;
  try { execSync(`chmod -RN "${p}"`, { stdio: "ignore" }); } catch {}
  try { execSync(`chflags -R nouchg,noschg "${p}"`, { stdio: "ignore" }); } catch {}
  try { execSync(`chmod -R u+rwX "${p}"`, { stdio: "ignore" }); } catch {}
}

function makeWritableDir(p) {
  ensureDir(p);
  if (process.platform === "darwin") {
    // scrub both dir and its parents a bit
    const parts = p.split(path.sep);
    for (let i = parts.length; i > 1; i--) {
      const seg = parts.slice(0, i).join(path.sep);
      scrubAclsAndFlags(seg);
    }
  }
}

// ----- COPY -----
function copyTree_node(src, dst) {
  // exclude test junk + fortran sources
  const EXCLUDE_RE = /(^|\/)(__pycache__|tests?|benchmarks|examples)(\/|$)|\.(pyc|pyo)$|\.f90$|\.f$/;
  fs.cpSync(src, dst, {
    recursive: true,
    force: true,
    dereference: true,
    errorOnExist: false,
    filter: (p) => !EXCLUDE_RE.test(p),
  });
}

function copyTree_rsync(src, dst) {
  // DO NOT use -a; it preserves perms/owners/times. We want *simple*.
  const args = [
    "-rL",                // recursive, follow links
    "--delete",
    "--omit-dir-times",
    "--no-perms", "--no-owner", "--no-group",
    "--exclude", "__pycache__/",
    "--exclude", "tests/",
    "--exclude", "test/",
    "--exclude", "benchmarks/",
    "--exclude", "examples/",
    "--exclude", "*.pyc",
    "--exclude", "*.pyo",
    "--exclude", "*.f90",
    "--exclude", "*.f",
    src + "/.", dst
  ];
  const r = spawnSync("rsync", args, { stdio: "inherit" });
  if (r.status !== 0) throw new Error(`rsync failed ${r.status}`);
}

function copyTree_ditto(src, dst) {
  // ditto ignores quarantine with --noqtn; creates dirs as needed; copies xattrs sanely
  const r = spawnSync("ditto", ["--noqtn", src, dst], { stdio: "inherit" });
  if (r.status !== 0) throw new Error(`ditto failed ${r.status}`);
}

function copyTree(src, dst) {
  if (!exists(src)) return false;
  rmrf(dst);
  makeWritableDir(path.dirname(dst)); // ensure parent path is writable

  // try Node first
  try {
    copyTree_node(src, dst);
    return true;
  } catch (e1) {
    console.warn("[afterPack] cpSync failed:", String(e1).split("\n")[0]);
  }

  // then rsync (non-preserving)
  try {
    copyTree_rsync(src, dst);
    return true;
  } catch (e2) {
    console.warn("[afterPack] rsync failed:", String(e2).split("\n")[0]);
  }

  // last, ditto
  copyTree_ditto(src, dst);
  return true;
}

// ----- PERMS -----
function chmodExecRecursive(dir) {
  if (!isDir(dir)) return;
  const stack = [dir];
  while (stack.length) {
    const cur = stack.pop();
    for (const name of fs.readdirSync(cur)) {
      const p = path.join(cur, name);
      const st = statSafe(p); if (!st) continue;
      if (st.isDirectory()) { stack.push(p); }
      else {
        if (/\/bin\/[^/]+$/.test(p) || /(\.sh|\.bin|^activate)$/.test(name)) {
          try { fs.chmodSync(p, 0o755); } catch {}
        }
      }
    }
  }
}

// ----- RES PATH -----
function resolveAppResources(context) {
  const appName = context?.packager?.appInfo?.productFilename || "Blur";
  const appOutDir = context?.appOutDir || process.cwd();
  const appPath = path.join(appOutDir, `${appName}.app`);
  const resDir = exists(appPath) ? path.join(appPath, "Contents", "Resources")
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
  const resCoreBin = path.join(resDir, "core", "bin");

  // 🔑 DESTINATION SCRUB: make .app writable before we copy
  if (process.platform === "darwin") {
    clearQuarantine(appPath);
    scrubAclsAndFlags(appPath);
    // ensure Contents and Resources can be written
    makeWritableDir(path.join(appPath, "Contents"));
    makeWritableDir(resDir);
  }

  const venvs = dualVenv ? ["blur_env-darwin-x64", "blur_env-darwin-arm64"]
                         : (singleVenv ? [singleVenv] : []);

  const missing = [];
  for (const v of venvs) {
    const srcV = path.join(projectDir, v);
    const dstV = path.join(resDir, v);

    console.log(`[afterPack] Processing venv: ${v}`);
    if (!exists(srcV)) { const msg = `[afterPack] ❌ Source venv not found: ${srcV}`; console.error(msg); missing.push(msg); continue; }

    // scrub source too (xattrs/ACL/flags)
    if (process.platform === "darwin") { clearQuarantine(srcV); scrubAclsAndFlags(srcV); }

    console.log(`[afterPack] Copying and dereferencing venv → ${v}`);
    copyTree(srcV, dstV);

    const pythonExe = path.join(dstV, "bin", "python3");
    if (!exists(pythonExe)) { const msg = `[afterPack] ❌ Python not found in packaged venv: ${pythonExe}`; console.error(msg); missing.push(msg); }
    chmodExecRecursive(path.join(dstV, "bin"));
  }

  chmodExecRecursive(resCoreBin);

  // final pass to remove quarantine on bundle
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
