// build/afterPack.cjs — v7 (v5 semantics + fix: copy when chunks missing)
const fs = require("fs");
const path = require("path");
const { execSync } = require("child_process");

/* ---------------- helpers ---------------- */
const exists = (p) => { try { return fs.existsSync(p); } catch { return false; } };
const isDir  = (p) => { try { return fs.statSync(p).isDirectory(); } catch { return false; } };
const ensureDir = (p) => fs.mkdirSync(p, { recursive: true });
const statSafe = (p) => { try { return fs.statSync(p); } catch { return null; } };

function copyDir(src, dst) {
  if (!exists(src)) return false;
  const st = statSafe(src); if (!st) return false;

  if (st.isDirectory()) {
    ensureDir(dst);
    for (const name of fs.readdirSync(src)) {
      copyDir(path.join(src, name), path.join(dst, name));
    }
  } else {
    ensureDir(path.dirname(dst));
    fs.copyFileSync(src, dst);
  }
  return true;
}

function chmodExecRecursive(dir) {
  if (!isDir(dir)) return;
  const stack = [dir];
  while (stack.length) {
    const cur = stack.pop();
    for (const name of fs.readdirSync(cur)) {
      const p = path.join(cur, name);
      const st = statSafe(p); if (!st) continue;
      if (st.isDirectory()) stack.push(p);
      else { try { fs.chmodSync(p, 0o755); } catch {} }
    }
  }
}

function clearQuarantine(p) {
  try { execSync(`xattr -dr com.apple.quarantine "${p}"`, { stdio: "ignore" }); } catch {}
}

/* ---------------- main hook ---------------- */
module.exports = async (context) => {
  const strict = process.env.STRICT_AFTERPACK === "1";
  const refreshBlurchive = process.env.BLUR_REFRESH_BLURCHIVE === "1";

  const appName = context?.packager?.appInfo?.productFilename || "Blur";
  const projectDir = context?.packager?.projectDir || process.cwd();

  // Resolve Resources dir (mac .app) or fallback (other targets)
  const appPath = path.join(context.appOutDir, `${appName}.app`);
  const resDir = exists(appPath)
    ? path.join(appPath, "Contents", "Resources")
    : path.join(context.appOutDir, "resources");

  ensureDir(resDir);
  try { fs.writeFileSync(path.join(resDir, "BLUR_PACKAGED"), "1"); } catch {}

  // Layout inside Resources
  const resCore      = path.join(resDir, "core");
  const resCoreBin   = path.join(resCore, "bin");
  const resVenvBin   = path.join(resDir, "blur_env-darwin-arm64", "bin");
  const resBlurchive = path.join(resCore, "ouinet", "blurchive", "ecosystem");
  const dstChunks    = path.join(resBlurchive, "knowledge_chunks.jsonl");

  // Source resolution (env first, then repo)
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

  /* --- 1) Ensure core exists (don’t stomp extraResources output) --- */
  const coreMain = path.join(resCore, "convo_chat_core.py");
  if (!exists(coreMain) && srcCore) {
    console.log(`[afterPack] core missing in Resources → copying from: ${srcCore}`);
    copyDir(srcCore, resCore);
  }

  /* --- 2) Ensure blurchive dataset ---
     FIX: also copy when dir exists BUT chunks file is missing. */
  if (srcBlurchive) {
    const needSync = refreshBlurchive || !exists(resBlurchive) || !exists(dstChunks);
    if (needSync) {
      console.log(`[afterPack] syncing blurchive dataset → ${resBlurchive} (refresh=${refreshBlurchive}, existsDir=${exists(resBlurchive)}, hasChunks=${exists(dstChunks)})`);
      copyDir(srcBlurchive, resBlurchive);
    }
  }

  /* --- 3) Permissions for executables (recursive) --- */
  chmodExecRecursive(resVenvBin);
  chmodExecRecursive(resCoreBin);

  /* --- 4) Sanity checks --- */
  const mustHave = [
    path.join(resDir, "blur_env-darwin-arm64", "bin", "python3"),
    coreMain,
  ];
  const missing = mustHave.filter((p) => !exists(p));
  for (const p of missing) console.warn("[afterPack] ⚠ missing expected asset:", p);

  // Dataset sanity (ecosystem-only)
  if (exists(resBlurchive) && !exists(dstChunks)) {
    const msg = `[afterPack] ⚠ blurchive present but knowledge_chunks.jsonl missing: ${dstChunks}`;
    if (strict) throw new Error(msg);
    console.warn(msg);
  }

  /* --- 5) Clear quarantine for local test runs on macOS --- */
  if (process.platform === "darwin" && exists(appPath)) {
    clearQuarantine(appPath);
  }

  if (missing.length && strict) {
    throw new Error(`[afterPack] ❌ strict mode: missing assets:\n - ${missing.join("\n - ")}`);
  }

  console.log("[afterPack] ✅ finalized Resources at:", resDir);
};
