// build/afterPack.cjs — v8 (hardened DMG payload copy + strict sanity checks)
const fs = require("fs");
const path = require("path");
const { execSync } = require("child_process");

/* ---------------- helpers ---------------- */
const exists   = p => { try { return fs.existsSync(p); } catch { return false; } };
const statSafe = p => { try { return fs.statSync(p); } catch { return null; } };
const isDir    = p => !!(statSafe(p)?.isDirectory());
const ensure   = p => fs.mkdirSync(p, { recursive: true });

function list(dir) { try { return fs.readdirSync(dir); } catch { return []; } }

function copyDir(src, dst) {
  const st = statSafe(src);
  if (!st) return false;
  if (st.isDirectory()) {
    ensure(dst);
    for (const name of fs.readdirSync(src)) {
      const s = path.join(src, name);
      const d = path.join(dst, name);
      copyDir(s, d);
    }
  } else {
    ensure(path.dirname(dst));
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

function must(human, cond, detail) {
  if (!cond) {
    const msg = `[afterPack] ❌ ${human}${detail ? ` — ${detail}` : ""}`;
    throw new Error(msg);
  }
}

/* ---------------- main hook ---------------- */
module.exports = async (context) => {
  const strict = (process.env.STRICT_AFTERPACK === "1"); // we default to true in scripts
  const appName    = context?.packager?.appInfo?.productFilename || "Blur";
  const projectDir = context?.packager?.projectDir || process.cwd();

  // Where the .app lives for this target
  const appPath = path.join(context.appOutDir, `${appName}.app`);
  const resDir  = exists(appPath)
    ? path.join(appPath, "Contents", "Resources")
    : path.join(context.appOutDir, "resources");
  ensure(resDir);

  // Sentinels for later runtime checks
  try { fs.writeFileSync(path.join(resDir, "BLUR_PACKAGED"), "1"); } catch {}

  // Layout inside Resources
  const resCore        = path.join(resDir, "core");
  const resCoreBin     = path.join(resCore, "bin");
  const resVenv        = path.join(resDir, "blur_env-darwin-arm64");
  const resVenvBin     = path.join(resVenv, "bin");
  const resPython      = path.join(resVenvBin, "python3");
  const resConfig      = path.join(resDir, "config.yaml");
  const resAcheflip    = path.join(resDir, "acheflip.yaml");
  const resModels      = path.join(resDir, "models");
  const resBlurchive   = path.join(resCore, "ouinet", "blurchive", "ecosystem");
  const resChunks      = path.join(resBlurchive, "knowledge_chunks.jsonl");

  // Source roots (env preferred, then repo)
  const envRoot      = process.env.BLUR_BUNDLE_ROOT;                       // e.g. "$PWD" or "$HOME/blur"
  const srcVenv      = process.env.BLUR_BUNDLE_VENV === "1"
    ? path.join(envRoot || projectDir, "blur_env-darwin-arm64")
    : path.join(projectDir, "blur_env-darwin-arm64"); // still try local folder
  const srcCoreA     = path.join(projectDir, "core");
  const srcCoreB     = path.join(projectDir, "electron", "backend");
  const srcCore      = exists(srcCoreA) ? srcCoreA : (exists(srcCoreB) ? srcCoreB : null);

  const srcModels    = process.env.BLUR_BUNDLE_MODELS || path.join(projectDir, "models");
  const srcConfig    = path.join(projectDir, "config.yaml");
  const srcAcheflip  = path.join(projectDir, "acheflip.yaml");

  const envBlurchive = process.env.BLUR_BUNDLE_BLURCHIVE || null;
  const srcBlurchiveA= path.join(srcCoreA, "ouinet", "blurchive", "ecosystem");
  const srcBlurchiveB= path.join(srcCoreB, "ouinet", "blurchive", "ecosystem");
  const srcBlurchive = (envBlurchive && exists(envBlurchive)) ? envBlurchive
                        : exists(srcBlurchiveA) ? srcBlurchiveA
                        : exists(srcBlurchiveB) ? srcBlurchiveB
                        : null;

  console.log("[afterPack] appOutDir:", context.appOutDir);
  console.log("[afterPack] Resources:", resDir);
  console.log("[afterPack] sources:", {
    srcVenv,
    srcCore,
    srcModels: exists(srcModels) ? srcModels : "(missing)",
    srcConfig: exists(srcConfig) ? srcConfig : "(missing)",
    srcAcheflip: exists(srcAcheflip) ? srcAcheflip : "(missing)",
    srcBlurchive: srcBlurchive || "(missing)",
  });

  /* ---- Copy: core (only if missing; extraResources usually places it) ---- */
  const neededCoreMain = path.join(resCore, "convo_chat_core.py");
  if (!exists(neededCoreMain) && srcCore) {
    console.log("[afterPack] core missing → copying core/ → Resources/core");
    copyDir(srcCore, resCore);
  }

  /* ---- Copy: config & acheflip (overwrite allowed; you treat bundle as truth) ---- */
  if (exists(srcConfig))   { ensure(path.dirname(resConfig));   fs.copyFileSync(srcConfig, resConfig); }
  if (exists(srcAcheflip)) { ensure(path.dirname(resAcheflip)); fs.copyFileSync(srcAcheflip, resAcheflip); }

  /* ---- Copy: models (optional; do not fail build if absent) ---- */
  if (exists(srcModels)) {
    if (!exists(resModels) || list(resModels).length === 0) {
      console.log("[afterPack] copying models/ (first seed) …");
      copyDir(srcModels, resModels);
    } else {
      console.log("[afterPack] models already present, skipping heavy copy.");
    }
  } else {
    console.log("[afterPack] models not found in source — continuing without models.");
  }

  /* ---- Copy: blurchive dataset (copy when directory missing OR chunks missing) ---- */
  if (srcBlurchive) {
    const need = !exists(resBlurchive) || !exists(resChunks);
    if (need) {
      console.log("[afterPack] syncing blurchive dataset →", resBlurchive);
      copyDir(srcBlurchive, resBlurchive);
    }
  }

  /* ---- Copy: venv (critical) ----
     We *always* ensure a full venv with bin/python3 lives under Resources.
     If extraResources already placed it but it's partial, we fix it here. */
  const hasPython = exists(resPython);
  if (!hasPython) {
    console.log("[afterPack] venv python missing → copying blur_env-darwin-arm64 → Resources/");
    must("venv source directory must exist", exists(srcVenv) && isDir(srcVenv), srcVenv);
    copyDir(srcVenv, resVenv);
  }

  /* ---- Fix perms on executables ---- */
  chmodExecRecursive(resVenvBin);
  chmodExecRecursive(resCoreBin);

  /* ---- Clear quarantine (local test) ---- */
  if (process.platform === "darwin" && exists(appPath)) {
    clearQuarantine(appPath);
  }

  /* ---- Sanity checks (STRICT) ---- */
  const missing = [];
  if (!exists(resPython))           missing.push(resPython);
  if (!exists(neededCoreMain))      missing.push(neededCoreMain);
  if (!exists(resConfig))           missing.push(resConfig);

  if (missing.length) {
    const msg = `[afterPack] missing required assets:\n - ${missing.join("\n - ")}`;
    if (strict) throw new Error(msg);
    console.warn(msg);
  }

  console.log("[afterPack] ✅ finalized Resources at:", resDir);
};
