// scripts/notarize.js

const { notarize } = require('@electron/notarize');

exports.default = async function notarizing(context) {
  const { electronPlatformName, appOutDir, packager } = context;
  if (electronPlatformName !== 'darwin') return;

  const appName = packager.appInfo.productFilename;
  const appPath = `${appOutDir}/${appName}.app`;

  console.log("Notarizing:", appPath);

  return await notarize({
    appBundleId: packager.appInfo.id,  // e.g. com.mycompany.myapp
    appPath: appPath,
    // Option 1: Apple ID + app-specific password
    appleId: process.env.APPLE_ID,
    appleIdPassword: process.env.APPLE_ID_PASSWORD,
    teamId: process.env.APPLE_TEAM_ID,  // optional but safer

    // Option 2: App Store Connect API key (preferable in CI)
    // appleApiKey: process.env.APPLE_API_KEY,        // path or content of .p8
    // appleApiKeyId: process.env.APPLE_API_KEY_ID,
    // appleApiIssuer: process.env.APPLE_API_ISSUER_ID,
  });
};
