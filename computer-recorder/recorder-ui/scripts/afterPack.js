const { execSync } = require('child_process');
const path = require('path');

// Developer ID for code signing. Must match the identity build_backend.sh used,
// so set CREC_SIGNING_IDENTITY once for both. Without it the backend is left
// unsigned: the app still runs, but macOS will prompt for Accessibility and
// Screen Recording separately instead of letting the helper inherit them.
const SIGNING_IDENTITY = process.env.CREC_SIGNING_IDENTITY;

function hasSigningIdentity() {
    if (!SIGNING_IDENTITY) {
        return false;
    }
    try {
        const output = execSync('security find-identity -v -p codesigning', {
            encoding: 'utf8',
            stdio: ['ignore', 'pipe', 'pipe']
        });
        return output.includes(SIGNING_IDENTITY);
    } catch (error) {
        return false;
    }
}

/**
 * afterPack hook for electron-builder
 * This signs all helper binaries in the app bundle with Developer ID
 * to ensure they can inherit TCC permissions from the parent app.
 */
exports.default = async function(context) {
    // Only run on macOS
    if (context.packager.platform.name !== 'mac') {
        return;
    }

    const appPath = context.appOutDir;
    const appName = context.packager.appInfo.productFilename;
    const appBundlePath = path.join(appPath, `${appName}.app`);
    const resourcesPath = path.join(appBundlePath, 'Contents', 'Resources');
    const backendPath = path.join(resourcesPath, 'backend', 'crec-service');
    const entitlementsPath = path.join(context.packager.projectDir, 'resources', 'entitlements.inherit.plist');

    console.log('afterPack: Starting post-packaging steps...');

    if (!hasSigningIdentity()) {
        console.log(
            SIGNING_IDENTITY
                ? `afterPack: Signing identity not found in keychain, skipping backend codesign: ${SIGNING_IDENTITY}`
                : 'afterPack: CREC_SIGNING_IDENTITY not set, skipping backend codesign.'
        );
        return;
    }
    
    // Clean extended attributes from the entire app bundle to prevent signing errors
    // "resource fork, Finder information, or similar detritus not allowed"
    // The com.apple.provenance and com.apple.FinderInfo xattrs cause codesign to fail
    console.log(`Cleaning extended attributes from: ${appBundlePath}`);
    
    // Multiple passes to ensure all xattrs are removed
    const cleanupCommands = [
        // First pass: use xattr -cr recursively
        `xattr -cr "${appBundlePath}"`,
        // Specifically target the Frameworks directory where Electron helpers live
        `xattr -cr "${path.join(appBundlePath, 'Contents', 'Frameworks')}"`,
        // Use find + xargs as backup, handles filenames with spaces
        `find "${appBundlePath}" -print0 | xargs -0 xattr -c 2>/dev/null || true`,
        // Explicitly remove problematic attributes
        `find "${appBundlePath}" -exec xattr -d com.apple.provenance {} 2>/dev/null \\; || true`,
        `find "${appBundlePath}" -exec xattr -d com.apple.FinderInfo {} 2>/dev/null \\; || true`,
        `find "${appBundlePath}" -exec xattr -d com.apple.fileprovider.fpfs#P {} 2>/dev/null \\; || true`,
    ];
    
    for (const cmd of cleanupCommands) {
        try {
            execSync(cmd, { stdio: 'inherit' });
        } catch (e) {
            // Ignore errors - some files may not have these attrs
        }
    }
    console.log('Extended attribute cleanup complete.');

    console.log('afterPack: Signing backend executables with Developer ID...');
    console.log(`Backend path: ${backendPath}`);

    try {
        // Sign all .so and .dylib files in _internal with Developer ID
        const internalPath = path.join(backendPath, '_internal');
        console.log(`Signing libraries in: ${internalPath}`);
        
        try {
            execSync(`find "${internalPath}" -type f \\( -name "*.so" -o -name "*.dylib" \\) -exec codesign --force --sign "${SIGNING_IDENTITY}" {} \\;`, {
                stdio: 'inherit'
            });
        } catch (e) {
            console.log('Some libraries may have skipped signing (this is normal)');
        }

        // Sign the main crec-service executable with Developer ID and inherit entitlements
        const mainExe = path.join(backendPath, 'crec-service');
        console.log(`Signing: ${mainExe}`);
        execSync(`codesign --force --options runtime --deep --sign "${SIGNING_IDENTITY}" --entitlements "${entitlementsPath}" "${mainExe}"`, {
            stdio: 'inherit'
        });

        console.log('afterPack: Backend signing complete with Developer ID');
    } catch (error) {
        console.error('afterPack: Error signing backend:', error.message);
        throw error; // Fail the build if signing fails
    }
};
