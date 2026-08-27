import React, { useEffect, useState } from 'react'
import { CheckCircle, XCircle, Settings, Monitor, MousePointer, Keyboard, RefreshCw, HelpCircle, Plus } from 'lucide-react'

// Define the interface for permissions status
interface PermissionStatus {
    screenRecording: boolean;
    accessibility: boolean;
    inputMonitoring: boolean;
}

const PermissionsPage: React.FC<{ onComplete: () => void }> = ({ onComplete }) => {
    const [status, setStatus] = useState<PermissionStatus>({
        screenRecording: false,
        accessibility: false,
        inputMonitoring: false,
    });
    const [checking, setChecking] = useState(false);
    // Manual override is now handled by IPC, no local state needed for it
    const [showHelp, setShowHelp] = useState<string | null>(null);

    const checkPermissions = async () => {
        setChecking(true);
        try {
            const hasPermissions = await window.electron.ipcRenderer.invoke('check-permissions');
            setStatus(prev => ({
                ...prev,
                ...hasPermissions
            }));
        } catch (e) {
            console.error("Failed to check permissions:", e);
        } finally {
            setTimeout(() => setChecking(false), 500);
        }
    };

    useEffect(() => {
        // Initial check
        checkPermissions();
        // Poll every few seconds
        const interval = setInterval(checkPermissions, 5000);
        return () => clearInterval(interval);
    }, []);



    const allGranted = status.screenRecording && status.accessibility && status.inputMonitoring;

    const toggleHelp = (id: string) => {
        setShowHelp(showHelp === id ? null : id);
    };

    return (
        <div className="flex flex-col items-center justify-center min-h-screen bg-gray-50 text-gray-800 p-8 space-y-8">
            <div className="text-center space-y-2">
                <h1 className="text-3xl font-bold tracking-tight text-gray-900">Permissions Required</h1>
                <p className="text-gray-500 max-w-md">
                    To record your workflow, this app needs access to your screen, accessibility features, and input devices.
                </p>
            </div>

            <div className="w-full max-w-lg space-y-4">
                <PermissionItem
                    icon={<Monitor className="w-6 h-6" />}
                    title="Screen Recording"
                    description="Required to capture your screen content."
                    granted={status.screenRecording}
                    onGrant={() => { window.electron.ipcRenderer.invoke('open-settings', 'screen'); }}
                    showHelp={showHelp === 'screen'}
                    onToggleHelp={() => toggleHelp('screen')}
                />

                {/* Accessibility */}
                <PermissionItem
                    icon={<Settings className="w-6 h-6" />}
                    title="Accessibility"
                    description="Required to observe window events and automation."
                    granted={status.accessibility}
                    onGrant={() => { window.electron.ipcRenderer.invoke('open-settings', 'accessibility'); }}
                    showHelp={showHelp === 'accessibility'}
                    onToggleHelp={() => toggleHelp('accessibility')}
                />

                {/* Input Monitoring */}
                <PermissionItem
                    icon={<MousePointer className="w-6 h-6" />}
                    title="Input Monitoring"
                    description="Required to track mouse and keyboard activity."
                    granted={status.inputMonitoring}
                    onGrant={() => { window.electron.ipcRenderer.invoke('open-settings', 'input'); }}
                    showHelp={showHelp === 'input'}
                    onToggleHelp={() => toggleHelp('input')}
                />
            </div>

            <div className="flex space-x-4">
                <button
                    onClick={checkPermissions}
                    className={`flex items-center px-4 py-3 rounded-xl font-medium transition-all duration-200 bg-white border border-gray-200 text-gray-600 hover:bg-gray-50 hover:border-gray-300 ${checking ? 'opacity-70 cursor-wait' : ''}`}
                    disabled={checking}
                >
                    <RefreshCw className={`w-5 h-5 mr-2 ${checking ? 'animate-spin' : ''}`} />
                    Refresh Status
                </button>

                <button
                    onClick={onComplete}
                    disabled={!allGranted}
                    className={`px-8 py-3 rounded-xl font-medium transition-all duration-200 ${allGranted
                        ? 'bg-blue-600 text-white hover:bg-blue-500 shadow-lg hover:shadow-xl transform hover:-translate-y-0.5'
                        : 'bg-gray-200 text-gray-400 cursor-not-allowed'
                        }`}
                >
                    Continue
                </button>
            </div>

            {/* Dev Bypass */}
            <div className="absolute bottom-4 right-4">
                <button
                    onClick={async () => {
                        if (confirm("Reset all saved permission preferences?")) {
                            await window.electron.ipcRenderer.invoke('reset-settings');
                            checkPermissions();
                        }
                    }}
                    className="text-xs text-gray-400 hover:text-red-500 mr-4"
                >
                    Reset Saved Status
                </button>
                <button
                    onClick={() => {
                        setStatus({ screenRecording: true, accessibility: true, inputMonitoring: true });
                    }}
                    className="text-xs text-gray-300 hover:text-gray-500"
                >
                    [Dev: Bypass Permissions]
                </button>
            </div>
        </div>
    );
};

const PermissionItem: React.FC<{
    icon: React.ReactNode;
    title: string;
    description: string;
    granted: boolean;
    onGrant: () => void;
    customAction?: React.ReactNode;
    showHelp: boolean;
    onToggleHelp: () => void;
    helpContent?: React.ReactNode;
}> = ({ icon, title, description, granted, onGrant, customAction, showHelp, onToggleHelp, helpContent }) => (
    <div className="flex flex-col bg-white rounded-xl shadow-sm border border-gray-100 transition-shadow hover:shadow-md overflow-hidden">
        <div className="flex items-center p-4">
            <div className={`p-3 rounded-full mr-4 ${granted ? 'bg-green-100 text-green-600' : 'bg-blue-50 text-blue-600'}`}>
                {granted ? <CheckCircle className="w-6 h-6" /> : icon}
            </div>
            <div className="flex-1">
                <h3 className="font-semibold text-gray-900">{title}</h3>
                <p className="text-sm text-gray-500">{description}</p>
                {customAction}
            </div>
            <div className="flex flex-col items-end space-y-2">
                <button
                    onClick={onGrant}
                    className={`px-4 py-2 text-sm font-medium rounded-lg transition-colors whitespace-nowrap ${granted ? 'text-gray-600 bg-gray-100 hover:bg-gray-200' : 'text-blue-600 bg-blue-50 hover:bg-blue-100'}`}
                >
                    {granted ? 'Manage Settings' : 'Open Settings'}
                </button>
                {!granted && (
                    <button
                        onClick={onToggleHelp}
                        className="flex items-center text-xs text-gray-400 hover:text-blue-500 transition-colors"
                    >
                        <HelpCircle className="w-3 h-3 mr-1" />
                        Don't see app?
                    </button>
                )}
            </div>
        </div>

        {/* Help Section */}
        {showHelp && !granted && (
            <div className="px-4 pb-4 bg-gray-50 border-t border-gray-100">
                {helpContent || (
                    <div className="mt-3 text-sm text-gray-600 space-y-2">
                        <p className="font-medium text-gray-800">If you don't see "Computer Recorder" in the list:</p>
                        <ol className="list-decimal pl-5 space-y-1">
                            <li>Look for the <Plus className="w-3 h-3 inline mx-1 border border-gray-400 rounded-sm" /> button at the bottom of the list.</li>
                            <li>Navigate to your <strong>Applications</strong> folder.</li>
                            <li>Select <strong>Computer Recorder</strong> and click <strong>Open</strong>.</li>
                            <li>Ensure the toggle switch next to the app is turned <strong>ON</strong>.</li>
                        </ol>
                    </div>
                )}
            </div>
        )}
    </div>
);

export default PermissionsPage;
