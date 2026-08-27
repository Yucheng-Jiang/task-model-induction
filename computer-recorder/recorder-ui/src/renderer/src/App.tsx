import { useState } from 'react'
import PermissionsPage from './components/PermissionsPage'
import RecordingControl from './components/RecordingControl'

function App(): JSX.Element {
    const [hasPermissions, setHasPermissions] = useState(false);

    return (
        <div className="font-sans">
            {!hasPermissions ? (
                <PermissionsPage onComplete={() => setHasPermissions(true)} />
            ) : (
                <RecordingControl />
            )}
        </div>
    )
}

export default App
