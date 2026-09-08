use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Mutex;

use bevy::prelude::*;
use game::core::platform::Platform;
use serde::Deserialize;

#[derive(Deserialize)]
struct Env {
    /// Where the snapshot store lives; one JSON object keyed like the browser's localStorage.
    spin_save_path: PathBuf,
    /// Script commands to run at startup, one JSON object per line.
    spin_script_path: Option<PathBuf>,
}

/// A desktop host: storage is a JSON file, the pointer lock is whatever was asked for, and a
/// script file stands in for the browser's automation hook.
pub struct FilePlatform {
    path: PathBuf,
    script: Mutex<Vec<String>>,
}

impl FilePlatform {
    pub fn from_env() -> FilePlatform {
        let env: Env = envy::from_env().expect("SPIN_SAVE_PATH must be set");
        let script = env
            .spin_script_path
            .and_then(|path| std::fs::read_to_string(path).ok())
            .map(|text| text.lines().map(str::to_owned).collect())
            .unwrap_or_default();
        FilePlatform {
            path: env.spin_save_path,
            script: Mutex::new(script),
        }
    }

    fn read_store(&self) -> BTreeMap<String, String> {
        std::fs::read_to_string(&self.path)
            .ok()
            .and_then(|text| serde_json::from_str(&text).ok())
            .unwrap_or_default()
    }
}

impl Platform for FilePlatform {
    fn load(&self, key: &str) -> Option<String> {
        self.read_store().remove(key)
    }

    fn save(&self, key: &str, value: &str) {
        let mut store = self.read_store();
        store.insert(key.to_owned(), value.to_owned());
        if let Ok(text) = serde_json::to_string(&store) {
            let _ = std::fs::write(&self.path, text);
        }
    }

    fn sync_window(&self, _window: &mut Window) {}

    fn pointer_locked(&self, requested: bool) -> bool {
        requested
    }

    fn take_script_commands(&self) -> Vec<String> {
        let mut script = self.script.lock().unwrap_or_else(|e| e.into_inner());
        std::mem::take(&mut *script)
    }

    fn publish_status(&self, _status: &str) {}
}

pub fn primary_window() -> Window {
    Window {
        title: "Spin-gravity wheel".to_owned(),
        resolution: (1400, 900).into(),
        ..default()
    }
}
