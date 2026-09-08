use bevy::prelude::*;

/// Host capabilities the client needs that bevy doesn't provide.
/// Each target implements it in its binary and installs it as the [`ClientPlatform`] resource.
pub trait Platform: Send + Sync + 'static {
    fn load(&self, key: &str) -> Option<String>;
    fn save(&self, key: &str, value: &str);
    fn sync_window(&self, window: &mut Window);
    /// Whether the host actually holds the pointer lock the app last requested; hosts may release it
    /// on their own (a browser does on Escape) without the app being told through bevy.
    fn pointer_locked(&self, requested: bool) -> bool;
    /// Script commands queued by the host (an automation hook), oldest first.
    fn take_script_commands(&self) -> Vec<String>;
    /// The latest status the host may hand back to scripts.
    fn publish_status(&self, status: &str);
}

#[derive(Resource)]
pub struct ClientPlatform(pub Box<dyn Platform>);
