mod platform;

fn main() {
    let platform = platform::FilePlatform::from_env();
    game::systems::app::build(platform, platform::primary_window()).run();
}
