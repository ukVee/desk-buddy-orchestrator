mod audio;
mod config;
mod pipeline;
mod vad;
mod wake_word;

use config::{CliArgs, Config};
use clap::Parser;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = CliArgs::parse();
    let config = Config::from_cli(args);

    // Initialize logging.
    tracing_subscriber::fmt()
        .with_env_filter(&config.log_level)
        .init();

    tracing::info!("JARVIS orchestrator starting");
    tracing::info!("Config: {config:#?}");

    // --- Audio device selection ---
    let input_device = audio::devices::find_input_device(&config.audio.capture_device)?;
    let input_config = audio::devices::log_input_configs(&input_device)?;

    // We also find the output device now, even though playback is Phase 3.
    // This validates it exists at startup rather than failing later.
    let _output_device = audio::devices::find_output_device(&config.audio.playback_device)?;
    tracing::info!("Output device found (playback deferred to Phase 3)");

    // --- Start audio capture ---
    let capture_config = audio::capture::CaptureConfig {
        target_rate: config.audio.target_sample_rate,
        frame_duration_ms: config.vad.frame_duration_ms,
    };

    let (_stream, audio_rx) = audio::capture::start_capture(
        &input_device,
        &input_config,
        capture_config,
    )?;
    // `_stream` must stay alive — dropping it stops capture.

    // --- Wake word detector ---
    let wake_word = if config.wake_word.bypass {
        tracing::warn!("Wake word detection bypassed via config");
        None
    } else {
        match wake_word::detector::WakeWordDetector::new(&config.wake_word) {
            Ok(detector) => {
                tracing::info!(
                    "Wake word detector loaded (wants {} samples/frame)",
                    detector.samples_per_frame(),
                );
                Some(detector)
            }
            Err(e) => {
                tracing::warn!("Failed to load wake word model: {e}");
                tracing::warn!("Falling back to wake word bypass mode");
                None
            }
        }
    };

    // --- VAD ---
    let vad = vad::detector::VadDetector::new(&config.vad)?;

    // --- Run pipeline ---
    tracing::info!("All subsystems initialized, starting voice pipeline");
    pipeline::voice::run_pipeline(&config, audio_rx, wake_word, vad).await;

    Ok(())
}
