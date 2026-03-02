mod audio;
mod config;
mod llm;
mod pipeline;
mod stt;
mod tts;
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

    let output_device = audio::devices::find_output_device(&config.audio.playback_device)?;
    tracing::info!("Output device ready for playback");

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

    // --- Service clients ---
    let stt = stt::client::SttClient::new(&config.services.whisper_url);
    let llm = llm::client::LlmClient::new(&config.services.ollama_url, &config.services.ollama_model);
    let tts = tts::client::TtsClient::new(&config.services.piper_url);

    // Optional: health-check upstream services at startup.
    // Non-fatal — services might come up after the orchestrator.
    check_service_health(&stt, &llm, &tts).await;

    let services = pipeline::voice::PipelineServices {
        stt,
        llm,
        tts,
        output_device,
    };

    // --- Run pipeline ---
    tracing::info!("All subsystems initialized, starting voice pipeline");
    pipeline::voice::run_pipeline(&config, audio_rx, wake_word, vad, services).await;

    Ok(())
}

/// Best-effort health checks on upstream services.
/// Logs warnings on failure but doesn't block startup.
async fn check_service_health(
    stt: &stt::client::SttClient,
    llm: &llm::client::LlmClient,
    tts: &tts::client::TtsClient,
) {
    tracing::info!("Checking upstream service health...");

    match stt.health().await {
        Ok(()) => tracing::info!("  whisper-stt: OK"),
        Err(e) => tracing::warn!("  whisper-stt: UNREACHABLE ({e})"),
    }

    match tts.health().await {
        Ok(()) => tracing::info!("  piper-tts: OK"),
        Err(e) => tracing::warn!("  piper-tts: UNREACHABLE ({e})"),
    }

    match llm.health().await {
        Ok(()) => tracing::info!("  ollama: OK"),
        Err(e) => tracing::warn!("  ollama: UNREACHABLE ({e})"),
    }
}
