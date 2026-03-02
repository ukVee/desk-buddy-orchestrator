use crate::audio::capture::AudioFrame;
use crate::config::Config;
use crate::vad::detector::{VadDetector, VadResult, Utterance};
use crate::wake_word::detector::{WakeWordDetector, DetectionResult};
use tokio::sync::mpsc;

/// Pipeline states.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PipelineState {
    /// Waiting for wake word.
    Idle,
    /// Wake word detected, VAD is accumulating speech.
    Listening,
    /// Utterance captured, processing STT → LLM → TTS (Phase 3+).
    Processing,
    /// Playing TTS audio (Phase 3+).
    Speaking,
}

impl std::fmt::Display for PipelineState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Idle => write!(f, "IDLE"),
            Self::Listening => write!(f, "LISTENING"),
            Self::Processing => write!(f, "PROCESSING"),
            Self::Speaking => write!(f, "SPEAKING"),
        }
    }
}

/// Runs the voice pipeline loop.
///
/// Consumes audio frames and drives the state machine:
///   Idle (wake word) → Listening (VAD) → [Processing → Speaking → Idle]
///
/// In Phase 2, the Processing/Speaking states just log the utterance
/// and return to Idle.
pub async fn run_pipeline(
    config: &Config,
    mut audio_rx: mpsc::Receiver<AudioFrame>,
    wake_word: Option<WakeWordDetector>,
    mut vad: VadDetector,
) {
    let bypass_wake_word = config.wake_word.bypass || wake_word.is_none();
    let mut wake_word = wake_word;
    let mut state = if bypass_wake_word {
        tracing::warn!("Wake word bypassed — pipeline starts in LISTENING mode");
        PipelineState::Listening
    } else {
        PipelineState::Idle
    };

    tracing::info!("Pipeline started in {state} state");

    while let Some(frame) = audio_rx.recv().await {
        match state {
            PipelineState::Idle => {
                // Feed frames to wake word detector.
                if let Some(ref mut detector) = wake_word {
                    match detector.process(&frame) {
                        DetectionResult::Detected { score } => {
                            tracing::info!(">>> Wake word detected (score={score:.3}), now LISTENING");
                            state = PipelineState::Listening;
                            vad.reset();
                        }
                        DetectionResult::None => {}
                    }
                }
            }

            PipelineState::Listening => {
                match vad.process(&frame) {
                    VadResult::Continue => {}
                    VadResult::CompleteUtterance(utterance) => {
                        tracing::info!(
                            ">>> Utterance captured: {} samples ({} ms)",
                            utterance.samples.len(),
                            utterance.duration_ms,
                        );

                        // Phase 2: just log the utterance and go back to idle.
                        // Phase 3: this becomes state = PipelineState::Processing
                        // and we hand the utterance off to STT → LLM → TTS.
                        handle_utterance_phase2(&utterance).await;

                        state = if bypass_wake_word {
                            // In bypass mode, go back to listening immediately.
                            PipelineState::Listening
                        } else {
                            PipelineState::Idle
                        };

                        tracing::info!("Pipeline returned to {state}");
                    }
                }
            }

            PipelineState::Processing | PipelineState::Speaking => {
                // Phase 3+: these states will be populated.
                // For now, this shouldn't happen.
                tracing::warn!("Unexpected state {state} in Phase 2 pipeline, resetting to Idle");
                state = PipelineState::Idle;
            }
        }
    }

    tracing::warn!("Audio channel closed, pipeline shutting down");
}

/// Phase 2 placeholder: log utterance stats and optionally save as WAV for debugging.
async fn handle_utterance_phase2(utterance: &Utterance) {
    tracing::info!(
        "Phase 2 stub — would send {} ms utterance to STT",
        utterance.duration_ms,
    );

    // Debug: save the utterance to a WAV file so you can listen back
    // and verify the capture + VAD pipeline is working correctly.
    if let Err(e) = save_debug_wav(utterance) {
        tracing::debug!("Could not save debug WAV: {e}");
    }
}

/// Save utterance as a WAV file for debugging purposes.
fn save_debug_wav(utterance: &Utterance) -> anyhow::Result<()> {
    use hound::{WavSpec, WavWriter};

    let path = "/data/debug_utterance.wav";
    let spec = WavSpec {
        channels: 1,
        sample_rate: 16000,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };

    let mut writer = WavWriter::create(path, spec)?;
    for &sample in &utterance.samples {
        writer.write_sample(sample)?;
    }
    writer.finalize()?;

    tracing::info!("Debug utterance saved to {path}");
    Ok(())
}
