use crate::audio::capture::AudioFrame;
use crate::config::Config;
use crate::llm::client::LlmClient;
use crate::stt::client::SttClient;
use crate::tts::client::TtsClient;
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
    /// Utterance captured, processing STT -> LLM -> TTS.
    Processing,
    /// Playing TTS audio.
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

/// Everything the pipeline needs to process utterances end-to-end.
pub struct PipelineServices {
    pub stt: SttClient,
    pub llm: LlmClient,
    pub tts: TtsClient,
    pub output_device: cpal::Device,
}

/// Runs the voice pipeline loop.
///
/// Consumes audio frames and drives the state machine:
///   Idle (wake word) -> Listening (VAD) -> Processing (STT/LLM/TTS) -> Speaking -> Idle
pub async fn run_pipeline(
    config: &Config,
    mut audio_rx: mpsc::Receiver<AudioFrame>,
    wake_word: Option<WakeWordDetector>,
    mut vad: VadDetector,
    services: PipelineServices,
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
                            tracing::info!(
                                ">>> Wake word detected (score={score:.3}), now LISTENING"
                            );
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

                        state = PipelineState::Processing;
                        tracing::info!("Pipeline → {state}");

                        // Process the full voice loop.
                        // On error, log it and return to listening — don't crash.
                        match process_utterance(&utterance, &services).await {
                            Ok(()) => {
                                tracing::info!("Voice loop completed successfully");
                            }
                            Err(e) => {
                                tracing::error!("Voice loop failed: {e:#}");
                            }
                        }

                        // Return to listening or idle.
                        state = if bypass_wake_word {
                            PipelineState::Listening
                        } else {
                            PipelineState::Idle
                        };
                        tracing::info!("Pipeline → {state}");
                    }
                }
            }

            PipelineState::Processing | PipelineState::Speaking => {
                // During processing/speaking, we drop incoming audio frames.
                // This is intentional — we don't want to queue up utterances
                // while the system is busy responding.
                // Phase 7 could add barge-in (interrupt on wake word during speaking).
            }
        }
    }

    tracing::warn!("Audio channel closed, pipeline shutting down");
}

/// The full voice loop: STT → LLM → TTS → playback.
///
/// This is called from the Processing state. On success, the pipeline
/// transitions through Speaking and back to Idle/Listening.
async fn process_utterance(
    utterance: &Utterance,
    services: &PipelineServices,
) -> anyhow::Result<()> {
    let total_start = std::time::Instant::now();

    // ① STT: audio → text
    let transcript = services.stt.transcribe(&utterance.samples).await?;

    if transcript.is_empty() {
        tracing::warn!("STT returned empty transcript, skipping");
        return Ok(());
    }

    // Filter out common whisper hallucinations on silence/noise.
    if is_hallucination(&transcript) {
        tracing::warn!("STT likely hallucinated: \"{transcript}\", skipping");
        return Ok(());
    }

    // ② LLM: text → response
    // For Phase 3, we use the simple ask() helper with default system prompt.
    // Phase 4 will add context management (conversation history).
    let llm_response = services.llm.ask(&transcript).await?;

    if llm_response.text.is_empty() {
        tracing::warn!("LLM returned empty response, skipping TTS");
        return Ok(());
    }

    // ③ TTS: response text → WAV audio
    let wav_bytes = services.tts.synthesize(&llm_response.text).await?;

    // ④ Playback: WAV → speakers
    tracing::info!("Pipeline → SPEAKING");
    crate::audio::playback::play_wav(&services.output_device, &wav_bytes).await?;

    let total_elapsed = total_start.elapsed();
    tracing::info!(
        "Full voice loop completed in {:.1}s (utterance: {}ms, transcript: \"{}\")",
        total_elapsed.as_secs_f64(),
        utterance.duration_ms,
        truncate(&transcript, 80),
    );

    // Debug: also save the last utterance WAV for inspection.
    if let Err(e) = save_debug_wav(utterance) {
        tracing::debug!("Could not save debug WAV: {e}");
    }

    Ok(())
}

/// Detect common whisper.cpp hallucinations on silence/noise.
///
/// whisper.cpp sometimes generates repetitive or boilerplate text
/// when given silence or background noise. These are well-known patterns.
fn is_hallucination(text: &str) -> bool {
    let lower = text.to_lowercase();
    let trimmed = lower.trim();

    // Empty or whitespace-only.
    if trimmed.is_empty() {
        return true;
    }

    // Common whisper hallucination patterns:
    let hallucination_patterns = [
        "thank you",
        "thanks for watching",
        "please subscribe",
        "like and subscribe",
        "see you next time",
        "bye bye",
        "you",  // single word "you" on noise
        "(bell",
        "[music]",
        "[silence]",
        "[blank_audio]",
        "...",
    ];

    for pattern in hallucination_patterns {
        if trimmed == pattern || trimmed == format!("{pattern}.") {
            return true;
        }
    }

    // Very short output that's just punctuation or a single short word.
    let alpha_chars: usize = trimmed.chars().filter(|c| c.is_alphabetic()).count();
    if alpha_chars <= 2 {
        return true;
    }

    false
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
    tracing::debug!("Debug utterance saved to {path}");

    Ok(())
}

/// Truncate a string for log output.
fn truncate(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}...", &s[..max_len])
    }
}
