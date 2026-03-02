use crate::audio::capture::AudioFrame;
use crate::config::VadConfig;
use anyhow::{Context, Result};
use webrtc_vad::{Vad, SampleRate, VadMode};

/// A complete utterance captured between speech onset and silence timeout.
pub struct Utterance {
    /// Accumulated 16kHz mono i16 PCM samples.
    pub samples: Vec<i16>,
    /// Duration in milliseconds.
    pub duration_ms: u64,
}

/// Voice Activity Detector with utterance segmentation.
///
/// Wraps webrtc-vad and adds silence timeout logic to detect
/// complete utterances (start of speech → end of speech).
pub struct VadDetector {
    vad: Vad,
    config: VadConfig,
    state: VadState,
    /// Accumulated speech samples for the current utterance.
    utterance_buffer: Vec<i16>,
    /// Count of consecutive non-speech frames since last speech.
    silence_frames: u32,
    /// Number of silent frames needed to trigger end-of-speech.
    silence_threshold: u32,
    /// Minimum speech frames before we consider it a valid utterance
    /// (filters out clicks, pops, brief noise).
    min_speech_frames: u32,
    /// How many speech frames we've seen in the current utterance.
    speech_frame_count: u32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum VadState {
    /// Waiting for speech to start.
    Waiting,
    /// Currently in a speech segment, accumulating frames.
    InSpeech,
}

/// What the VAD returns after processing a frame.
pub enum VadResult {
    /// Frame processed, no complete utterance yet.
    Continue,
    /// A complete utterance has been detected (speech → silence timeout).
    CompleteUtterance(Utterance),
}

impl VadDetector {
    pub fn new(config: &VadConfig) -> Result<Self> {
        let mut vad = Vad::new_with_rate(SampleRate::Rate16kHz);

        let mode = match config.aggressiveness {
            0 => VadMode::Quality,
            1 => VadMode::LowBitrate,
            2 => VadMode::Aggressive,
            _ => VadMode::VeryAggressive,
        };
        vad.set_mode(mode);

        // How many silent frames = silence_timeout_ms?
        // Each frame is frame_duration_ms long.
        let silence_threshold = (config.silence_timeout_ms as u32)
            / (config.frame_duration_ms as u32);

        // Require at least ~200ms of speech to be a valid utterance.
        let min_speech_frames = 200 / config.frame_duration_ms as u32;

        tracing::info!(
            "VAD ready: aggressiveness={}, silence_timeout={}ms ({} frames), min_speech={} frames",
            config.aggressiveness, config.silence_timeout_ms, silence_threshold, min_speech_frames,
        );
        Ok(Self {
            vad,
            config: config.clone(),
            state: VadState::Waiting,
            utterance_buffer: Vec::new(),
            silence_frames: 0,
            silence_threshold,
            min_speech_frames,
            speech_frame_count: 0,
        })
    }

    /// Process a 16kHz mono i16 audio frame.
    ///
    /// Returns `VadResult::CompleteUtterance` when a full utterance
    /// has been captured (speech followed by sufficient silence).
    pub fn process(&mut self, frame: &AudioFrame) -> VadResult {
        let is_speech = self.vad.is_voice_segment(&frame.samples)
            .unwrap_or(false);

        match self.state {
            VadState::Waiting => {
                if is_speech {
                    tracing::debug!("VAD: speech onset detected");
                    self.state = VadState::InSpeech;
                    self.utterance_buffer.clear();
                    self.utterance_buffer.extend_from_slice(&frame.samples);
                    self.silence_frames = 0;
                    self.speech_frame_count = 1;
                }
                VadResult::Continue
            }

            VadState::InSpeech => {
                // Always append the frame (including silence frames during
                // the timeout window — they're part of the utterance).
                self.utterance_buffer.extend_from_slice(&frame.samples);

                if is_speech {
                    self.silence_frames = 0;
                    self.speech_frame_count += 1;
                } else {
                    self.silence_frames += 1;
                }

                if self.silence_frames >= self.silence_threshold {
                    // Silence timeout reached — end of utterance.
                    self.state = VadState::Waiting;

                    if self.speech_frame_count < self.min_speech_frames {
                        tracing::debug!(
                            "VAD: utterance too short ({} frames), discarding",
                            self.speech_frame_count,
                        );
                        self.utterance_buffer.clear();
                        return VadResult::Continue;
                    }

                    // Trim trailing silence from the buffer.
                    // We accumulated `silence_threshold` frames of silence;
                    // remove them (or most of them — keep a tiny tail).
                    let trailing_silence_samples =
                        (self.silence_threshold as usize - 1) * frame.samples.len();
                    let trim_to = self.utterance_buffer
                        .len()
                        .saturating_sub(trailing_silence_samples);
                    self.utterance_buffer.truncate(trim_to);

                    let duration_ms = (self.utterance_buffer.len() as u64 * 1000) / 16000;
                    tracing::info!(
                        "VAD: utterance complete, {} samples ({} ms)",
                        self.utterance_buffer.len(),
                        duration_ms,
                    );

                    let utterance = Utterance {
                        samples: std::mem::take(&mut self.utterance_buffer),
                        duration_ms,
                    };

                    VadResult::CompleteUtterance(utterance)
                } else {
                    VadResult::Continue
                }
            }
        }
    }

    /// Reset the detector to the Waiting state, discarding any in-progress utterance.
    pub fn reset(&mut self) {
        self.state = VadState::Waiting;
        self.utterance_buffer.clear();
        self.silence_frames = 0;
        self.speech_frame_count = 0;
    }
}
