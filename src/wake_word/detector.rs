use crate::audio::capture::AudioFrame;
use crate::config::WakeWordConfig;
use anyhow::{Result, bail};

/// Wraps wake word detection.
/// Phase 2 stub — rustpotter removed due to candle-core build issues.
/// Will be re-added when we have a trained model and resolve the dependency.
pub struct WakeWordDetector;

/// Result of processing an audio frame through the detector.
#[derive(Debug)]
pub enum DetectionResult {
    /// No wake word detected in this frame.
    None,
    /// Wake word detected with the given confidence score.
    Detected { score: f32 },
}

impl WakeWordDetector {
    pub fn new(config: &WakeWordConfig) -> Result<Self> {
        bail!(
            "Wake word detection not yet available (rustpotter disabled). \
             Use --wakeword-bypass=true or set WAKEWORD_BYPASS=true."
        )
    }

    pub fn process(&mut self, _frame: &AudioFrame) -> DetectionResult {
        DetectionResult::None
    }

    pub fn samples_per_frame(&self) -> usize {
        480 // 30ms at 16kHz
    }
}
