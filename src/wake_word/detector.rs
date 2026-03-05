//! OpenWakeWord-compatible wake word detector using tract-onnx.
//!
//! Three-stage ONNX pipeline:
//!   1. melspectrogram.onnx  — raw audio → mel spectrogram features
//!   2. embedding_model.onnx — mel features window → embedding vector
//!   3. <keyword>.onnx       — embedding window → detection probability
//!
//! Models: https://huggingface.co/davidscripka/openwakeword
//!
//! Pipeline dimensions below match the standard OpenWakeWord model set.
//! If you train custom models with different shapes, update the constants
//! and verify with `inspect_model()` at startup.

use anyhow::{Context, Result};
use std::collections::VecDeque;
use tract_onnx::prelude::*;

use crate::audio::capture::AudioFrame;
use crate::config::WakeWordConfig;

// ── OpenWakeWord pipeline constants ──────────────────────────────────
// Standard dimensions for models from davidscripka/openwakeword.
// The melspec model takes 80ms of audio and outputs one mel frame.
// Embeddings are computed over a sliding window of mel frames.
// The verifier scores a sliding window of embeddings.

/// Audio samples per melspec model invocation (80ms at 16kHz).
const MELSPEC_CHUNK_SAMPLES: usize = 1280;

/// Mel frequency bands per frame (melspec output width).
const MEL_BANDS: usize = 32;

/// Number of mel frames required for one embedding computation.
/// 76 frames × 80ms/frame ≈ 6 seconds of audio context.
const EMBED_WINDOW: usize = 76;

/// Embedding vector dimensionality (embedding model output width).
const EMBED_DIM: usize = 96;

/// Number of embedding vectors for one verifier prediction.
/// 16 embeddings × 80ms/embedding ≈ 1.3 seconds of embedding context.
const VERIFY_WINDOW: usize = 16;

// ── Types ────────────────────────────────────────────────────────────

/// Result of processing an audio frame through the detector.
#[derive(Debug)]
pub enum DetectionResult {
    /// No wake word detected in this frame.
    None,
    /// Wake word detected with the given confidence score.
    Detected { score: f32 },
}

// tract's runnable model type after optimization.
// If this doesn't compile with your tract version, run:
//   cargo doc -p tract-onnx --open
// and find the return type of TypedModel::into_runnable().
type TractPlan = SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>;

// ── Detector ─────────────────────────────────────────────────────────

pub struct WakeWordDetector {
    // ONNX model stages
    melspec: TractPlan,
    embedding: TractPlan,
    verifier: TractPlan,

    // Audio accumulator (f32, 16kHz mono) → feeds melspec model
    audio_buf: Vec<f32>,

    // Mel feature sliding window → feeds embedding model
    mel_ring: VecDeque<Vec<f32>>,

    // Embedding sliding window → feeds verifier model
    embed_ring: VecDeque<Vec<f32>>,

    // Detection threshold (0.0–1.0). Higher = fewer false positives.
    threshold: f32,

    // Tracks whether we've logged the warmup completion.
    warmed_up: bool,
}

impl WakeWordDetector {
    /// Load the three OpenWakeWord ONNX models and initialize the pipeline.
    ///
    /// Expected directory layout:
    ///   {models_dir}/melspectrogram.onnx
    ///   {models_dir}/embedding_model.onnx
    ///   {keyword_model_path}          (can be inside models_dir or elsewhere)
    pub fn new(config: &WakeWordConfig) -> Result<Self> {
        let models_dir = &config.models_dir;
        let keyword_path = &config.keyword_model_path;

        let melspec_path = format!("{models_dir}/melspectrogram.onnx");
        let embedding_path = format!("{models_dir}/embedding_model.onnx");

        tracing::info!("Loading OpenWakeWord models from {models_dir}");
        tracing::info!("  Keyword model: {keyword_path}");

        // Load each model with its expected input shape.
        // If a model's actual shape differs, tract will error here —
        // check the logged shapes and adjust the constants above.
        let melspec = load_onnx(&melspec_path, &[1, MELSPEC_CHUNK_SAMPLES])
            .context("melspectrogram model")?;
        let embedding = load_onnx(&embedding_path, &[1, EMBED_WINDOW, MEL_BANDS])
            .context("embedding model")?;
        let verifier = load_onnx(keyword_path, &[1, VERIFY_WINDOW, EMBED_DIM])
            .context("verifier/keyword model")?;

        tracing::info!("All wake word models loaded successfully");
        tracing::info!(
            "Pipeline: {}s audio → melspec({}×{}) → embed({}×{}) → verify → threshold={:.2}",
            MELSPEC_CHUNK_SAMPLES as f32 / 16000.0,
            EMBED_WINDOW,
            MEL_BANDS,
            VERIFY_WINDOW,
            EMBED_DIM,
            config.threshold,
        );

        let warmup_seconds =
            ((EMBED_WINDOW + VERIFY_WINDOW) as f32 * MELSPEC_CHUNK_SAMPLES as f32) / 16000.0;
        tracing::info!(
            "Warmup period: ~{:.1}s before first detection possible",
            warmup_seconds,
        );

        Ok(Self {
            melspec,
            embedding,
            verifier,
            audio_buf: Vec::with_capacity(MELSPEC_CHUNK_SAMPLES * 4),
            mel_ring: VecDeque::with_capacity(EMBED_WINDOW + 8),
            embed_ring: VecDeque::with_capacity(VERIFY_WINDOW + 4),
            threshold: config.threshold,
            warmed_up: false,
        })
    }

    /// Process one 30ms audio frame through the pipeline.
    ///
    /// Audio accumulates internally. When enough samples gather (~3 frames),
    /// the three-stage pipeline runs. Returns `Detected` when the verifier
    /// score exceeds the threshold.
    pub fn process(&mut self, frame: &AudioFrame) -> DetectionResult {
        // Convert i16 PCM → f32 (raw values, NOT normalized).
        // OpenWakeWord's melspectrogram model expects int16-range values.
        for &sample in &frame.samples {
            self.audio_buf.push(sample as f32);
        }

        // Process all available complete chunks through the pipeline.
        while self.audio_buf.len() >= MELSPEC_CHUNK_SAMPLES {
            let chunk: Vec<f32> = self.audio_buf.drain(..MELSPEC_CHUNK_SAMPLES).collect();

            // Stage 1: audio → mel features
            if let Err(e) = self.run_melspec(&chunk) {
                tracing::warn!("melspec inference failed: {e:#}");
                continue;
            }

            // Stage 2: mel features → embedding (only when window is full)
            if self.mel_ring.len() >= EMBED_WINDOW {
                if let Err(e) = self.run_embedding() {
                    tracing::warn!("embedding inference failed: {e:#}");
                    continue;
                }
            }

            // Stage 3: embeddings → detection score (only when window is full)
            if self.embed_ring.len() >= VERIFY_WINDOW {
                if !self.warmed_up {
                    tracing::info!("Wake word pipeline warmed up, detection active");
                    self.warmed_up = true;
                }

                match self.run_verifier() {
                    Ok(score) => {
                        tracing::trace!("wake word score: {score:.4}");
                        if score >= self.threshold {
                            // Clear buffers to prevent re-triggering on the
                            // same audio. The pipeline will need to warm up
                            // again, but that prevents rapid-fire detections.
                            self.embed_ring.clear();
                            return DetectionResult::Detected { score };
                        }
                    }
                    Err(e) => {
                        tracing::warn!("verifier inference failed: {e:#}");
                    }
                }
            }
        }

        DetectionResult::None
    }

    /// Samples per frame expected by the pipeline (30ms at 16kHz).
    /// Matches the AudioFrame size produced by the capture module.
    pub fn samples_per_frame(&self) -> usize {
        480 // 30ms at 16kHz
    }

    // ── Internal pipeline stages ─────────────────────────────────────

    /// Stage 1: Run the melspec model on one audio chunk.
    /// Appends the resulting mel frame(s) to mel_ring.
    fn run_melspec(&mut self, audio_chunk: &[f32]) -> Result<()> {
        debug_assert_eq!(audio_chunk.len(), MELSPEC_CHUNK_SAMPLES);

        let input = tract_ndarray::Array2::from_shape_vec(
            [1, MELSPEC_CHUNK_SAMPLES],
            audio_chunk.to_vec(),
        )?
        .into_dyn();

        let outputs = self.melspec.run(tvec!(input.into_tensor().into()))?;
        let mel = outputs[0].to_array_view::<f32>()?;
        let shape = mel.shape();

        // Output shape is typically [1, 1, 32] (one mel frame per chunk)
        // or [1, N, 32] (multiple frames if the model uses hop internally).
        // Handle both cases.
        match shape.len() {
            3 => {
                let n_frames = shape[1];
                let bands = shape[2];
                if bands != MEL_BANDS {
                    anyhow::bail!(
                        "melspec output has {bands} bands, expected {MEL_BANDS}"
                    );
                }
                for f in 0..n_frames {
                    let mel_frame: Vec<f32> =
                        (0..MEL_BANDS).map(|b| mel[[0, f, b]]).collect();
                    self.mel_ring.push_back(mel_frame);
                }
            }
            2 => {
                // [1, 32] — single frame, no time axis
                let bands = shape[1];
                if bands != MEL_BANDS {
                    anyhow::bail!(
                        "melspec output has {bands} bands, expected {MEL_BANDS}"
                    );
                }
                let mel_frame: Vec<f32> = (0..MEL_BANDS).map(|b| mel[[0, b]]).collect();
                self.mel_ring.push_back(mel_frame);
            }
            4 => {
                // tract may produce [1, 1, 5, 32] — batch, channel, frames, bands
                let n_frames = shape[2];
                let bands = shape[3];
                if bands != MEL_BANDS {
                    anyhow::bail!(
                        "melspec output has {bands} bands, expected {MEL_BANDS}"
                    );
                }
                for f in 0..n_frames {
                    let mel_frame: Vec<f32> =
                        (0..MEL_BANDS).map(|b| mel[[0, 0, f, b]]).collect();
                    self.mel_ring.push_back(mel_frame);
                }
            }
            _ => {
                anyhow::bail!(
                    "unexpected melspec output shape: {shape:?} (expected 2D, 3D, or 4D)"
                );
            }
        }

        // Keep at most EMBED_WINDOW frames (sliding window, drop oldest).
        while self.mel_ring.len() > EMBED_WINDOW {
            self.mel_ring.pop_front();
        }

        Ok(())
    }

    /// Stage 2: Run the embedding model on the current mel window.
    /// Appends the resulting embedding vector to embed_ring.
    fn run_embedding(&mut self) -> Result<()> {
        debug_assert!(self.mel_ring.len() >= EMBED_WINDOW);

        // Flatten the last EMBED_WINDOW mel frames into [1, 76, 32].
        let mut data = Vec::with_capacity(EMBED_WINDOW * MEL_BANDS);
        for frame in self.mel_ring.iter() {
            data.extend_from_slice(frame);
        }

        // If we have more than EMBED_WINDOW frames (shouldn't happen after
        // the cap in run_melspec, but defensive), take only the last ones.
        let expected = EMBED_WINDOW * MEL_BANDS;
        if data.len() > expected {
            data = data[data.len() - expected..].to_vec();
        }

        let input = tract_ndarray::Array3::from_shape_vec(
            [1, EMBED_WINDOW, MEL_BANDS],
            data,
        )?
        .into_dyn();

        let outputs = self.embedding.run(tvec!(input.into_tensor().into()))?;
        let emb = outputs[0].to_array_view::<f32>()?;
        let shape = emb.shape();

        // Output: [1, 96] or [1, 1, 96]
        let embed_vec: Vec<f32> = match shape.len() {
            2 => {
                if shape[1] != EMBED_DIM {
                    anyhow::bail!(
                        "embedding output dim={}, expected {EMBED_DIM}",
                        shape[1],
                    );
                }
                (0..EMBED_DIM).map(|i| emb[[0, i]]).collect()
            }
            3 => {
                if shape[2] != EMBED_DIM {
                    anyhow::bail!(
                        "embedding output dim={}, expected {EMBED_DIM}",
                        shape[2],
                    );
                }
                (0..EMBED_DIM).map(|i| emb[[0, 0, i]]).collect()
            }
            4 => {
                // tract may produce [1, 1, 1, 96]
                if shape[3] != EMBED_DIM {
                    anyhow::bail!(
                        "embedding output dim={}, expected {EMBED_DIM}",
                        shape[3],
                    );
                }
                (0..EMBED_DIM).map(|i| emb[[0, 0, 0, i]]).collect()
            }
            _ => {
                anyhow::bail!(
                    "unexpected embedding output shape: {shape:?}"
                );
            }
        };

        self.embed_ring.push_back(embed_vec);

        // Sliding window — keep last VERIFY_WINDOW embeddings.
        while self.embed_ring.len() > VERIFY_WINDOW {
            self.embed_ring.pop_front();
        }

        Ok(())
    }

    /// Stage 3: Run the verifier model on the current embedding window.
    /// Returns the detection probability (0.0–1.0).
    fn run_verifier(&self) -> Result<f32> {
        debug_assert!(self.embed_ring.len() >= VERIFY_WINDOW);

        // Flatten embeddings into [1, 16, 96].
        let mut data = Vec::with_capacity(VERIFY_WINDOW * EMBED_DIM);
        for emb in self.embed_ring.iter() {
            data.extend_from_slice(emb);
        }

        let input = tract_ndarray::Array3::from_shape_vec(
            [1, VERIFY_WINDOW, EMBED_DIM],
            data,
        )?
        .into_dyn();

        let outputs = self.verifier.run(tvec!(input.into_tensor().into()))?;
        let score_tensor = outputs[0].to_array_view::<f32>()?;
        let shape = score_tensor.shape();

        // Output: [1, 1] or [1]
        let score = match shape.len() {
            1 => score_tensor[[0]],
            2 => score_tensor[[0, 0]],
            _ => {
                anyhow::bail!("unexpected verifier output shape: {shape:?}");
            }
        };

        Ok(score)
    }
}

// ── Model loading ────────────────────────────────────────────────────

/// Load an ONNX model with a known input shape, optimize it, and
/// prepare it for inference.
///
/// The input shape must include the batch dimension (typically 1).
/// This is necessary because many ONNX models have a dynamic batch
/// dim that tract needs resolved before optimization.
fn load_onnx(path: &str, input_shape: &[usize]) -> Result<TractPlan> {
    tracing::info!("  Loading: {path}");
    tracing::info!("    Expected input shape: {input_shape:?}");

    // Convert usize shape to TDim for tract's inference fact.
    let shape: TVec<TDim> = input_shape.iter().map(|&d| (d as i64).into()).collect();

    let model = tract_onnx::onnx()
        .model_for_path(path)
        .with_context(|| format!("failed to read ONNX file: {path}"))?
        .with_input_fact(0, InferenceFact::dt_shape(f32::datum_type(), shape))
        .with_context(|| format!("failed to set input shape for {path}"))?
        .into_optimized()
        .with_context(|| format!("failed to optimize {path}"))?;

    // Log the output shape so the user can verify correctness.
    if let Ok(fact) = model.output_fact(0) {
        tracing::info!("    Output shape: {:?}", fact.shape);
    }

    let runnable = model
        .into_runnable()
        .with_context(|| format!("failed to make {path} runnable"))?;

    Ok(runnable)
}

/// Utility: inspect an ONNX model's input/output metadata without running it.
/// Call this from a test or CLI subcommand to verify model shapes before
/// wiring up the full pipeline.
#[allow(dead_code)]
pub fn inspect_model(path: &str) -> Result<()> {
    println!("Inspecting ONNX model: {path}");

    let model = tract_onnx::onnx()
        .model_for_path(path)
        .with_context(|| format!("failed to load {path}"))?;

    for (i, input) in model.input_outlets()?.iter().enumerate() {
        let fact = model.outlet_fact(*input)?;
        println!("  Input  {i}: {:?}", fact);
    }
    for (i, output) in model.output_outlets()?.iter().enumerate() {
        let fact = model.outlet_fact(*output)?;
        println!("  Output {i}: {:?}", fact);
    }

    Ok(())
}