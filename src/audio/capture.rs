use anyhow::{Context, Result};
use cpal::traits::{DeviceTrait, StreamTrait};
use cpal::{SampleFormat, StreamConfig};
use rubato::{SincFixedIn, SincInterpolationParameters, SincInterpolationType, Resampler, WindowFunction};
use std::sync::{Arc, Mutex};
use tokio::sync::mpsc;

/// A chunk of 16kHz mono i16 PCM audio, ready for wake word / VAD processing.
/// Each chunk is exactly `frame_duration_ms` worth of audio.
pub struct AudioFrame {
    /// 16kHz mono signed 16-bit samples.
    pub samples: Vec<i16>,
}

/// Configuration for the audio capture pipeline.
pub struct CaptureConfig {
    /// The target output sample rate (should be 16000).
    pub target_rate: u32,
    /// Frame duration in ms. Determines AudioFrame size.
    /// 30ms at 16kHz = 480 samples per frame.
    pub frame_duration_ms: usize,
}

impl CaptureConfig {
    /// Number of samples per output frame.
    pub fn frame_size(&self) -> usize {
        (self.target_rate as usize * self.frame_duration_ms) / 1000
    }
}

/// Starts the audio capture pipeline.
///
/// Opens the given device, captures audio, resamples to 16kHz mono,
/// and sends frames through the returned channel.
///
/// Returns the cpal Stream (must be kept alive) and a receiver for AudioFrames.
pub fn start_capture(
    device: &cpal::Device,
    device_config: &cpal::SupportedStreamConfig,
    capture_config: CaptureConfig,
) -> Result<(cpal::Stream, mpsc::Receiver<AudioFrame>)> {
    let sample_format = device_config.sample_format();
    let channels = device_config.channels() as usize;
    let source_rate = device_config.sample_rate().0;
    let target_rate = capture_config.target_rate;
    let frame_size = capture_config.frame_size();

    tracing::info!(
        "Capture pipeline: {}Hz {}ch {:?} → {}Hz mono i16, frame={}ms ({}samples)",
        source_rate, channels, sample_format,
        target_rate, capture_config.frame_duration_ms, frame_size,
    );

    // Channel from cpal thread → async world.
    // Bounded to ~1 second of frames to handle backpressure.
    let buffer_frames = 1000 / capture_config.frame_duration_ms;
    let (tx, rx) = mpsc::channel::<AudioFrame>(buffer_frames);

    // Shared state between cpal callback and the resampling logic.
    // The cpal callback pushes raw mono f32 samples into this buffer.
    // When enough accumulate, we resample and emit a frame.
    let processor = Arc::new(Mutex::new(CaptureProcessor::new(
        source_rate,
        target_rate,
        channels,
        frame_size,
        tx,
    )?));

    let processor_clone = Arc::clone(&processor);
    let err_fn = |err| tracing::error!("ALSA capture stream error: {err}");

    // Open the stream in the device's native format.
    let stream_config: StreamConfig = device_config.clone().into();
    let stream = match sample_format {
        SampleFormat::I16 => device.build_input_stream(
            &stream_config,
            move |data: &[i16], _: &cpal::InputCallbackInfo| {
                let f32_data: Vec<f32> = data.iter().map(|&s| s as f32 / i16::MAX as f32).collect();
                if let Ok(mut proc) = processor_clone.lock() {
                    proc.push_samples(&f32_data);
                }
            },
            err_fn,
            None,
        ),
        SampleFormat::F32 => device.build_input_stream(
            &stream_config,
            move |data: &[f32], _: &cpal::InputCallbackInfo| {
                if let Ok(mut proc) = processor_clone.lock() {
                    proc.push_samples(data);
                }
            },
            err_fn,
            None,
        ),
        format => {
            anyhow::bail!("Unsupported sample format: {:?}", format);
        }
    }
    .context("Failed to build input stream")?;

    stream.play().context("Failed to start input stream")?;
    tracing::info!("Audio capture stream started");

    Ok((stream, rx))
}

/// Internal processor that lives on the cpal callback thread.
/// Accumulates raw samples, converts stereo→mono, resamples, and emits frames.
struct CaptureProcessor {
    channels: usize,
    /// Buffer of mono f32 samples at the source sample rate.
    mono_buffer: Vec<f32>,
    /// How many source-rate mono samples we need before running the resampler.
    /// This is the resampler's input chunk size.
    resample_input_size: usize,
    /// The resampler instance.
    resampler: SincFixedIn<f32>,
    /// Buffer of resampled 16kHz samples waiting to be chunked into frames.
    output_buffer: Vec<f32>,
    /// Samples per output frame (e.g. 480 for 30ms at 16kHz).
    frame_size: usize,
    /// Channel to send completed frames.
    tx: mpsc::Sender<AudioFrame>,
}

impl CaptureProcessor {
    fn new(
        source_rate: u32,
        target_rate: u32,
        channels: usize,
        frame_size: usize,
        tx: mpsc::Sender<AudioFrame>,
    ) -> Result<Self> {
        let resample_ratio = target_rate as f64 / source_rate as f64;

        // Configure rubato sinc resampler.
        // chunk_size here is the number of *output* frames per resampling call.
        // We use the frame_size so each resample call produces exactly one frame.
        let params = SincInterpolationParameters {
            sinc_len: 64,
            f_cutoff: 0.95,
            interpolation: SincInterpolationType::Linear,
            oversampling_factor: 128,
            window: WindowFunction::BlackmanHarris2,
        };

        let resampler = SincFixedIn::new(
            resample_ratio,
            1.0,    // max relative ratio adjustment (none)
            params,
            frame_size, // chunk size (output frames per call — we set to our frame size)
            1,          // mono (we do stereo→mono before resampling)
        )
        .context("Failed to create resampler")?;

        let resample_input_size = resampler.input_frames_next();

        tracing::debug!(
            "Resampler: ratio={:.4}, input_chunk={}, output_chunk={}",
            resample_ratio, resample_input_size, frame_size,
        );

        Ok(Self {
            channels,
            mono_buffer: Vec::with_capacity(resample_input_size * 2),
            resample_input_size,
            resampler,
            output_buffer: Vec::with_capacity(frame_size * 2),
            frame_size,
            tx,
        })
    }

    /// Called from the cpal audio thread with raw interleaved samples.
    fn push_samples(&mut self, interleaved: &[f32]) {
        // Step 1: Stereo → mono (average channels).
        // If already mono, this is a no-op copy.
        for chunk in interleaved.chunks_exact(self.channels) {
            let mono_sample: f32 = chunk.iter().sum::<f32>() / self.channels as f32;
            self.mono_buffer.push(mono_sample);
        }

        // Step 2: When we have enough mono samples, resample.
        while self.mono_buffer.len() >= self.resample_input_size {
            let input_chunk: Vec<f32> = self.mono_buffer
                .drain(..self.resample_input_size)
                .collect();

            match self.resampler.process(&[&input_chunk], None) {
                Ok(output) => {
                    if let Some(channel_data) = output.first() {
                        self.output_buffer.extend_from_slice(channel_data);
                    }
                }
                Err(e) => {
                    tracing::warn!("Resample error: {e}");
                    continue;
                }
            }

            // Update for next call — rubato may vary input size slightly.
            self.resample_input_size = self.resampler.input_frames_next();
        }

        // Step 3: Emit complete frames from the output buffer.
        while self.output_buffer.len() >= self.frame_size {
            let frame_f32: Vec<f32> = self.output_buffer.drain(..self.frame_size).collect();

            // Convert f32 [-1.0, 1.0] → i16
            let samples: Vec<i16> = frame_f32
                .iter()
                .map(|&s| {
                    let clamped = s.clamp(-1.0, 1.0);
                    (clamped * i16::MAX as f32) as i16
                })
                .collect();

            let frame = AudioFrame { samples };

            // Non-blocking send. If the consumer is behind, drop the frame.
            if self.tx.try_send(frame).is_err() {
                tracing::trace!("Audio frame dropped (consumer behind)");
            }
        }
    }
}
