use anyhow::{Context, Result};
use cpal::traits::{DeviceTrait, StreamTrait};
use cpal::StreamConfig;
use std::sync::{Arc, Mutex};

/// Plays WAV audio bytes through the specified output device.
///
/// Decodes the WAV, adapts the audio to the device's native format
/// (rate conversion, channel upmix, sample format), then plays
/// and waits for completion.
pub async fn play_wav(device: &cpal::Device, wav_bytes: &[u8]) -> Result<()> {
    // Decode WAV from memory.
    let cursor = std::io::Cursor::new(wav_bytes);
    let mut reader = hound::WavReader::new(cursor).context("Failed to decode WAV")?;

    let spec = reader.spec();
    tracing::info!(
        "Playback: WAV is {}Hz {}ch {}bit {:?}",
        spec.sample_rate,
        spec.channels,
        spec.bits_per_sample,
        spec.sample_format,
    );

    // Read all samples as f32 (normalized).
    let samples_f32: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Int => {
            let max_val = (1i64 << (spec.bits_per_sample - 1)) as f32;
            reader
                .samples::<i32>()
                .map(|s| s.map(|v| v as f32 / max_val))
                .collect::<hound::Result<Vec<f32>>>()
                .context("Failed to read WAV samples")?
        }
        hound::SampleFormat::Float => {
            reader
                .samples::<f32>()
                .collect::<hound::Result<Vec<f32>>>()
                .context("Failed to read WAV samples")?
        }
    };

    if samples_f32.is_empty() {
        tracing::warn!("Playback: WAV contains no samples, skipping");
        return Ok(());
    }

    // Query what the device actually supports.
    let output_config = device
        .default_output_config()
        .context("No default output config for playback device")?;

    let device_rate = output_config.sample_rate().0;
    let device_channels = output_config.channels() as usize;
    let device_format = output_config.sample_format();

    tracing::info!(
        "Playback device native config: {}Hz {}ch {:?}",
        device_rate,
        device_channels,
        device_format,
    );

    // Adapt audio: WAV format → device format.
    let wav_rate = spec.sample_rate;
    let wav_channels = spec.channels as usize;

    // Step 1: Resample if rates differ.
    let resampled = if wav_rate != device_rate {
        tracing::info!(
            "Playback: resampling {}Hz → {}Hz",
            wav_rate,
            device_rate,
        );
        resample(&samples_f32, wav_rate, device_rate)?
    } else {
        samples_f32
    };

    // Step 2: Channel adapt (mono → stereo if needed).
    let adapted = if wav_channels == 1 && device_channels == 2 {
        tracing::info!("Playback: upmixing mono → stereo");
        mono_to_stereo(&resampled)
    } else if wav_channels == device_channels {
        resampled
    } else {
        tracing::warn!(
            "Playback: channel mismatch (WAV={}ch, device={}ch), using as-is",
            wav_channels,
            device_channels,
        );
        resampled
    };

    let duration_ms = (adapted.len() as u64 * 1000)
        / (device_rate as u64 * device_channels as u64);
    tracing::info!(
        "Playback: {} samples, ~{}ms duration (adapted to device format)",
        adapted.len(),
        duration_ms,
    );

    // Build stream config matching the device's native settings.
    let stream_config = StreamConfig {
        channels: device_channels as u16,
        sample_rate: cpal::SampleRate(device_rate),
        buffer_size: cpal::BufferSize::Default,
    };

    // Shared state for the playback callback.
    let playback_state = Arc::new(Mutex::new(PlaybackState {
        samples: adapted,
        position: 0,
        done: false,
    }));

    // Oneshot channel to signal playback completion.
    let (done_tx, done_rx) = tokio::sync::oneshot::channel::<()>();
    let done_tx = Arc::new(Mutex::new(Some(done_tx)));

    let err_fn = |err| tracing::error!("ALSA playback stream error: {err}");

    // Build the output stream in the device's native format.
    let stream = match device_format {
        cpal::SampleFormat::I16 => {
            let state_clone = Arc::clone(&playback_state);
            let done_clone = Arc::clone(&done_tx);
            device.build_output_stream(
                &stream_config,
                move |data: &mut [i16], _: &cpal::OutputCallbackInfo| {
                    fill_buffer_i16(&state_clone, &done_clone, data);
                },
                err_fn,
                None,
            )
        }
        cpal::SampleFormat::I32 => {
            let state_clone = Arc::clone(&playback_state);
            let done_clone = Arc::clone(&done_tx);
            device.build_output_stream(
                &stream_config,
                move |data: &mut [i32], _: &cpal::OutputCallbackInfo| {
                    fill_buffer_i32(&state_clone, &done_clone, data);
                },
                err_fn,
                None,
            )
        }
        cpal::SampleFormat::F32 => {
            let state_clone = Arc::clone(&playback_state);
            let done_clone = Arc::clone(&done_tx);
            device.build_output_stream(
                &stream_config,
                move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
                    fill_buffer_f32(&state_clone, &done_clone, data);
                },
                err_fn,
                None,
            )
        }
        format => {
            anyhow::bail!("Unsupported output sample format: {:?}", format);
        }
    }
    .context("Failed to build output stream")?;

    stream.play().context("Failed to start output stream")?;
    tracing::debug!("Playback stream started");

    // Wait for playback to finish.
    let _ = done_rx.await;

    // Grace period for ALSA to drain its buffer.
    tokio::time::sleep(std::time::Duration::from_millis(150)).await;

    drop(stream);
    tracing::info!("Playback complete");

    Ok(())
}

/// Simple linear resampling from one rate to another.
///
/// This is a basic implementation. For higher quality, rubato could be
/// used here too, but for TTS output (already synthetic) this is fine.
fn resample(samples: &[f32], from_rate: u32, to_rate: u32) -> Result<Vec<f32>> {
    if from_rate == to_rate {
        return Ok(samples.to_vec());
    }

    let ratio = to_rate as f64 / from_rate as f64;
    let output_len = (samples.len() as f64 * ratio).ceil() as usize;
    let mut output = Vec::with_capacity(output_len);

    for i in 0..output_len {
        let src_pos = i as f64 / ratio;
        let src_idx = src_pos as usize;
        let frac = (src_pos - src_idx as f64) as f32;

        let sample = if src_idx + 1 < samples.len() {
            samples[src_idx] * (1.0 - frac) + samples[src_idx + 1] * frac
        } else if src_idx < samples.len() {
            samples[src_idx]
        } else {
            0.0
        };

        output.push(sample);
    }

    Ok(output)
}

/// Duplicate mono samples into interleaved stereo.
fn mono_to_stereo(mono: &[f32]) -> Vec<f32> {
    let mut stereo = Vec::with_capacity(mono.len() * 2);
    for &sample in mono {
        stereo.push(sample);
        stereo.push(sample);
    }
    stereo
}

// --- Playback state and buffer fillers ---

struct PlaybackState {
    samples: Vec<f32>,
    position: usize,
    done: bool,
}

fn fill_buffer_f32(
    state: &Arc<Mutex<PlaybackState>>,
    done_tx: &Arc<Mutex<Option<tokio::sync::oneshot::Sender<()>>>>,
    data: &mut [f32],
) {
    let mut state = state.lock().unwrap();
    if state.done {
        data.fill(0.0);
        return;
    }

    let remaining = state.samples.len() - state.position;
    let to_copy = remaining.min(data.len());
    data[..to_copy].copy_from_slice(
        &state.samples[state.position..state.position + to_copy],
    );
    state.position += to_copy;

    if to_copy < data.len() {
        data[to_copy..].fill(0.0);
    }
    if state.position >= state.samples.len() {
        state.done = true;
        if let Some(tx) = done_tx.lock().unwrap().take() {
            let _ = tx.send(());
        }
    }
}

fn fill_buffer_i16(
    state: &Arc<Mutex<PlaybackState>>,
    done_tx: &Arc<Mutex<Option<tokio::sync::oneshot::Sender<()>>>>,
    data: &mut [i16],
) {
    let mut state = state.lock().unwrap();
    if state.done {
        data.fill(0);
        return;
    }

    let remaining = state.samples.len() - state.position;
    let to_copy = remaining.min(data.len());
    for (i, &sample) in state.samples[state.position..state.position + to_copy]
        .iter()
        .enumerate()
    {
        data[i] = (sample.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
    }
    state.position += to_copy;

    if to_copy < data.len() {
        data[to_copy..].fill(0);
    }
    if state.position >= state.samples.len() {
        state.done = true;
        if let Some(tx) = done_tx.lock().unwrap().take() {
            let _ = tx.send(());
        }
    }
}

fn fill_buffer_i32(
    state: &Arc<Mutex<PlaybackState>>,
    done_tx: &Arc<Mutex<Option<tokio::sync::oneshot::Sender<()>>>>,
    data: &mut [i32],
) {
    let mut state = state.lock().unwrap();
    if state.done {
        data.fill(0);
        return;
    }

    let remaining = state.samples.len() - state.position;
    let to_copy = remaining.min(data.len());
    for (i, &sample) in state.samples[state.position..state.position + to_copy]
        .iter()
        .enumerate()
    {
        data[i] = (sample.clamp(-1.0, 1.0) * i32::MAX as f32) as i32;
    }
    state.position += to_copy;

    if to_copy < data.len() {
        data[to_copy..].fill(0);
    }
    if state.position >= state.samples.len() {
        state.done = true;
        if let Some(tx) = done_tx.lock().unwrap().take() {
            let _ = tx.send(());
        }
    }
}