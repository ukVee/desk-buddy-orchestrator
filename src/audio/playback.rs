use anyhow::Result;

/// Plays WAV audio bytes through the specified output device.
///
/// Phase 3 implementation. For now, this is a stub that logs what it would do.
pub async fn play_wav(_device: &cpal::Device, wav_bytes: &[u8]) -> Result<()> {
    tracing::info!("play_wav: would play {} bytes of WAV audio", wav_bytes.len());
    // TODO Phase 3: decode WAV headers, open output stream, write PCM samples.
    // Need to handle 22050Hz Piper output → whatever the output device expects.
    // PCH card 1 should accept 22050Hz via plughw.
    Ok(())
}
