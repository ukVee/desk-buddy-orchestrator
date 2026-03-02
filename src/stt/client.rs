use anyhow::{Context, Result};
use reqwest::multipart;

/// Client for the whisper-stt k3s service.
///
/// Sends WAV audio via multipart POST to /inference,
/// receives transcribed text.
#[derive(Clone)]
pub struct SttClient {
    http: reqwest::Client,
    /// Base URL, e.g. "http://whisper-stt.jarvis.svc.cluster.local:8080"
    base_url: String,
}

/// Response from whisper.cpp server's /inference endpoint.
#[derive(serde::Deserialize, Debug)]
struct WhisperResponse {
    text: String,
}

impl SttClient {
    pub fn new(base_url: &str) -> Self {
        Self {
            http: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(30))
                .build()
                .expect("failed to build HTTP client"),
            base_url: base_url.trim_end_matches('/').to_string(),
        }
    }

    /// Transcribe 16kHz mono i16 PCM samples to text.
    ///
    /// Encodes the samples as a WAV in memory, sends to whisper-stt,
    /// returns the trimmed transcription text.
    pub async fn transcribe(&self, samples: &[i16]) -> Result<String> {
        let wav_bytes = encode_wav(samples)?;

        let duration_ms = (samples.len() as u64 * 1000) / 16000;
        tracing::info!(
            "STT: sending {} ms utterance ({} bytes WAV) to {}",
            duration_ms,
            wav_bytes.len(),
            self.base_url,
        );

        let part = multipart::Part::bytes(wav_bytes)
            .file_name("audio.wav")
            .mime_str("audio/wav")
            .context("mime type")?;

        let form = multipart::Form::new().part("file", part);

        let url = format!("{}/inference", self.base_url);
        let start = std::time::Instant::now();

        let resp = self.http
            .post(&url)
            .multipart(form)
            .send()
            .await
            .context("STT request failed")?;

        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            anyhow::bail!("STT returned {status}: {body}");
        }

        let whisper_resp: WhisperResponse = resp
            .json()
            .await
            .context("Failed to parse whisper response")?;

        let text = whisper_resp.text.trim().to_string();
        let elapsed = start.elapsed();

        tracing::info!(
            "STT: transcribed in {:.1}s → \"{}\"",
            elapsed.as_secs_f64(),
            text,
        );

        Ok(text)
    }

    /// Health check — GET /health.
    pub async fn health(&self) -> Result<()> {
        let url = format!("{}/health", self.base_url);
        let resp = self.http.get(&url).send().await.context("STT health check failed")?;
        if resp.status().is_success() {
            Ok(())
        } else {
            anyhow::bail!("STT health check returned {}", resp.status())
        }
    }
}

/// Encode 16kHz mono i16 PCM samples into an in-memory WAV.
fn encode_wav(samples: &[i16]) -> Result<Vec<u8>> {
    let mut cursor = std::io::Cursor::new(Vec::new());

    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 16000,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };

    let mut writer = hound::WavWriter::new(&mut cursor, spec)
        .context("Failed to create WAV writer")?;

    for &sample in samples {
        writer.write_sample(sample).context("WAV write sample")?;
    }
    writer.finalize().context("WAV finalize")?;

    Ok(cursor.into_inner())
}
