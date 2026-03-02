use anyhow::{Context, Result};

/// Client for the piper-tts k3s service.
///
/// Sends text via JSON POST to /synthesize,
/// receives WAV audio bytes.
#[derive(Clone)]
pub struct TtsClient {
    http: reqwest::Client,
    /// Base URL, e.g. "http://piper-tts.jarvis.svc.cluster.local:8081"
    base_url: String,
}

/// Request body for the piper-tts /synthesize endpoint.
#[derive(serde::Serialize)]
struct SynthesizeRequest {
    text: String,
}

impl TtsClient {
    pub fn new(base_url: &str) -> Self {
        Self {
            http: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(30))
                .build()
                .expect("failed to build HTTP client"),
            base_url: base_url.trim_end_matches('/').to_string(),
        }
    }

    /// Synthesize text to WAV audio bytes.
    ///
    /// Returns raw WAV data (including headers) as returned by Piper.
    /// Piper outputs 22050Hz mono 16-bit PCM WAV.
    pub async fn synthesize(&self, text: &str) -> Result<Vec<u8>> {
        let url = format!("{}/synthesize", self.base_url);

        tracing::info!(
            "TTS: synthesizing {} chars",
            text.len(),
        );
        tracing::debug!("TTS input: \"{}\"", text);

        let start = std::time::Instant::now();

        let resp = self.http
            .post(&url)
            .json(&SynthesizeRequest {
                text: text.to_string(),
            })
            .send()
            .await
            .context("TTS request failed")?;

        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            anyhow::bail!("TTS returned {status}: {body}");
        }

        let wav_bytes = resp
            .bytes()
            .await
            .context("Failed to read TTS response body")?
            .to_vec();

        let elapsed = start.elapsed();
        tracing::info!(
            "TTS: synthesized in {:.2}s ({} bytes WAV)",
            elapsed.as_secs_f64(),
            wav_bytes.len(),
        );

        if wav_bytes.len() < 44 {
            anyhow::bail!(
                "TTS response too small to be a valid WAV ({} bytes)",
                wav_bytes.len(),
            );
        }

        Ok(wav_bytes)
    }

    /// Health check — GET /health.
    pub async fn health(&self) -> Result<()> {
        let url = format!("{}/health", self.base_url);
        let resp = self.http
            .get(&url)
            .send()
            .await
            .context("TTS health check failed")?;
        if resp.status().is_success() {
            Ok(())
        } else {
            anyhow::bail!("TTS health check returned {}", resp.status())
        }
    }
}
