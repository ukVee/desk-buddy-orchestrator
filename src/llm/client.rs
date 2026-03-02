use anyhow::{Context, Result};
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};

/// Client for the Ollama LLM service on the laptop.
///
/// Sends chat messages to /api/chat with streaming enabled,
/// accumulates tokens as they arrive, returns the full response.
#[derive(Clone)]
pub struct LlmClient {
    http: reqwest::Client,
    /// Base URL, e.g. "http://192.168.0.XXX:11434"
    base_url: String,
    /// Model name, e.g. "mistral:latest"
    model: String,
}

/// A message in the Ollama chat format.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

/// The request body for Ollama /api/chat.
#[derive(Serialize)]
struct ChatRequest {
    model: String,
    messages: Vec<ChatMessage>,
    stream: bool,
    // Phase 5: add `tools` field here for function calling.
}

/// A single streaming chunk from Ollama.
#[derive(Deserialize)]
struct StreamChunk {
    message: Option<ChunkMessage>,
    done: bool,
    // Present on the final chunk:
    #[serde(default)]
    total_duration: Option<u64>,
    #[serde(default)]
    eval_count: Option<u64>,
    #[serde(default)]
    eval_duration: Option<u64>,
}

#[derive(Deserialize)]
struct ChunkMessage {
    #[serde(default)]
    content: String,
    // Phase 5: tool_calls will appear here.
}

/// Result of an LLM chat completion.
#[derive(Debug)]
pub struct LlmResponse {
    /// The full accumulated text response.
    pub text: String,
    /// Tokens generated (from Ollama stats, if available).
    pub eval_count: Option<u64>,
    /// Token generation duration in nanoseconds (from Ollama stats).
    pub eval_duration_ns: Option<u64>,
    // Phase 5: pub tool_calls: Vec<ToolCall>,
}

impl LlmResponse {
    /// Tokens per second, if stats are available.
    pub fn tokens_per_sec(&self) -> Option<f64> {
        match (self.eval_count, self.eval_duration_ns) {
            (Some(count), Some(dur)) if dur > 0 => {
                Some(count as f64 / (dur as f64 / 1_000_000_000.0))
            }
            _ => None,
        }
    }
}

/// Default system prompt for JARVIS.
/// This lives here for now; Phase 4 will move it to the context manager.
pub const DEFAULT_SYSTEM_PROMPT: &str = "\
You are JARVIS, a voice-activated AI assistant. \
You are running on a self-hosted homelab. \
Keep your responses concise and conversational — they will be spoken aloud via TTS. \
Avoid markdown formatting, bullet points, and code blocks unless explicitly asked. \
Prefer short, direct answers.";

impl LlmClient {
    pub fn new(base_url: &str, model: &str) -> Self {
        Self {
            http: reqwest::Client::builder()
                // LLM inference can be slow — generous timeout.
                .timeout(std::time::Duration::from_secs(120))
                .build()
                .expect("failed to build HTTP client"),
            base_url: base_url.trim_end_matches('/').to_string(),
            model: model.to_string(),
        }
    }

    /// Send a chat completion request with streaming.
    ///
    /// `messages` should include the system prompt and conversation history.
    /// Tokens are accumulated as they stream in and the full response is returned.
    pub async fn chat(&self, messages: Vec<ChatMessage>) -> Result<LlmResponse> {
        let url = format!("{}/api/chat", self.base_url);

        let body = ChatRequest {
            model: self.model.clone(),
            messages,
            stream: true,
        };

        tracing::info!("LLM: sending request to {} (model={})", url, self.model);
        let start = std::time::Instant::now();

        let resp = self.http
            .post(&url)
            .json(&body)
            .send()
            .await
            .context("LLM request failed")?;

        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            anyhow::bail!("LLM returned {status}: {body}");
        }

        // Stream the NDJSON response line by line.
        let mut accumulated = String::new();
        let mut eval_count = None;
        let mut eval_duration = None;
        let mut first_token = true;

        let mut stream = resp.bytes_stream();
        let mut line_buffer = String::new();

        while let Some(chunk_result) = stream.next().await {
            let chunk_bytes = chunk_result.context("Error reading LLM stream")?;
            let chunk_str = String::from_utf8_lossy(&chunk_bytes);

            // Ollama sends one JSON object per line, but chunks may not
            // align to line boundaries. Buffer and split on newlines.
            line_buffer.push_str(&chunk_str);

            while let Some(newline_pos) = line_buffer.find('\n') {
                let line: String = line_buffer.drain(..=newline_pos).collect();
                let line = line.trim();
                if line.is_empty() {
                    continue;
                }

                match serde_json::from_str::<StreamChunk>(line) {
                    Ok(chunk) => {
                        if let Some(ref msg) = chunk.message {
                            if !msg.content.is_empty() {
                                if first_token {
                                    let ttft = start.elapsed();
                                    tracing::debug!(
                                        "LLM: first token in {:.2}s",
                                        ttft.as_secs_f64(),
                                    );
                                    first_token = false;
                                }
                                accumulated.push_str(&msg.content);
                            }
                        }

                        if chunk.done {
                            eval_count = chunk.eval_count;
                            eval_duration = chunk.eval_duration;
                        }
                    }
                    Err(e) => {
                        tracing::warn!("LLM: failed to parse stream chunk: {e}");
                    }
                }
            }
        }

        // Handle any remaining data in the line buffer (no trailing newline).
        let remaining = line_buffer.trim();
        if !remaining.is_empty() {
            if let Ok(chunk) = serde_json::from_str::<StreamChunk>(remaining) {
                if let Some(ref msg) = chunk.message {
                    accumulated.push_str(&msg.content);
                }
                if chunk.done {
                    eval_count = chunk.eval_count;
                    eval_duration = chunk.eval_duration;
                }
            }
        }

        let elapsed = start.elapsed();
        let response = LlmResponse {
            text: accumulated.trim().to_string(),
            eval_count,
            eval_duration_ns: eval_duration,
        };

        let tps_str = response
            .tokens_per_sec()
            .map(|t| format!("{t:.1} tok/s"))
            .unwrap_or_else(|| "n/a".into());

        tracing::info!(
            "LLM: response in {:.1}s ({} chars, {})",
            elapsed.as_secs_f64(),
            response.text.len(),
            tps_str,
        );
        tracing::debug!("LLM response text: \"{}\"", response.text);

        Ok(response)
    }

    /// Convenience: build a simple [system, user] message list and call chat().
    pub async fn ask(&self, user_text: &str) -> Result<LlmResponse> {
        let messages = vec![
            ChatMessage {
                role: "system".into(),
                content: DEFAULT_SYSTEM_PROMPT.into(),
            },
            ChatMessage {
                role: "user".into(),
                content: user_text.into(),
            },
        ];
        self.chat(messages).await
    }

    /// Health check — GET /api/tags (lists available models).
    pub async fn health(&self) -> Result<()> {
        let url = format!("{}/api/tags", self.base_url);
        let resp = self.http.get(&url).send().await.context("LLM health check failed")?;
        if resp.status().is_success() {
            Ok(())
        } else {
            anyhow::bail!("LLM health check returned {}", resp.status())
        }
    }
}
