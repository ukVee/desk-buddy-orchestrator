use clap::Parser;
use serde::Deserialize;

/// Runtime configuration for the JARVIS orchestrator.
///
/// Hierarchy: CLI args > env vars > config file > defaults.
#[derive(Debug, Clone)]
pub struct Config {
    pub audio: AudioConfig,
    pub wake_word: WakeWordConfig,
    pub vad: VadConfig,
    pub services: ServiceConfig,
    pub log_level: String,
}

#[derive(Debug, Clone)]
pub struct AudioConfig {
    /// Substring match against ALSA device name for capture.
    /// From Phase 0: Blue mic is card 0.
    pub capture_device: String,

    /// Substring match against ALSA device name for playback.
    /// From Phase 0: PCH is card 1, use plughw:1,0.
    pub playback_device: String,

    /// Sample rate we normalize everything to internally.
    /// Whisper and rustpotter both want 16kHz.
    pub target_sample_rate: u32,
}

#[derive(Debug, Clone)]
pub struct WakeWordConfig {
    /// Path to the .rpw model file.
    pub model_path: String,

    /// Detection threshold (0.0 - 1.0). Higher = fewer false positives.
    pub threshold: f32,

    /// If true, skip wake word detection and trigger on any speech.
    /// Useful for testing without a trained model.
    pub bypass: bool,
}

#[derive(Debug, Clone)]
pub struct VadConfig {
    /// Silence duration (ms) before end-of-speech is declared.
    pub silence_timeout_ms: u64,

    /// webrtc-vad aggressiveness mode (0-3). Higher = more aggressive
    /// at filtering non-speech. 2-3 recommended for noisy environments.
    pub aggressiveness: i32,

    /// Frame duration in ms. webrtc-vad supports 10, 20, or 30ms.
    pub frame_duration_ms: usize,
}

#[derive(Debug, Clone)]
pub struct ServiceConfig {
    /// Whisper STT endpoint (k3s ClusterIP).
    /// e.g. "http://whisper-stt.jarvis.svc.cluster.local:8080"
    pub whisper_url: String,

    /// Piper TTS endpoint (k3s ClusterIP).
    /// e.g. "http://piper-tts.jarvis.svc.cluster.local:8081"
    pub piper_url: String,

    /// Ollama endpoint (direct LAN to laptop).
    /// e.g. "http://192.168.0.XXX:11434"
    pub ollama_url: String,

    /// Ollama model name.
    /// e.g. "mistral:latest"
    pub ollama_model: String,
}

/// CLI arguments — thin layer that feeds into Config.
#[derive(Parser, Debug)]
#[command(name = "jarvis-orchestrator", about = "JARVIS voice assistant orchestrator")]
pub struct CliArgs {
    /// Capture device name substring (e.g. "Blue")
    #[arg(long, env = "CAPTURE_DEVICE", default_value = "Blue")]
    pub capture_device: String,

    /// Playback device name substring (e.g. "PCH")
    #[arg(long, env = "PLAYBACK_DEVICE", default_value = "PCH")]
    pub playback_device: String,

    /// Wake word model path
    #[arg(long, env = "WAKEWORD_MODEL", default_value = "/data/wakeword/hey_jarvis.rpw")]
    pub wakeword_model: String,

    /// Wake word detection threshold (0.0-1.0)
    #[arg(long, env = "WAKEWORD_THRESHOLD", default_value = "0.5")]
    pub wakeword_threshold: f32,

    /// Bypass wake word detection (trigger on any speech)
    #[arg(long, env = "WAKEWORD_BYPASS", default_value = "false")]
    pub wakeword_bypass: bool,

    /// VAD silence timeout in milliseconds
    #[arg(long, env = "VAD_SILENCE_MS", default_value = "800")]
    pub vad_silence_ms: u64,

    /// VAD aggressiveness (0-3)
    #[arg(long, env = "VAD_AGGRESSIVENESS", default_value = "2")]
    pub vad_aggressiveness: i32,

    /// Whisper STT service URL
    #[arg(long, env = "WHISPER_URL", default_value = "http://whisper-stt.jarvis.svc.cluster.local:8080")]
    pub whisper_url: String,

    /// Piper TTS service URL
    #[arg(long, env = "PIPER_URL", default_value = "http://piper-tts.jarvis.svc.cluster.local:8081")]
    pub piper_url: String,

    /// Ollama LLM service URL
    #[arg(long, env = "OLLAMA_URL", default_value = "http://192.168.0.72:11434")]
    pub ollama_url: String,

    /// Ollama model name
    #[arg(long, env = "OLLAMA_MODEL", default_value = "mistral:latest")]
    pub ollama_model: String,

    /// Log level (trace, debug, info, warn, error)
    #[arg(long, env = "RUST_LOG", default_value = "jarvis=info")]
    pub log_level: String,
}

impl Config {
    pub fn from_cli(args: CliArgs) -> Self {
        Config {
            audio: AudioConfig {
                capture_device: args.capture_device,
                playback_device: args.playback_device,
                target_sample_rate: 16000,
            },
            wake_word: WakeWordConfig {
                model_path: args.wakeword_model,
                threshold: args.wakeword_threshold,
                bypass: args.wakeword_bypass,
            },
            vad: VadConfig {
                silence_timeout_ms: args.vad_silence_ms,
                aggressiveness: args.vad_aggressiveness,
                frame_duration_ms: 30, // 30ms frames — best balance for webrtc-vad
            },
            services: ServiceConfig {
                whisper_url: args.whisper_url,
                piper_url: args.piper_url,
                ollama_url: args.ollama_url,
                ollama_model: args.ollama_model,
            },
            log_level: args.log_level,
        }
    }
}
