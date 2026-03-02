use anyhow::{Context, Result, bail};
use cpal::traits::{DeviceTrait, HostTrait};

/// Find a cpal input device whose name contains the given substring.
pub fn find_input_device(name_substr: &str) -> Result<cpal::Device> {
    let host = cpal::default_host();

    let devices: Vec<_> = host
        .input_devices()
        .context("Failed to enumerate input devices")?
        .collect();

    tracing::info!("Available input devices:");
    for d in &devices {
        let name = d.name().unwrap_or_else(|_| "<unknown>".into());
        tracing::info!("  - {name}");
    }

    devices
        .into_iter()
        .find(|d| {
            d.name()
                .map(|n| n.contains(name_substr))
                .unwrap_or(false)
        })
        .with_context(|| format!("No input device matching '{name_substr}'"))
}

/// Find a cpal output device whose name contains the given substring.
pub fn find_output_device(name_substr: &str) -> Result<cpal::Device> {
    let host = cpal::default_host();

    let devices: Vec<_> = host
        .output_devices()
        .context("Failed to enumerate output devices")?
        .collect();

    tracing::info!("Available output devices:");
    for d in &devices {
        let name = d.name().unwrap_or_else(|_| "<unknown>".into());
        tracing::info!("  - {name}");
    }

    devices
        .into_iter()
        .find(|d| {
            d.name()
                .map(|n| n.contains(name_substr))
                .unwrap_or(false)
        })
        .with_context(|| format!("No output device matching '{name_substr}'"))
}

/// Query and log the supported input configs for a device.
/// Returns the default config (what we'll use to open the stream).
pub fn log_input_configs(device: &cpal::Device) -> Result<cpal::SupportedStreamConfig> {
    let name = device.name().unwrap_or_else(|_| "<unknown>".into());

    if let Ok(configs) = device.supported_input_configs() {
        tracing::debug!("Supported input configs for '{name}':");
        for cfg in configs {
            tracing::debug!(
                "  channels={}, rate={}..{}, format={:?}",
                cfg.channels(),
                cfg.min_sample_rate().0,
                cfg.max_sample_rate().0,
                cfg.sample_format(),
            );
        }
    }

    let config = device
        .default_input_config()
        .context("No default input config")?;

    tracing::info!(
        "Using input config for '{name}': channels={}, rate={}, format={:?}",
        config.channels(),
        config.sample_rate().0,
        config.sample_format(),
    );

    Ok(config)
}
