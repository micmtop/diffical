//! `inspect` command: print the parsed request, phones and notes.

use anyhow::{Context, Result};
use diffical_core::config::AudioConfig;
use diffical_core::flags::PhoneFlags;
use diffical_core::protocol::parse_request;
use std::path::Path;

fn read_text(path: &Path) -> Result<String> {
    std::fs::read_to_string(path).with_context(|| format!("read {path:?}"))
}

pub fn inspect(path: &str) -> Result<()> {
    let req = parse_request(&read_text(Path::new(path))?)?;
    let cfg = AudioConfig::default();
    let total_ms = req.duration_ms + req.leading_ms;
    let total_frames = (total_ms / cfg.frame_ms()).ceil().max(1.0) as usize;
    let mode = if req.pitches.is_empty() {
        "missing"
    } else if req.pitches.len() as f64 >= total_frames as f64 * 0.9
        && req.pitches.len() as f64 <= total_frames as f64 * 1.1
    {
        "per-frame"
    } else {
        "5-tick"
    };
    println!(
        "parsed OK: {} phones, {} notes, {:.3} ms total ({} frames @ {:.3} ms)",
        req.phones.len(),
        req.notes.len(),
        total_ms,
        total_frames,
        cfg.frame_ms()
    );
    println!(
        "pitch curve: {} points ({mode}); dynamics={} gender={} breathiness={} tone_shift={}",
        req.pitches.len(),
        req.dynamics.len(),
        req.gender.len(),
        req.breathiness.len(),
        req.tone_shift.len()
    );
    let curve_names: Vec<&str> = req.curves.iter().map(|c| c.0.as_str()).collect();
    println!(
        "curves ({}): {}",
        curve_names.len(),
        if curve_names.is_empty() {
            "none".to_string()
        } else {
            curve_names.join(", ")
        }
    );
    println!("phones:");
    for (i, phone) in req.phones.iter().enumerate() {
        let flags = PhoneFlags::from_values(&phone.flags);
        let active = flags.active();
        let mut notes = Vec::new();
        if !active.is_empty() {
            notes.push(format!("active=[{}]", active.join(" ")));
        }
        if !flags.unknown.is_empty() {
            notes.push(format!("ignored=[{}]", flags.unknown.join(" ")));
        }
        println!(
            "  {i:>2} {:>6} tone={:<3} start={:>9.3} dur={:>7.3} end={:>9.3} preutter={:>8.3} overlap={:>8.3} vel={:.2} env_pts={} flags={} {}",
            format!("{:?}", phone.phoneme),
            phone.tone,
            phone.position_ms,
            phone.duration_ms,
            phone.end_ms,
            phone.preutter_ms,
            phone.overlap_ms,
            phone.velocity,
            phone.envelope.len(),
            phone.flags.len(),
            notes.join(" ")
        );
    }
    println!("notes:");
    for (i, note) in req.notes.iter().enumerate() {
        println!(
            "  {i:>2} tone={:<3} pos={:>9.3} dur={:>7.3} end={:>9.3} lyric={:?}",
            note.tone, note.position_ms, note.duration_ms, note.end_ms, note.lyric
        );
    }
    Ok(())
}
