//! Render request schema (`backend/protocol.py`) plus the camelCase key
//! normalisation done by `backend/server.py::_parse_request`.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, Deserialize, Default)]
pub struct RenderNote {
    #[serde(default)]
    pub lyric: String,
    #[serde(default)]
    pub tone: i64,
    #[serde(default)]
    pub position: i64,
    #[serde(default)]
    pub duration: i64,
    #[serde(default)]
    pub end: i64,
    #[serde(default)]
    pub position_ms: f64,
    #[serde(default)]
    pub duration_ms: f64,
    #[serde(default)]
    pub end_ms: f64,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum EnvelopePoint {
    Xy { x: f64, y: f64 },
    Pair(f64, f64),
}

impl EnvelopePoint {
    pub fn x(&self) -> f64 {
        match self {
            EnvelopePoint::Xy { x, .. } => *x,
            EnvelopePoint::Pair(x, _) => *x,
        }
    }

    pub fn y(&self) -> f64 {
        match self {
            EnvelopePoint::Xy { y, .. } => *y,
            EnvelopePoint::Pair(_, y) => *y,
        }
    }
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct RenderPhone {
    #[serde(default)]
    pub position: i64,
    #[serde(default)]
    pub duration: i64,
    #[serde(default)]
    pub end: i64,
    #[serde(default)]
    pub leading: i64,
    #[serde(default)]
    pub position_ms: f64,
    #[serde(default)]
    pub duration_ms: f64,
    #[serde(default)]
    pub end_ms: f64,
    #[serde(default)]
    pub leading_ms: f64,
    #[serde(default)]
    pub phoneme: String,
    #[serde(default)]
    pub tone: i64,
    #[serde(default)]
    pub note_index: i64,
    #[serde(default)]
    pub tempo: f64,
    #[serde(default)]
    pub preutter_ms: f64,
    #[serde(default)]
    pub overlap_ms: f64,
    #[serde(default)]
    pub suffix: String,
    #[serde(default)]
    pub oto_file: String,
    #[serde(default)]
    pub oto_offset: f64,
    #[serde(default)]
    pub oto_consonant: f64,
    #[serde(default)]
    pub oto_cutoff: f64,
    #[serde(default)]
    pub oto_preutter: f64,
    #[serde(default)]
    pub oto_overlap: f64,
    #[serde(default = "one")]
    pub volume: f64,
    #[serde(default = "one")]
    pub velocity: f64,
    #[serde(default)]
    pub modulation: f64,
    /// Flags are kept for inspection but not applied yet (HN-SEP stage).
    #[serde(default)]
    pub flags: Vec<Value>,
    #[serde(default)]
    pub direct: bool,
    #[serde(default)]
    pub envelope: Vec<EnvelopePoint>,
    #[serde(default)]
    pub phoneme_features: Option<Value>,
    #[serde(default)]
    pub cross_lang_source: Option<String>,
    #[serde(default)]
    pub cross_lang_target: Option<String>,
}

fn one() -> f64 {
    1.0
}

pub type Curve = (String, Vec<f64>);

#[derive(Debug, Clone, Deserialize)]
pub struct RenderRequest {
    #[serde(default)]
    pub position: i64,
    #[serde(default)]
    pub duration: i64,
    #[serde(default)]
    pub end: i64,
    #[serde(default)]
    pub leading: i64,
    #[serde(default)]
    pub position_ms: f64,
    #[serde(default)]
    pub duration_ms: f64,
    #[serde(default)]
    pub end_ms: f64,
    #[serde(default)]
    pub leading_ms: f64,
    #[serde(default)]
    pub notes: Vec<RenderNote>,
    #[serde(default)]
    pub phones: Vec<RenderPhone>,
    #[serde(default)]
    pub pitches: Vec<f64>,
    #[serde(default)]
    pub dynamics: Vec<f64>,
    #[serde(default)]
    pub gender: Vec<f64>,
    #[serde(default)]
    pub breathiness: Vec<f64>,
    #[serde(default)]
    pub tone_shift: Vec<f64>,
    #[serde(default)]
    pub tension: Vec<f64>,
    #[serde(default)]
    pub voicing: Vec<f64>,
    #[serde(default)]
    pub growl: Vec<f64>,
    #[serde(default)]
    pub lowcut: Vec<f64>,
    #[serde(default)]
    pub brel: Vec<f64>,
    #[serde(default)]
    pub breh: Vec<f64>,
    #[serde(default)]
    pub warm: Vec<f64>,
    #[serde(default)]
    pub hcmp: Vec<f64>,
    #[serde(default)]
    pub curves: Vec<Curve>,
    #[serde(default)]
    pub singer_path: String,
    #[serde(default)]
    pub singer_type: String,
    #[serde(default)]
    pub singer_language: String,
    #[serde(default)]
    pub language: String,
    #[serde(default)]
    pub hash: i64,
    #[serde(default)]
    pub is_pre_render: bool,
    #[serde(default)]
    pub track_no: i64,
}

impl RenderRequest {
    /// Bounds `backend/renderer.py::_clamp_request` applies before rendering.
    pub fn clamp(&mut self) {
        if !self.duration_ms.is_finite() {
            self.duration_ms = 0.0;
        }
        if !self.leading_ms.is_finite() {
            self.leading_ms = 0.0;
        }
        if self.duration_ms < 0.0 {
            self.duration_ms = 0.0;
        }
        if self.leading_ms < 0.0 {
            self.leading_ms = 0.0;
        }
        let leading_max = 5000.0_f64.max(self.duration_ms * 0.5);
        if self.leading_ms > leading_max {
            self.leading_ms = leading_max;
        }
    }

    pub fn phone_volume_reference(&self) -> f64 {
        let mut positive: Vec<f64> = self
            .phones
            .iter()
            .map(|p| p.volume)
            .filter(|v| v.is_finite() && *v > 0.0)
            .collect();
        if positive.is_empty() {
            return 0.0;
        }
        positive.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let mid = positive.len() / 2;
        if positive.len() % 2 == 0 {
            (positive[mid - 1] + positive[mid]) / 2.0
        } else {
            positive[mid]
        }
    }
}

/// Parse JSON emitted by the C# client. Mirrors `_convert_keys` + `_parse_request`:
/// recursively camelCase -> snake_case, then deserialize.
pub fn parse_request(text: &str) -> Result<RenderRequest> {
    let value: Value = serde_json::from_str(text).context("parse render request JSON")?;
    let mut request: RenderRequest =
        serde_json::from_value(snakeize_keys(value)).context("deserialize render request")?;
    request.clamp();
    anyhow::ensure!(
        request.duration_ms.is_finite() && request.leading_ms.is_finite(),
        "phrase timing must be finite"
    );
    let total_ms = request.duration_ms + request.leading_ms;
    anyhow::ensure!(
        total_ms >= 0.0 && total_ms < 600_000.0,
        "implausible phrase duration {total_ms} ms (limit 600000 ms)"
    );
    Ok(request)
}

fn snakeize_keys(value: Value) -> Value {
    match value {
        Value::Object(map) => Value::Object(
            map.into_iter()
                .map(|(k, v)| (camel_to_snake(&k), snakeize_keys(v)))
                .collect(),
        ),
        Value::Array(items) => Value::Array(items.into_iter().map(snakeize_keys).collect()),
        other => other,
    }
}

fn camel_to_snake(name: &str) -> String {
    let chars: Vec<char> = name.chars().collect();
    let mut out = String::with_capacity(name.len() + 4);
    for (i, &c) in chars.iter().enumerate() {
        if i > 0 && c.is_ascii_uppercase() {
            let next_lower = chars.get(i + 1).is_some_and(|n| n.is_ascii_lowercase());
            let prev = chars[i - 1];
            let prev_is_word = prev.is_ascii_lowercase() || prev.is_ascii_digit();
            if next_lower || prev_is_word {
                out.push('_');
            }
        }
        out.push(c.to_ascii_lowercase());
    }
    out
}

/// Render response body (`backend/protocol.py::RenderResultResponse`).
#[derive(Debug, Clone, Serialize)]
pub struct RenderResultResponse {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    pub leading_ms: f64,
    pub position_ms: f64,
    pub estimated_length_ms: f64,
    pub pitch_ticks: Option<Vec<f32>>,
    pub pitch_tones: Option<Vec<f32>>,
    pub real_curves: Option<Vec<Curve>>,
}

#[derive(Debug, Clone, Serialize)]
pub struct HealthResponse {
    pub status: &'static str,
    pub version: &'static str,
    pub model_sha: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn camel_keys_match_python() {
        for (input, want) in [
            ("positionMs", "position_ms"),
            ("isPreRender", "is_pre_render"),
            ("toneShift", "tone_shift"),
            ("preutter_ms", "preutter_ms"),
            ("userID", "user_id"),
            ("HTTPServer", "http_server"),
        ] {
            assert_eq!(camel_to_snake(input), want, "{input}");
        }
    }

    #[test]
    fn parses_test_request() {
        let text = std::fs::read_to_string("/tmp/diffical_requests/req_1f87cfc723804b5b.json");
        let Ok(text) = text else { return };
        let req = parse_request(&text).expect("request");
        assert_eq!(req.phones.len(), 10);
        assert_eq!(req.notes.len(), 10);
        assert!((req.leading_ms - 19.0).abs() < 1e-9);
        assert_eq!(req.phones[0].phoneme, "- と");
        assert!(!req.phones[0].envelope.is_empty());
        assert!((req.phones[0].envelope[4].x() - 88.23529).abs() < 1e-4);
    }

    #[test]
    fn clamps_negative_timing() {
        let json = r#"{"durationMs": -5.0, "leadingMs": -1.0}"#;
        let req = parse_request(json).expect("parse");
        assert_eq!(req.duration_ms, 0.0);
        assert_eq!(req.leading_ms, 0.0);
    }

    #[test]
    fn rejects_bad_json() {
        let err = parse_request("{ not json").unwrap_err();
        assert!(err.to_string().contains("JSON") || err.to_string().contains("key"));
    }

    #[test]
    fn tolerant_defaults_for_unknown_phone() {
        let json = r#"{"durationMs": 1.0, "leadingMs": 0.0, "phones": [{"volume": 1}]}"#;
        let req = parse_request(json).expect("tolerant defaults");
        assert_eq!(req.phones.len(), 1);
        assert_eq!(req.phones[0].volume, 1.0);
    }
}
