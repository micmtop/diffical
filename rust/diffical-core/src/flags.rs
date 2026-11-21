//! Standard UTAU resampler flags: parsing and documented engine semantics.
//!
//! Names are case-sensitive, the first occurrence of a name wins, and values
//! are optional. A missing or non-finite value falls back to that flag's
//! default (never an error, never a panic). See `FLAGS.md` for the table and
//! `README.md` for the parity story.

use serde_json::Value;

/// Every flag this engine knows about. Other names are ignored (and reported
/// by `unknown` for `inspect`).
pub const STANDARD_FLAGS: &[&str] = &[
    "fe", "fl", "fo", "fv", "fp", "ve", "vo", "g", "t", "A", "B", "G", "P", "S", "p", "R", "D",
    "C", "Z", "Hv", "Hb", "Ht", "He", "HG",
];

/// Two-character names are matched before single characters when parsing a
/// flags string (`Hb200` is `Hb=200`, not `H` followed by junk).
const NAMES_BY_LEN: &[&str] = &[
    "Hv", "Hb", "Ht", "He", "HG", "fe", "fl", "fo", "fv", "fp", "ve", "vo", "g", "t", "A", "B",
    "G", "P", "S", "p", "R", "D", "C", "Z",
];

#[derive(Debug, Clone, Default)]
pub struct PhoneFlags {
    /// `(canonical name, value)` for the first occurrence of each name.
    /// `None` means the flag was present without a usable value.
    entries: Vec<(&'static str, Option<f64>)>,
    /// Names that are not in the standard set (parsed, ignored).
    pub unknown: Vec<String>,
}

/// HN-SEP parameters after the reference mappings.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HnsepParams {
    /// 0..100, `clip((Hb - 100) / 5, 0, 100)`; absent Hb -> 0.
    pub breath: f64,
    /// 0..100, `clip(Hv / 1.5, 0, 100)` for an explicit value; an absent or
    /// valueless Hv keeps the mix neutral at 100.
    pub voicing: f64,
    /// Raw tension in -100..100 (`Ht`); absent Ht -> 0.
    pub tension: f64,
}

fn canonical(name: &str) -> Option<&'static str> {
    STANDARD_FLAGS.iter().find(|flag| **flag == name).copied()
}

fn parse_number(value: Option<&Value>) -> Option<f64> {
    let number = match value? {
        Value::Number(n) => n.as_f64(),
        Value::String(s) => s.trim().parse::<f64>().ok(),
        _ => None,
    }?;
    number.is_finite().then_some(number)
}

impl PhoneFlags {
    pub fn from_values(values: &[Value]) -> Self {
        let mut out = Self::default();
        for value in values {
            let Some(items) = value.as_array() else { continue };
            let Some(raw_name) = items.first().and_then(Value::as_str) else {
                continue;
            };
            let Some(name) = canonical(raw_name) else {
                if !out.unknown.iter().any(|u| u == raw_name) {
                    out.unknown.push(raw_name.to_string());
                }
                continue;
            };
            if out.entries.iter().any(|(existing, _)| *existing == name) {
                continue; // first occurrence wins
            }
            out.entries.push((name, parse_number(items.get(1))));
        }
        out
    }

    pub fn present(&self, name: &str) -> bool {
        self.entries.iter().any(|(n, _)| *n == name)
    }

    fn value(&self, name: &str) -> Option<f64> {
        self.entries
            .iter()
            .find(|(n, _)| *n == name)
            .and_then(|(_, v)| *v)
    }

    /// Value clamped to `lo..hi`; absent/valueless/non-finite -> `default`.
    fn number(&self, name: &str, default: f64, lo: f64, hi: f64) -> f64 {
        self.value(name).map_or(default, |v| v.clamp(lo, hi))
    }

    /// `g`: formant shift in semitones, `clip(g, -100, 100) / 100`.
    ///
    /// g is accepted over -600..600; like the reference mapping the effective
    /// formant shift saturates at +/-1 semitone.
    pub fn gender_key_shift(&self) -> f64 {
        self.number("g", 0.0, -600.0, 600.0).clamp(-100.0, 100.0) / 100.0
    }

    /// `t`: pitch offset in cents, added to the phone's voiced F0 frames.
    pub fn pitch_cents(&self) -> f64 {
        self.number("t", 0.0, -1200.0, 1200.0)
    }

    /// `Hb`/`Hv`: `Some` when the HN-SEP separation changes the mix
    /// (breath > 0.5 or |voicing - 100| > 0.5), with the reference mapping.
    /// `Ht` alone does not need separation (the mix is identity); its effect is
    /// the tilt applied by the caller, so it does not load the HN-SEP model.
    pub fn hnsep(&self) -> Option<HnsepParams> {
        let breath = if self.present("Hb") {
            ((self.number("Hb", 100.0, 0.0, 500.0) - 100.0) / 5.0).clamp(0.0, 100.0)
        } else {
            0.0
        };
        let voicing = match self.value("Hv") {
            Some(v) => (v.clamp(0.0, 150.0) / 1.5).clamp(0.0, 100.0),
            // Absent or valueless Hv leaves the noise mix at unity, matching
            // the reference's per-phone default (explicit Hv=100 is 66.7).
            None => 100.0,
        };
        let tension = if self.present("Ht") {
            self.number("Ht", 0.0, -100.0, 100.0)
        } else {
            0.0
        };
        // The separation is only needed when the mix is not the identity.
        let trigger = breath > 0.5 || (voicing - 100.0).abs() > 0.5;
        trigger.then_some(HnsepParams {
            breath,
            voicing,
            tension,
        })
    }

    /// Raw `Ht` tension in -100..100 (0 when absent). The HN-SEP trigger uses
    /// the reference mapping; the tilt stage uses this raw value.
    pub fn tension(&self) -> f64 {
        if self.present("Ht") {
            self.number("Ht", 0.0, -100.0, 100.0)
        } else {
            0.0
        }
    }

    /// `HG`: growl amount 0..100.
    pub fn growl(&self) -> f64 {
        self.number("HG", 0.0, 0.0, 100.0)
    }

    /// `A`: pitch-following amplitude modulation, -100..100.
    pub fn amplitude_mod(&self) -> f64 {
        self.number("A", 0.0, -100.0, 100.0)
    }

    /// `He`: long-note mel loop (a switch, no value).
    pub fn long_note_loop(&self) -> bool {
        self.present("He")
    }

    /// `G`: force cache regeneration. Parsed and documented as a no-op.
    pub fn force_cache(&self) -> bool {
        self.present("G")
    }

    /// `P`: loudness normalisation. Parsed and documented as a no-op.
    pub fn loudness_norm(&self) -> bool {
        self.present("P")
    }

    /// Names of the effects this phone actually activates (for `inspect`).
    pub fn active(&self) -> Vec<&'static str> {
        let mut out = Vec::new();
        if self.gender_key_shift() != 0.0 {
            out.push("g");
        }
        if self.pitch_cents() != 0.0 {
            out.push("t");
        }
        if self.hnsep().is_some() {
            out.push("Hb/Hv/Ht");
        }
        if self.growl() > 0.0 {
            out.push("HG");
        }
        if self.amplitude_mod() != 0.0 {
            out.push("A");
        }
        if self.long_note_loop() {
            out.push("He");
        }
        out
    }
}

/// One flag parsed from a flags *string* (the `--inject-flags` form).
#[derive(Debug, Clone, PartialEq)]
pub struct FlagEntry {
    pub name: &'static str,
    pub value: Option<f64>,
}

/// Parse `"Hb200Hv120Ht30g100Dt-50"` the way the request format encodes flags:
/// case-sensitive names, optional signed numeric value, first occurrence wins.
/// Unrecognised characters are skipped, matching the reference's tolerant scan.
pub fn parse_flag_string(text: &str) -> Vec<FlagEntry> {
    let bytes = text.as_bytes();
    let mut out: Vec<FlagEntry> = Vec::new();
    let mut i = 0;
    while i < bytes.len() {
        let rest = &text[i..];
        let Some(name) = NAMES_BY_LEN.iter().find(|n| rest.starts_with(**n)) else {
            // Skip one UTF-8 character (flags strings are ASCII in practice).
            i += rest.chars().next().map_or(1, char::len_utf8);
            continue;
        };
        i += name.len();
        let mut j = i;
        if j < bytes.len() && (bytes[j] == b'+' || bytes[j] == b'-') {
            j += 1;
        }
        let digits_start = j;
        while j < bytes.len() && bytes[j].is_ascii_digit() {
            j += 1;
        }
        let value = if j > digits_start {
            text[i..j].parse::<f64>().ok()
        } else {
            None
        };
        if let Some(v) = value {
            if !v.is_finite() {
                i = j;
                continue;
            }
        }
        i = j;
        let canonical = canonical(name).expect("known name");
        if out.iter().any(|e| e.name == canonical) {
            continue; // first occurrence wins
        }
        out.push(FlagEntry {
            name: canonical,
            value,
        });
    }
    out
}

/// Replace every phone's flags with the injected list (test helper).
pub fn inject_into_request(req: &mut crate::protocol::RenderRequest, entries: &[FlagEntry]) {
    let values: Vec<Value> = entries
        .iter()
        .map(|entry| {
            let mut item = vec![Value::String(entry.name.to_string())];
            item.push(match entry.value {
                Some(v) => Value::from(v),
                None => Value::Null,
            });
            item.push(Value::String(entry.name.to_string()));
            Value::Array(item)
        })
        .collect();
    for phone in &mut req.phones {
        phone.flags = values.clone();
    }
}

#[cfg(test)]
mod tests {
    use super::{inject_into_request, parse_flag_string, PhoneFlags, FlagEntry};
    use serde_json::json;

    fn flags(items: &[(&str, Option<f64>)]) -> PhoneFlags {
        let values: Vec<serde_json::Value> = items
            .iter()
            .map(|(name, value)| match value {
                Some(v) => json!([name, v, "test"]),
                None => json!([name, null, "test"]),
            })
            .collect();
        PhoneFlags::from_values(&values)
    }

    #[test]
    fn first_occurrence_wins() {
        let f = flags(&[("g", Some(100.0)), ("g", Some(-600.0))]);
        assert_eq!(f.gender_key_shift(), 1.0);
    }

    #[test]
    fn names_are_case_sensitive() {
        let f = PhoneFlags::from_values(&[json!(["Hb", 200, "x"]), json!(["hb", 500, "x"])]);
        assert_eq!(f.hnsep().unwrap().breath, 20.0);
        assert_eq!(f.unknown, vec!["hb".to_string()]);
    }

    #[test]
    fn non_finite_values_fall_back_to_defaults() {
        for raw in ["NaN", "inf", "-inf"] {
            let f = PhoneFlags::from_values(&[json!(["Hb", raw, "x"])]);
            assert_eq!(f.hnsep(), None, "{raw}");
        }
        let f = PhoneFlags::from_values(&[json!(["Hb", f64::NAN, "x"])]);
        assert_eq!(f.hnsep(), None);
    }

    #[test]
    fn hnsep_trigger_matches_reference() {
        // defaults / absent -> no separation
        assert_eq!(flags(&[]).hnsep(), None);
        assert_eq!(flags(&[("Hb", Some(100.0))]).hnsep(), None);
        assert_eq!(flags(&[("Hv", Some(150.0))]).hnsep(), None);
        assert_eq!(flags(&[("Ht", Some(-100.0))]).hnsep(), None);
        // Ht alone leaves the mix at identity; it only needs the tilt.
        assert_eq!(flags(&[("Ht", Some(30.0))]).hnsep(), None);
        assert_eq!(flags(&[("Ht", Some(30.0))]).tension(), 30.0);
        // reference mappings
        let p = flags(&[("Hb", Some(200.0)), ("Hv", Some(120.0)), ("Ht", Some(30.0))])
            .hnsep()
            .unwrap();
        assert!((p.breath - 20.0).abs() < 1e-9);
        assert!((p.voicing - 80.0).abs() < 1e-9);
        assert!((p.tension - 30.0).abs() < 1e-9);
    }

    #[test]
    fn ranges_are_clamped() {
        assert_eq!(flags(&[("Hb", Some(9999.0))]).hnsep().unwrap().breath, 80.0);
        // Hb <= 100 maps to zero breath and is neutral on its own.
        assert!(flags(&[("Hb", Some(-5.0))]).hnsep().is_none());
        assert_eq!(
            flags(&[("Hb", Some(-5.0)), ("Hv", Some(120.0))])
                .hnsep()
                .unwrap()
                .breath,
            0.0
        );
        assert_eq!(flags(&[("Hv", Some(-3.0))]).hnsep().unwrap().voicing, 0.0);
        // Hv = 150 maps to voicing 100 (the neutral value), so it is not a
        // trigger by itself; an explicit Hv = 100 maps to 66.7 per the
        // reference formula and does trigger. A valueless Hv stays neutral.
        assert!(flags(&[("Hv", Some(150.0))]).hnsep().is_none());
        assert!((flags(&[("Hv", Some(100.0))]).hnsep().unwrap().voicing - 66.666_666).abs() < 1e-5);
        assert!(flags(&[("Hv", None)]).hnsep().is_none());
        assert_eq!(
            flags(&[("Hv", Some(150.0)), ("Hb", Some(200.0))])
                .hnsep()
                .unwrap()
                .voicing,
            100.0
        );
        assert_eq!(flags(&[("g", Some(600.0))]).gender_key_shift(), 1.0);
        assert_eq!(flags(&[("t", Some(5000.0))]).pitch_cents(), 1200.0);
        assert_eq!(flags(&[("HG", Some(-9.0))]).growl(), 0.0);
        assert_eq!(flags(&[("A", Some(500.0))]).amplitude_mod(), 100.0);
    }

    #[test]
    fn switches_ignore_values() {
        let f = flags(&[("He", None), ("G", None), ("P", Some(86.0))]);
        assert!(f.long_note_loop());
        assert!(f.force_cache());
        assert!(f.loudness_norm());
        assert!(f.active().contains(&"He"));
    }

    #[test]
    fn flag_string_parses_like_request_tuples() {
        let parsed = parse_flag_string("Hb200Hv120Ht30g100Dt-50He");
        assert_eq!(
            parsed,
            vec![
                FlagEntry { name: "Hb", value: Some(200.0) },
                FlagEntry { name: "Hv", value: Some(120.0) },
                FlagEntry { name: "Ht", value: Some(30.0) },
                FlagEntry { name: "g", value: Some(100.0) },
                FlagEntry { name: "D", value: None },
                FlagEntry { name: "t", value: Some(-50.0) },
                FlagEntry { name: "He", value: None },
            ]
        );
        // first occurrence wins
        assert_eq!(parse_flag_string("g10g-20"), vec![FlagEntry { name: "g", value: Some(10.0) }]);
        // unknown letters are skipped
        assert_eq!(parse_flag_string("xq100"), vec![]);
    }

    #[test]
    fn inject_overwrites_every_phone() {
        let mut req: crate::protocol::RenderRequest =
            crate::protocol::parse_request(r#"{"durationMs": 10, "phones": [{"volume": 1}, {}]}"#)
                .unwrap();
        let entries = parse_flag_string("Hb200He");
        inject_into_request(&mut req, &entries);
        assert_eq!(req.phones[0].flags.len(), 2);
        assert_eq!(req.phones[1].flags.len(), 2);
        let f = PhoneFlags::from_values(&req.phones[1].flags);
        assert!(f.long_note_loop());
        assert_eq!(f.hnsep().unwrap().breath, 20.0);
    }
}
