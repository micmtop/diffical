use anyhow::{anyhow, Context, Result};
use axum::{
    extract::State,
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use diffical_core::protocol::{parse_request, HealthResponse, RenderResultResponse};
use diffical_core::RenderEngine;
use sha2::{Digest, Sha256};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use tokio::net::TcpListener;

#[derive(Clone)]
struct AppState {
    engine: Arc<Mutex<RenderEngine>>,
    model_sha: Arc<str>,
}

const VERSION: &str = env!("CARGO_PKG_VERSION");

pub fn run(
    host: &str,
    port: u16,
    assembler: &Path,
    vocoder: &Path,
    hnsep: Option<PathBuf>,
) -> Result<()> {
    let model_sha = model_sha(&[assembler, vocoder])?;
    let engine = RenderEngine::with_hnsep(assembler, vocoder, hnsep.clone())?;
    println!(
        "models loaded: assembler={assembler:?} vocoder={vocoder:?} hnsep={hnsep:?} sha256={model_sha}"
    );
    let state = AppState {
        engine: Arc::new(Mutex::new(engine)),
        model_sha: Arc::from(model_sha),
    };
    let app = Router::new()
        .route("/", get(health))
        .route("/health", get(health))
        .route("/render", post(render))
        .with_state(state);

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .context("build tokio runtime")?;
    runtime.block_on(async move {
        let addr = format!("{host}:{port}");
        let listener = TcpListener::bind(&addr)
            .await
            .with_context(|| format!("bind {addr}"))?;
        println!("listening on http://{addr}");
        axum::serve(listener, app)
            .with_graceful_shutdown(shutdown_signal())
            .await
            .context("axum server")?;
        Ok(())
    })
}

async fn health(State(state): State<AppState>) -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok",
        version: VERSION,
        model_sha: state.model_sha.to_string(),
    })
}

async fn render(State(state): State<AppState>, body: String) -> impl IntoResponse {
    let req = match parse_request(&body) {
        Ok(req) => req,
        Err(err) => {
            let dump = std::env::temp_dir().join("diffical_bad_request.json");
            let _ = std::fs::write(&dump, &body);
            eprintln!("render 400: {err:#}; body saved to {:?}", dump);
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({"error": format!("{err:#}"), "type": "request"})),
            )
                .into_response()
        }
    };
    let flagged = req.phones.iter().filter(|p| !p.flags.is_empty()).count();
    eprintln!(
        "render: phones={} flagged={} duration_ms={:.1}",
        req.phones.len(), flagged, req.duration_ms
    );
    for (i, p) in req.phones.iter().enumerate().filter(|(_, p)| !p.flags.is_empty()).take(3) {
        eprintln!("  flags[{i}]={}", serde_json::to_string(&p.flags).unwrap_or_default());
    }
    let leading_ms = req.leading_ms;
    let position_ms = req.position_ms;
    let estimated_length_ms = req.duration_ms + req.leading_ms;
    let engine = Arc::clone(&state.engine);
    let result = tokio::task::spawn_blocking(move || -> Result<Vec<f32>> {
        let mut engine = engine
            .lock()
            .map_err(|_| anyhow!("render engine mutex poisoned"))?;
        Ok(engine.render(&req, false)?.samples)
    })
    .await;
    match result {
        Ok(Ok(samples)) => (
            StatusCode::OK,
            Json(RenderResultResponse {
                samples,
                sample_rate: diffical_core::AudioConfig::default().sample_rate,
                leading_ms,
                position_ms,
                estimated_length_ms,
                pitch_ticks: None,
                pitch_tones: None,
                real_curves: None,
            }),
        )
            .into_response(),
        Ok(Err(err)) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": format!("{err:#}"), "type": "render"})),
        )
            .into_response(),
        Err(err) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": format!("{err:#}"), "type": "join"})),
        )
            .into_response(),
    }
}

async fn shutdown_signal() {
    if tokio::signal::ctrl_c().await.is_ok() {
        println!("shutdown signal received");
    }
}

fn model_sha(paths: &[&Path]) -> Result<String> {
    let mut hasher = Sha256::new();
    for path in paths {
        let mut file =
            std::fs::File::open(path).with_context(|| format!("open {path:?} for sha256"))?;
        let mut buf = [0u8; 64 * 1024];
        loop {
            let n = file.read(&mut buf)?;
            if n == 0 {
                break;
            }
            hasher.update(&buf[..n]);
        }
    }
    Ok(format!("{:x}", hasher.finalize()))
}
