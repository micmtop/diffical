use anyhow::{Context, Result};
use ort::environment::{init as env_init, Environment};
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::Tensor;
use std::path::Path;

pub struct TensorData {
    pub shape: Vec<i64>,
    pub data: Vec<f32>,
}

pub struct IntTensorData {
    pub shape: Vec<i64>,
    pub data: Vec<i64>,
}

impl IntTensorData {
    fn into_tensor(self) -> Result<Tensor<i64>> {
        Tensor::from_array((self.shape, self.data)).map_err(Into::into)
    }
}

impl TensorData {
    fn into_tensor(self) -> Result<Tensor<f32>> {
        Tensor::from_array((self.shape, self.data)).map_err(Into::into)
    }
}

pub struct PointerOnnx {
    _env: Environment,
    session: Session,
}

impl PointerOnnx {
    pub fn load(path: &Path) -> Result<Self> {
        let env = env_init().with_execution_providers(crate::ep::providers()).build().context("create ort environment")?;
        Self::load_with_env(&env, path)
    }

    /// Reuse a process-wide ONNX Runtime environment. ort allows exactly one
    /// per process, and the render pipeline needs two sessions (assembler +
    /// vocoder), so the engine creates it once.
    pub fn load_with_env(env: &Environment, path: &Path) -> Result<Self> {
        let session = Session::builder(env)?
            .with_execution_providers(crate::ep::providers())
            .map_err(|e| anyhow::anyhow!(e.to_string()))?
            .with_auto_device(crate::ep::auto_policy())
            .map_err(|e| anyhow::anyhow!(e.to_string()))?
            .with_execution_providers(crate::ep::providers())
            .unwrap_or_else(|e| e.recover())
            .with_optimization_level(GraphOptimizationLevel::All)
            .unwrap_or_else(|e| e.recover())
            .with_intra_threads(4)
            .unwrap_or_else(|e| e.recover())
            .with_inter_threads(2)
            .unwrap_or_else(|e| e.recover())
            .commit_from_file(path)
            .with_context(|| format!("load {path:?}"))?;
        Ok(Self {
            _env: env.clone(),
            session,
        })
    }

    pub fn run(
        &mut self,
        segs: TensorData,
        seg_len: IntTensorData,
        seg_idx: IntTensorData,
        progress: TensorData,
        boundary: TensorData,
        f0: TensorData,
    ) -> Result<TensorData> {
        let outputs = self.session.run(ort::inputs![
            "segs" => segs.into_tensor()?,
            "seg_len" => seg_len.into_tensor()?,
            "seg_idx" => seg_idx.into_tensor()?,
            "progress" => progress.into_tensor()?,
            "boundary" => boundary.into_tensor()?,
            "f0" => f0.into_tensor()?,
        ])?;
        let (shape, data) = outputs["mel"].try_extract_tensor::<f32>()?;
        Ok(TensorData {
            shape: shape.to_vec(),
            data: data.to_vec(),
        })
    }
}
