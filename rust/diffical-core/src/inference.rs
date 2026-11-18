use anyhow::{Context, Result};
use ort::environment::{init as env_init, Environment};
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
        let env = env_init().build().context("create ort environment")?;
        let session = Session::builder(&env)?
            .commit_from_file(path)
            .with_context(|| format!("load {path:?}"))?;
        Ok(Self { _env: env, session })
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
            shape: shape.iter().map(|d| *d as i64).collect(),
            data: data.to_vec(),
        })
    }
}
