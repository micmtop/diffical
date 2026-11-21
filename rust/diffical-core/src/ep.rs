//! Execution provider selection from `DIFFICAL_EP`:
//! - `cpu` (default): CPUExecutionProvider only, most compatible.
//! - `auto`: let ONNX Runtime choose (CoreML on macOS); crashes have been seen
//!   inside Apple's Espresso kernels, so callers should be ready to fall back.
//! - `cuda`: CUDAExecutionProvider first, CPU second. Requires a build with the
//!   `cuda` feature and a CUDA runtime on the machine; silently degrades to the
//!   providers listed afterwards when CUDA cannot be loaded.

use ort::ep::CPU;
#[cfg(feature = "cuda")]
use ort::ep::CUDA;
use ort::ep::ExecutionProviderDispatch;
use ort::session::builder::AutoDevicePolicy;

pub fn auto_policy() -> AutoDevicePolicy {
    match std::env::var("DIFFICAL_EP").as_deref() {
        Ok("auto") | Ok("coreml") | Ok("gpu") => AutoDevicePolicy::MaxEfficiency,
        _ => AutoDevicePolicy::PreferCPU,
    }
}

pub fn providers() -> Vec<ExecutionProviderDispatch> {
    let mut list = Vec::new();
    #[cfg(feature = "cuda")]
    {
        if std::env::var("DIFFICAL_EP").as_deref() == Ok("cuda") {
            list.push(CUDA::default().build().error_on_failure());
        }
    }
    list.push(CPU::default().build());
    list
}
