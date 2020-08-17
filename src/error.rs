use rust_gpu_tools::opencl as cl;

#[derive(thiserror::Error, Debug)]
pub enum GPUError {
    #[error("Ocl Error: {0}")]
    OpenCL(#[from] cl::GPUError),
    #[error("Error: {0}")]
    Other(String),
}

#[allow(dead_code)]
pub type GPUResult<T> = std::result::Result<T, GPUError>;

#[derive(thiserror::Error, Debug)]
pub enum NSEError {
    #[error("Ocl Error: {0}")]
    GPU(#[from] GPUError),
    #[error("Neptune Error: {0}")]
    Neptune(#[from] neptune::error::Error),
}

pub type NSEResult<T> = std::result::Result<T, NSEError>;

impl From<cl::GPUError> for NSEError {
    fn from(error: cl::GPUError) -> Self {
        NSEError::GPU(GPUError::OpenCL(error))
    }
}
