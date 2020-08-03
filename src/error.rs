#[derive(thiserror::Error, Debug)]
pub enum GPUError {
    #[error("Ocl Error: {0}")]
    Ocl(ocl::Error),
    #[error("GPUError: {0}")]
    Simple(&'static str),
}

#[allow(dead_code)]
pub type GPUResult<T> = std::result::Result<T, GPUError>;

impl From<ocl::Error> for GPUError {
    fn from(error: ocl::Error) -> Self {
        GPUError::Ocl(error)
    }
}

#[derive(thiserror::Error, Debug)]
pub enum NSEError {
    #[error("Ocl Error: {0}")]
    GPU(#[from] GPUError),
    #[error("Neptune Error: {0}")]
    Neptune(#[from] neptune::error::Error),
}

pub type NSEResult<T> = std::result::Result<T, NSEError>;

impl From<ocl::Error> for NSEError {
    fn from(error: ocl::Error) -> Self {
        NSEError::GPU(GPUError::Ocl(error))
    }
}
