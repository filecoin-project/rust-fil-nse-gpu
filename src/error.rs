#[derive(thiserror::Error, Debug)]
pub enum GPUError {
    #[error("Ocl Error: {0}")]
    Ocl(ocl::Error),
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
}

pub type NSEResult<T> = std::result::Result<T, NSEError>;
