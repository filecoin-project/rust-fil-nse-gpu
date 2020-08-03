use crate::{GPUError, GPUResult};
use ocl::{Device, Platform};

pub fn get_bus_id(d: Device) -> GPUResult<u32> {
    const CL_DEVICE_PCI_BUS_ID_NV: u32 = 0x4008;
    let result = d.info_raw(CL_DEVICE_PCI_BUS_ID_NV)?;
    Ok((result[0] as u32)
        + ((result[1] as u32) << 8)
        + ((result[2] as u32) << 16)
        + ((result[3] as u32) << 24))
}

pub const GPU_NVIDIA_PLATFORM_NAME: &str = "NVIDIA CUDA";

pub fn get_devices(platform_name: &str) -> GPUResult<Vec<Device>> {
    let platform = Platform::list()?.into_iter().find(|&p| match p.name() {
        Ok(p) => p == platform_name,
        Err(_) => false,
    });
    match platform {
        Some(p) => Ok(Device::list_all(p)?),
        None => Err(GPUError::Simple("GPU platform not found!")),
    }
}

pub fn all_devices() -> GPUResult<Vec<Device>> {
    get_devices(GPU_NVIDIA_PLATFORM_NAME)
}

pub fn default_device() -> GPUResult<Device> {
    Ok(all_devices()?[0])
}
