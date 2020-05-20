use super::{
    sources, Config, GPUResult, Layer, NSEResult, NarrowStackedExpander, Node, Sha256Domain,
    COMBINE_BATCH_SIZE,
};
use log::info;
use ocl::builders::KernelBuilder;
use ocl::{Buffer, Device, OclPrm, Platform, ProQue};

const GPU_NVIDIA_PLATFORM_NAME: &str = "NVIDIA CUDA";

/// Gets list of devices by platform name
/// * `platform_name` - Platfrom name, e.g. "NVIDIA CUDA"
fn get_devices(platform_name: &str) -> GPUResult<Vec<Device>> {
    let platform = Platform::list()?.into_iter().find(|&p| match p.name() {
        Ok(p) => p == platform_name,
        Err(_) => false,
    });
    match platform {
        Some(p) => Ok(Device::list_all(p)?),
        None => Ok(Vec::new()),
    }
}

// Make `Node` movable to GPU buffers by implementing `OclPrm`
unsafe impl OclPrm for Node {}
unsafe impl OclPrm for Sha256Domain {}

// Manages buffers
struct GPUContext {
    pro_que: ProQue,
    config: Config,
    current_layer: Buffer<Node>, // This has the last generated layer (In ordinary form)
    kernel_buffer: Buffer<Node>, // Kernel output always goes here (Sometimes is used as an input)
}

impl GPUContext {
    pub fn new(device: Device, config: Config) -> GPUResult<GPUContext> {
        info!(
            "Initializing a new NSE GPU context on device: {}",
            device.name()?
        );

        info!("Compiling kernels...");
        let code = sources::generate_nse_program(config);
        let pro_que = ProQue::builder()
            .device(device)
            .src(code)
            .dims(config.num_nodes_window)
            .build()?;

        info!("Creating buffers...");
        let current_layer = pro_que.create_buffer::<Node>()?;
        let kernel_buffer = pro_que.create_buffer::<Node>()?;
        Ok(GPUContext {
            pro_que,
            config,
            current_layer,
            kernel_buffer,
        })
    }

    pub(crate) fn build_kernel(&mut self, kernel_name: &str) -> KernelBuilder {
        info!("Calling {}()...", kernel_name);
        let mut k = self.pro_que.kernel_builder(kernel_name);
        k.global_work_size([self.leaf_count()]);

        // `current_layer` and `kernel_buffer` buffers are passed to
        // all kernel calls by default, any kernel argument passed
        // to `call_kernel!` comes after these.
        k.arg(&self.current_layer);
        k.arg(&self.kernel_buffer);

        k
    }

    pub(crate) fn make_buffer_current(&mut self) {
        std::mem::swap(&mut self.kernel_buffer, &mut self.current_layer);
    }

    pub(crate) fn push_current(&mut self, segment: &Vec<Node>, offset: usize) -> GPUResult<()> {
        info!("Pushing data...");
        self.current_layer
            .create_sub_buffer(None, offset, segment.len())?
            .write(segment)
            .enq()?;
        Ok(())
    }

    pub(crate) fn pull_current(&mut self, segment: &mut Vec<Node>, offset: usize) -> GPUResult<()> {
        info!("Pulling results...");
        self.current_layer
            .create_sub_buffer(None, offset, segment.len())?
            .read(segment)
            .enq()?;
        Ok(())
    }

    pub(crate) fn leaf_count(&self) -> usize {
        self.config.num_nodes_window
    }
}

macro_rules! call_kernel {
    ($ctx:expr, $name:expr) => {{
        let kernel =
            $ctx
            .build_kernel($name)
            .build()?;
        unsafe {
            kernel.enq()?;
        }
    }};
    ($ctx:expr, $name:expr, $($arg:expr),+) => {{
        let kernel =
            $ctx
            .build_kernel($name)
            $(.arg($arg))*
            .build()?;
        unsafe {
            kernel.enq()?;
        }
    }};
}

pub struct GPU {
    context: GPUContext,
    combine_batch_size: usize,
    pub config: Config,
}

impl GPU {
    // Overwrite current layer
    pub fn push_layer(&mut self, layer: &Layer) -> NSEResult<()> {
        self.context.push_current(&layer.0, 0)?; // Push montgomery form in current_layer
        call_kernel!(self.context, "generate_ordinary"); // Calculate ordinary form in kernel_buffer
        self.context.make_buffer_current(); // Use ordinary form as inputs of other kernels
        Ok(())
    }
}

impl NarrowStackedExpander for GPU {
    fn new(config: Config) -> NSEResult<Self> {
        // Choose first NVIDIA GPU
        let device = *get_devices(GPU_NVIDIA_PLATFORM_NAME)?
            .first()
            .expect("GPU not found!");

        Ok(GPU {
            context: GPUContext::new(device, config)?,
            combine_batch_size: COMBINE_BATCH_SIZE,
            config,
        })
    }

    fn generate_mask_layer(
        &mut self,
        replica_id: Sha256Domain,
        window_index: usize,
    ) -> NSEResult<Layer> {
        let mut l = Layer(vec![Node::default(); self.leaf_count()]);
        call_kernel!(
            self.context,
            "generate_mask",
            replica_id,
            window_index as u32
        );
        call_kernel!(self.context, "generate_montgomery");
        self.context.pull_current(&mut l.0, 0)?;
        self.context.make_buffer_current();
        Ok(l)
    }

    fn generate_expander_layer(
        &mut self,
        replica_id: Sha256Domain,
        window_index: usize,
        layer_index: usize,
    ) -> NSEResult<Layer> {
        let mut l = Layer(vec![Node::default(); self.leaf_count()]);
        call_kernel!(
            self.context,
            "generate_expander",
            replica_id,
            window_index as u32,
            layer_index as u32
        );
        call_kernel!(self.context, "generate_montgomery");
        self.context.pull_current(&mut l.0, 0)?;
        self.context.make_buffer_current();
        Ok(l)
    }

    fn generate_butterfly_layer(
        &mut self,
        replica_id: Sha256Domain,
        window_index: usize,
        layer_index: usize,
    ) -> NSEResult<Layer> {
        let mut l = Layer(vec![Node::default(); self.leaf_count()]);
        call_kernel!(
            self.context,
            "generate_butterfly",
            replica_id,
            window_index as u32,
            layer_index as u32
        );
        call_kernel!(self.context, "generate_montgomery");
        self.context.pull_current(&mut l.0, 0)?;
        self.context.make_buffer_current();
        Ok(l)
    }

    fn combine_segment(
        &mut self,
        offset: usize,
        segment: &[Node],
        is_decode: bool,
    ) -> NSEResult<Vec<Node>> {
        // Montgomery form of mask is in kernel_buffer!
        let mut l = vec![Node::default(); segment.len()];
        self.context.push_current(&segment.to_vec(), offset)?;
        call_kernel!(
            self.context,
            "combine_segment",
            offset as u32,
            segment.len() as u32,
            is_decode as u32
        );
        self.context.pull_current(&mut l, offset)?;
        Ok(l)
    }

    fn combine_batch_size(&self) -> usize {
        self.combine_batch_size
    }

    fn leaf_count(&self) -> usize {
        self.context.leaf_count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ff::{Field, PrimeField};
    use paired::bls12_381::Fr;

    const TEST_CONFIG: Config = Config {
        k: 4,
        num_nodes_window: 1024,
        degree_expander: 96,
        degree_butterfly: 4,
        num_expander_layers: 4,
        num_butterfly_layers: 3,
    };
    const TEST_WINDOW_INDEX: usize = 1234567890;
    const TEST_REPLICA_ID: Sha256Domain = Sha256Domain([123u8; 32]);

    pub fn accumulate(l: &Vec<Node>) -> Node {
        let mut acc = Fr::zero();
        for n in l.iter() {
            acc.add_assign(&n.0);
        }
        Node(acc)
    }

    pub fn incrementing_layer(start: usize, count: usize) -> Layer {
        Layer(
            (start..start + count)
                .map(|i| Node(Fr::from_str(&i.to_string()).unwrap()))
                .collect(),
        )
    }

    #[test]
    fn test_generate_mask_layer() {
        let mut gpu = GPU::new(TEST_CONFIG).unwrap();
        let l = gpu
            .generate_mask_layer(TEST_REPLICA_ID, TEST_WINDOW_INDEX)
            .unwrap();
        assert_eq!(
            Fr::from_str(
                "6446969232366391856858003439695628724183208016254828395100207087840708265392"
            )
            .unwrap(),
            accumulate(&l.0).0
        );
    }

    #[test]
    fn test_generate_expander_layer() {
        let mut gpu = GPU::new(TEST_CONFIG).unwrap();
        gpu.push_layer(&incrementing_layer(123, TEST_CONFIG.num_nodes_window))
            .unwrap();
        let l = gpu
            .generate_expander_layer(TEST_REPLICA_ID, TEST_WINDOW_INDEX, 2)
            .unwrap();
        assert_eq!(
            Fr::from_str(
                "31927618342922418711037965387576862711609979706171976983782310220611346538648"
            )
            .unwrap(),
            accumulate(&l.0).0
        );
    }

    #[test]
    fn test_generate_butterfly_layer() {
        let mut gpu = GPU::new(TEST_CONFIG).unwrap();
        gpu.push_layer(&incrementing_layer(345, TEST_CONFIG.num_nodes_window))
            .unwrap();
        let l = gpu
            .generate_butterfly_layer(TEST_REPLICA_ID, TEST_WINDOW_INDEX, 2)
            .unwrap();
        assert_eq!(
            Fr::from_str(
                "28446803097318130282256338067690839150703163945352000894825958507969324842746"
            )
            .unwrap(),
            accumulate(&l.0).0
        );
    }

    #[test]
    fn test_combine_layer() {
        let mut gpu = GPU::new(TEST_CONFIG).unwrap();
        let data = incrementing_layer(567, TEST_CONFIG.num_nodes_window);
        let mask = incrementing_layer(234, TEST_CONFIG.num_nodes_window);
        gpu.push_layer(&mask).unwrap();
        let encode = gpu.combine_segment(0, &data.0, false).unwrap();
        let decode = gpu.combine_segment(0, &data.0, true).unwrap();
        assert_eq!(Fr::from_str("1867776").unwrap(), accumulate(&encode).0);
        assert_eq!(Fr::from_str("340992").unwrap(), accumulate(&decode).0);
    }
}
