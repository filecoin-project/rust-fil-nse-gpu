use super::{
    sources, Config, GPUResult, Layer, NSEResult, NarrowStackedExpander, Node, COMBINE_BATCH_SIZE,
};
use ocl::builders::KernelBuilder;
use ocl::{Buffer, Device, OclPrm, Platform, ProQue};

const GLOBAL_WORK_SIZE: usize = 2048;

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

// Manages buffers
pub struct GPUContext {
    pro_que: ProQue,
    current_layer: Buffer<Node>, // This has the last generated layer
    kernel_buffer: Buffer<Node>, // Kernel output always goes here (Sometimes is used as an input)
}

impl GPUContext {
    pub fn new(device: Device, node_count: usize) -> GPUResult<GPUContext> {
        let code = sources::generate_nse_program();
        let pro_que = ProQue::builder()
            .device(device)
            .src(code)
            .dims(node_count)
            .build()?;
        let current_layer = pro_que.create_buffer::<Node>()?;
        let kernel_buffer = pro_que.create_buffer::<Node>()?;
        Ok(GPUContext {
            pro_que,
            current_layer,
            kernel_buffer,
        })
    }

    pub(crate) fn build_kernel(
        &mut self,
        global_work_size: usize,
        kernel_name: &str,
    ) -> KernelBuilder {
        let mut k = self.pro_que.kernel_builder(kernel_name);
        k.global_work_size([global_work_size]);

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

    pub(crate) fn push_buffer(&mut self, segment: &Vec<Node>, offset: usize) -> GPUResult<()> {
        self.kernel_buffer
            .create_sub_buffer(None, offset, segment.len())?
            .write(segment)
            .enq()?;
        Ok(())
    }

    pub(crate) fn pull_buffer(&mut self, segment: &mut Vec<Node>, offset: usize) -> GPUResult<()> {
        self.kernel_buffer
            .create_sub_buffer(None, offset, segment.len())?
            .read(segment)
            .enq()?;
        Ok(())
    }
}

macro_rules! call_kernel {
    ($ctx:expr, $name:expr) => {{
        let kernel =
            $ctx
            .build_kernel(GLOBAL_WORK_SIZE, $name)
            .build()?;
        unsafe {
            kernel.enq()?;
        }
    }};
    ($ctx:expr, $name:expr, $($arg:expr),+) => {{
        let kernel =
            $ctx
            .build_kernel(GLOBAL_WORK_SIZE, $name)
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

impl NarrowStackedExpander for GPU {
    fn new(config: Config) -> NSEResult<Self> {
        // Choose first NVIDIA GPU
        let device = *get_devices(GPU_NVIDIA_PLATFORM_NAME)?
            .first()
            .expect("GPU not found!");

        Ok(GPU {
            context: GPUContext::new(device, config.n)?,
            combine_batch_size: COMBINE_BATCH_SIZE,
            config,
        })
    }

    fn generate_mask_layer(&mut self, replica_id: Node, window_index: usize) -> NSEResult<Layer> {
        call_kernel!(self.context, "generate_mask", replica_id, window_index);
        let mut l = Layer(Vec::<Node>::with_capacity(self.config.n));
        self.context.pull_buffer(&mut l.0, 0)?;
        self.context.make_buffer_current();
        Ok(l)
    }

    fn generate_expander_layer(&mut self, layer_index: usize) -> NSEResult<Layer> {
        call_kernel!(self.context, "generate_expander", layer_index);
        let mut l = Layer(Vec::<Node>::with_capacity(self.config.n));
        self.context.pull_buffer(&mut l.0, 0)?;
        self.context.make_buffer_current();
        Ok(l)
    }

    fn generate_butterfly_layer(&mut self, layer_index: usize) -> NSEResult<Layer> {
        call_kernel!(self.context, "generate_butterfly", layer_index);
        let mut l = Layer(Vec::<Node>::with_capacity(self.config.n));
        self.context.pull_buffer(&mut l.0, 0)?;
        self.context.make_buffer_current();
        Ok(l)
    }

    fn combine_segment(&mut self, offset: usize, segment: &[Node]) -> NSEResult<Vec<Node>> {
        self.context.push_buffer(&segment.to_vec(), offset)?;
        call_kernel!(self.context, "combine_segment");
        let mut l = Vec::<Node>::with_capacity(segment.len());
        self.context.pull_buffer(&mut l, offset)?;
        Ok(l)
    }

    fn combine_batch_size(&self) -> usize {
        self.combine_batch_size
    }

    fn leaf_count(&self) -> usize {
        self.config.n
    }
}
