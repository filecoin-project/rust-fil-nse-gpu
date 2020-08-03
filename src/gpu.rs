use super::{
    sources, utils, Config, GPUResult, Layer, NSEResult, NarrowStackedExpander, Node, ReplicaId,
    COMBINE_BATCH_SIZE,
};
use generic_array::typenum::U8;
use log::info;
use neptune::batch_hasher::BatcherType;
use neptune::cl::GPUSelector;
use neptune::tree_builder::TreeBuilder;
use ocl::builders::KernelBuilder;
use ocl::{Buffer, Device, OclPrm, ProQue};

fn write_buffer<T: OclPrm>(buff: &mut Buffer<T>, offset: usize, segment: &[T]) -> GPUResult<()> {
    info!("Pushing data...");
    buff.create_sub_buffer(None, offset, segment.len())?
        .write(segment)
        .enq()?;
    Ok(())
}

fn read_buffer<T: OclPrm>(buff: &Buffer<T>, offset: usize, segment: &mut [T]) -> GPUResult<()> {
    info!("Pulling results...");
    buff.create_sub_buffer(None, offset, segment.len())?
        .read(segment)
        .enq()?;
    Ok(())
}

// Make `Node` movable to GPU buffers by implementing `OclPrm`
unsafe impl OclPrm for Node {}
unsafe impl OclPrm for ReplicaId {}

#[derive(Debug, Clone, Copy)]
pub enum TreeOptions {
    Enabled { rows_to_discard: usize },
    Disabled,
}

// Manages buffers
pub struct GPUContext {
    pro_que: ProQue,
    tree_builder: Option<TreeBuilder<U8>>,
    config: Config,
}

impl GPUContext {
    pub fn default(config: Config, tree_options: TreeOptions) -> NSEResult<GPUContext> {
        GPUContext::new(utils::default_device()?, config, tree_options)
    }

    pub fn new(device: Device, config: Config, tree_options: TreeOptions) -> NSEResult<GPUContext> {
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

        Ok(GPUContext {
            pro_que,
            config,
            tree_builder: match tree_options {
                TreeOptions::Enabled { rows_to_discard } => Some(TreeBuilder::<U8>::new(
                    Some(BatcherType::CustomGPU(GPUSelector::BusId(
                        utils::get_bus_id(device)?,
                    ))),
                    config.num_nodes_window,
                    TREE_BUILDER_BATCH_SIZE,
                    rows_to_discard,
                )?),
                TreeOptions::Disabled => None,
            },
        })
    }

    pub(crate) fn build_kernel(&mut self, kernel_name: &str) -> KernelBuilder {
        info!("Calling {}()...", kernel_name);
        let mut k = self.pro_que.kernel_builder(kernel_name);
        k.global_work_size([self.leaf_count()]);
        k
    }

    pub(crate) fn create_buffer(&mut self) -> GPUResult<Buffer<Node>> {
        info!("Creating buffer...");
        Ok(self.pro_que.create_buffer::<Node>()?)
    }

    pub(crate) fn leaf_count(&self) -> usize {
        self.config.num_nodes_window
    }
}

macro_rules! call_kernel {
    ($ctx:expr, $name:expr, $($arg:expr),*) => {{
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

const TREE_BUILDER_BATCH_SIZE: usize = 400_000;

pub struct GPU {
    context: GPUContext,
    combine_batch_size: usize,
    current_layer: Buffer<Node>, // This has the last generated layer (In ordinary form)
    pub config: Config,
}

impl GPU {
    pub fn tree_builder(&mut self) -> &mut Option<TreeBuilder<U8>> {
        &mut self.context.tree_builder
    }

    fn replace_buffer(&mut self, buff: Buffer<Node>) {
        std::mem::replace(&mut self.current_layer, buff);
    }

    // Overwrite current layer
    pub fn push_layer(&mut self, layer: &Layer) -> NSEResult<()> {
        write_buffer(&mut self.current_layer, 0, &layer.0)?; // Push montgomery form in buffer
        let ordinary = self.context.create_buffer()?; // Create new temp buffer
        call_kernel!(
            self.context,
            "generate_ordinary",
            &self.current_layer,
            &ordinary
        );
        self.replace_buffer(ordinary); // Current buffer has now the ordinary form
        Ok(())
    }
}

impl NarrowStackedExpander for GPU {
    fn new(mut context: GPUContext, config: Config) -> NSEResult<Self> {
        let current_layer = context.create_buffer()?;

        Ok(GPU {
            context,
            current_layer,
            combine_batch_size: COMBINE_BATCH_SIZE,
            config,
        })
    }

    fn generate_mask_layer(
        &mut self,
        replica_id: ReplicaId,
        window_index: usize,
    ) -> NSEResult<Layer> {
        let mut l = Layer(vec![Node::default(); self.leaf_count()]);
        let ord_output = self.context.create_buffer()?;
        call_kernel!(
            self.context,
            "generate_mask",
            &ord_output,
            replica_id,
            window_index as u32
        );
        call_kernel!(
            self.context,
            "generate_montgomery",
            &ord_output,
            &self.current_layer
        );
        read_buffer(&self.current_layer, 0, &mut l.0)?;
        self.replace_buffer(ord_output);
        Ok(l)
    }

    fn generate_expander_layer(
        &mut self,
        replica_id: ReplicaId,
        window_index: usize,
        layer_index: usize,
    ) -> NSEResult<Layer> {
        let mut l = Layer(vec![Node::default(); self.leaf_count()]);
        let ord_output = self.context.create_buffer()?;
        call_kernel!(
            self.context,
            "generate_expander",
            &self.current_layer,
            &ord_output,
            replica_id,
            window_index as u32,
            layer_index as u32
        );
        call_kernel!(
            self.context,
            "generate_montgomery",
            &ord_output,
            &self.current_layer
        );
        read_buffer(&self.current_layer, 0, &mut l.0)?;
        self.replace_buffer(ord_output);
        Ok(l)
    }

    fn generate_butterfly_layer(
        &mut self,
        replica_id: ReplicaId,
        window_index: usize,
        layer_index: usize,
    ) -> NSEResult<Layer> {
        let mut l = Layer(vec![Node::default(); self.leaf_count()]);
        let ord_output = self.context.create_buffer()?;
        call_kernel!(
            self.context,
            "generate_butterfly",
            &self.current_layer,
            &ord_output,
            replica_id,
            window_index as u32,
            layer_index as u32
        );
        call_kernel!(
            self.context,
            "generate_montgomery",
            &ord_output,
            &self.current_layer
        );
        read_buffer(&self.current_layer, 0, &mut l.0)?;
        self.replace_buffer(ord_output);
        Ok(l)
    }

    fn finalize(&mut self) -> NSEResult<()> {
        call_kernel!(self.context, "to_montgomery", &self.current_layer);
        Ok(())
    }

    fn combine_segment(
        &mut self,
        offset: usize,
        segment: &[Node],
        is_decode: bool,
    ) -> NSEResult<Vec<Node>> {
        // Montgomery form of mask is in kernel_buffer!
        let mut l = vec![Node::default(); segment.len()];
        let mut data = self.context.create_buffer()?;
        write_buffer(&mut data, offset, &segment)?;
        call_kernel!(
            self.context,
            "combine_segment",
            &self.current_layer,
            &data,
            offset as u32,
            segment.len() as u32,
            is_decode as u32
        );
        read_buffer(&data, offset, &mut l)?;
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
    const TEST_REPLICA_ID: ReplicaId = ReplicaId([123u8; 32]);

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
        let ctx = GPUContext::default(TEST_CONFIG, TreeOptions::Disabled).unwrap();
        let mut gpu = GPU::new(ctx, TEST_CONFIG).unwrap();
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
        let ctx = GPUContext::default(TEST_CONFIG, TreeOptions::Disabled).unwrap();
        let mut gpu = GPU::new(ctx, TEST_CONFIG).unwrap();
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
        let ctx = GPUContext::default(TEST_CONFIG, TreeOptions::Disabled).unwrap();
        let mut gpu = GPU::new(ctx, TEST_CONFIG).unwrap();
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
        let ctx = GPUContext::default(TEST_CONFIG, TreeOptions::Disabled).unwrap();
        let mut gpu = GPU::new(ctx, TEST_CONFIG).unwrap();
        let data = incrementing_layer(567, TEST_CONFIG.num_nodes_window);
        let mask = incrementing_layer(234, TEST_CONFIG.num_nodes_window);
        gpu.push_layer(&mask).unwrap();
        gpu.finalize().unwrap();
        let encode = gpu.combine_segment(0, &data.0, false).unwrap();
        let decode = gpu.combine_segment(0, &data.0, true).unwrap();
        assert_eq!(Fr::from_str("1867776").unwrap(), accumulate(&encode).0);
        assert_eq!(Fr::from_str("340992").unwrap(), accumulate(&decode).0);
    }
}
