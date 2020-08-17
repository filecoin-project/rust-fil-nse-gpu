use super::{
    sources, Config, GPUError, GPUResult, Layer, NSEResult, NarrowStackedExpander, Node, ReplicaId,
    COMBINE_BATCH_SIZE,
};
use generic_array::typenum::U8;
use log::info;
use neptune::batch_hasher::BatcherType;
use neptune::cl::GPUSelector;
use neptune::tree_builder::TreeBuilder;
use rust_gpu_tools::opencl as cl;
use rust_gpu_tools::*;

unsafe impl cl::Parameter for Node {}
unsafe impl cl::Parameter for ReplicaId {}

#[derive(Debug, Clone, Copy)]
pub enum TreeOptions {
    Enabled { rows_to_discard: usize },
    Disabled,
}

// Manages buffers
pub struct GPUContext {
    program: cl::Program,
    tree_builder: Option<TreeBuilder<U8>>,
    config: Config,
}

impl GPUContext {
    pub fn default(config: Config, tree_options: TreeOptions) -> NSEResult<GPUContext> {
        GPUContext::new(cl::Device::all()?[0].clone(), config, tree_options)
    }

    pub fn new(
        device: cl::Device,
        config: Config,
        tree_options: TreeOptions,
    ) -> NSEResult<GPUContext> {
        info!(
            "Initializing a new NSE GPU context on device: {}",
            device.name()
        );

        if !device.is_little_endian()? {
            Err(GPUError::Other("Device should be little-endian!".into()))?;
        }

        info!("Compiling kernels...");
        let code = sources::generate_nse_program(config);
        let program = cl::Program::from_opencl(device.clone(), &code)?;

        Ok(GPUContext {
            program,
            config,
            tree_builder: match tree_options {
                TreeOptions::Enabled { rows_to_discard } => Some(TreeBuilder::<U8>::new(
                    Some(BatcherType::CustomGPU(GPUSelector::BusId(device.bus_id()))),
                    config.num_nodes_window,
                    TREE_BUILDER_BATCH_SIZE,
                    rows_to_discard,
                )?),
                TreeOptions::Disabled => None,
            },
        })
    }

    pub(crate) fn build_kernel(&mut self, kernel_name: &str) -> cl::Kernel {
        info!("Calling {}()...", kernel_name);
        self.program
            .create_kernel(kernel_name, self.leaf_count(), None)
    }

    pub(crate) fn create_buffer(&mut self) -> GPUResult<cl::Buffer<Node>> {
        info!("Creating buffer...");
        Ok(self.program.create_buffer::<Node>(self.leaf_count())?)
    }

    pub(crate) fn leaf_count(&self) -> usize {
        self.config.num_nodes_window
    }
}

const TREE_BUILDER_BATCH_SIZE: usize = 400_000;

pub struct GPU {
    context: GPUContext,
    combine_batch_size: usize,
    current_layer: cl::Buffer<Node>, // This has the last generated layer (In ordinary form)
    pub config: Config,
}

impl GPU {
    pub fn tree_builder(&mut self) -> &mut Option<TreeBuilder<U8>> {
        &mut self.context.tree_builder
    }

    fn replace_buffer(&mut self, buff: cl::Buffer<Node>) {
        std::mem::replace(&mut self.current_layer, buff);
    }

    // Overwrite current layer
    pub fn push_layer(&mut self, layer: &Layer) -> NSEResult<()> {
        self.current_layer.write_from(0, &layer.0)?; // Push montgomery form in buffer
        let ordinary = self.context.create_buffer()?; // Create new temp buffer
        call_kernel!(
            self.context.build_kernel("generate_ordinary"),
            &self.current_layer,
            &ordinary
        )?;
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
            self.context.build_kernel("generate_mask"),
            &ord_output,
            replica_id,
            window_index as u32
        )?;
        call_kernel!(
            self.context.build_kernel("generate_montgomery"),
            &ord_output,
            &self.current_layer
        )?;
        self.current_layer.read_into(0, &mut l.0)?;
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
            self.context.build_kernel("generate_expander"),
            &self.current_layer,
            &ord_output,
            replica_id,
            window_index as u32,
            layer_index as u32
        )?;
        call_kernel!(
            self.context.build_kernel("generate_montgomery"),
            &ord_output,
            &self.current_layer
        )?;
        self.current_layer.read_into(0, &mut l.0)?;
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
            self.context.build_kernel("generate_butterfly"),
            &self.current_layer,
            &ord_output,
            replica_id,
            window_index as u32,
            layer_index as u32
        )?;
        call_kernel!(
            self.context.build_kernel("generate_montgomery"),
            &ord_output,
            &self.current_layer
        )?;
        self.current_layer.read_into(0, &mut l.0)?;
        self.replace_buffer(ord_output);
        Ok(l)
    }

    fn finalize(&mut self) -> NSEResult<()> {
        call_kernel!(
            self.context.build_kernel("to_montgomery"),
            &self.current_layer
        )?;
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
        data.write_from(offset, &segment)?;
        call_kernel!(
            self.context.build_kernel("combine_segment"),
            &self.current_layer,
            &data,
            offset as u32,
            segment.len() as u32,
            is_decode as u32
        )?;
        data.read_into(offset, &mut l)?;
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
                "14782940608458749152052068546707738955373949446874699092685797723589199108169"
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
                "22705938218269600582111888137759768785425241059162326580787572603300938432305"
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
                "957721749774935859819530955474230185920900559990452793231841687285976045738"
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
