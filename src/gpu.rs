use super::{Config, Layer, NSEResult, NarrowStackedExpander, Node, COMBINE_BATCH_SIZE};
use ocl::OclPrm;

unsafe impl OclPrm for Node {}

pub struct GPU {
    combine_batch_size: usize,
    pub config: Config,
}

impl NarrowStackedExpander for GPU {
    fn new(config: Config) -> NSEResult<Self> {
        Ok(GPU {
            combine_batch_size: COMBINE_BATCH_SIZE,
            config,
        })
    }
    fn generate_mask_layer(&mut self, _replica_id: Node, _window_index: usize) -> NSEResult<Layer> {
        unimplemented!()
    }
    fn generate_expander_layer(&mut self, _layer_index: usize) -> NSEResult<Layer> {
        unimplemented!()
    }
    fn generate_butterfly_layer(&mut self, _layer_index: usize) -> NSEResult<Layer> {
        unimplemented!()
    }
    fn combine_segment(&self, _offset: usize, _segment: &[Node]) -> NSEResult<Vec<Node>> {
        unimplemented!()
    }
    fn combine_batch_size(&self) -> usize {
        self.combine_batch_size
    }
    fn leaf_count(&self) -> usize {
        self.config.n
    }
}
