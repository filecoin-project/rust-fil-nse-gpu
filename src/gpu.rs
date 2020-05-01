use super::{Layer, NarrowStackedExpander, COMBINE_BATCH_SIZE};
use paired::bls12_381::Fr;

pub struct GPU {
    leaf_count: usize,
    combine_batch_size: usize,
}

impl NarrowStackedExpander for GPU {
    fn new(leaf_count: usize) -> Self {
        GPU {
            leaf_count,
            combine_batch_size: COMBINE_BATCH_SIZE,
        }
    }
    fn generate_mask_layer(&mut self, _replica_id: Fr, _window_index: usize) -> Layer {
        unimplemented!()
    }
    fn generate_expander_layer(&mut self, _layer_index: usize) -> Layer {
        unimplemented!()
    }
    fn generate_butterfly_layer(&mut self, _layer_index: usize) -> Layer {
        unimplemented!()
    }
    fn combine_segment(&self, _offset: usize, _segment: &[Fr]) -> Vec<Fr> {
        unimplemented!()
    }
    fn combine_batch_size(&self) -> usize {
        self.combine_batch_size
    }
    fn leaf_count(&self) -> usize {
        self.leaf_count
    }
}
