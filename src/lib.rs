mod sources;

use paired::bls12_381::Fr;

pub struct Layer {
    nodes: Vec<Fr>,
}

pub trait NarrowStackedExpander {
    fn generate_mask_layer(&mut self, replica_id: Fr, window_index: usize) -> Layer;
    fn generate_expander_layer(&mut self, layer_index: usize) -> Layer;
    fn generate_butterfly_layer(&mut self, layer_index: usize) -> Layer;
    fn combine_segment(&self, offset: usize, segment: &mut [Fr]);
}
