mod gpu;
mod sources;

use ff::Field;
use gpu::*;
use paired::bls12_381::Fr;

// TODO: Move these constants into configuration of GPU, Sealer, KeyGenerator, etc.
const COMBINE_BATCH_SIZE: usize = 500000;

#[derive(PartialEq, Debug, Clone, Copy)]
#[repr(transparent)]
pub struct Node(pub Fr);

impl Default for Node {
    fn default() -> Self {
        Node(Fr::zero())
    }
}

pub struct Layer(Vec<Node>);

pub trait NarrowStackedExpander {
    fn new(leaf_count: usize) -> Self;
    fn generate_mask_layer(&mut self, replica_id: Node, window_index: usize) -> Layer;
    fn generate_expander_layer(&mut self, layer_index: usize) -> Layer;
    fn generate_butterfly_layer(&mut self, layer_index: usize) -> Layer;
    fn combine_layer(&self, layer: &Layer) -> Layer {
        let batch_size = self.combine_batch_size();
        let mut result = Vec::with_capacity(self.leaf_count());
        &(layer.0)
            .chunks(batch_size)
            .enumerate()
            .for_each(|(i, chunk)| result.extend(self.combine_segment(i * batch_size, chunk)));

        Layer(result)
    }
    fn combine_segment(&self, offset: usize, segment: &[Node]) -> Vec<Node>;
    fn combine_batch_size(&self) -> usize;
    fn leaf_count(&self) -> usize;
}

// NOTES:
// layers are 1-indexed,

/// The configuration parameters for NSE.
#[derive(Debug, Clone)]
pub struct Config {
    /// Batch hashing factor.
    pub k: u32,
    /// Window size in bytes.
    pub n: usize,
    /// Degree of the expander graph.
    pub degree_expander: usize,
    /// Degree of the butterfly graph.
    pub degree_butterfly: usize,
    /// Number of expander layers.
    pub num_expander_layers: usize, // 8
    /// Number of butterfly layers.
    pub num_butterfly_layers: usize, // 7
}

pub struct Sealer {
    original_data: Layer,
    key_generator: KeyGenerator,
}

impl Sealer {
    #[allow(dead_code)]
    fn new(
        config: Config,
        replica_id: Node,
        window_index: usize,
        original_data: Layer,
        gpu: GPU,
    ) -> Self {
        Self {
            original_data,
            key_generator: KeyGenerator::new(config, replica_id, window_index, gpu),
        }
    }
}

impl Iterator for Sealer {
    type Item = Layer;

    /// Returns successive layers, starting with mask layer, and ending with sealed replica layer.
    fn next(&mut self) -> Option<Self::Item> {
        if let Some(next_key_layer) = self.key_generator.next() {
            if self.key_generator.layers_remaining() == 0 {
                Some(self.key_generator.combine_layer(&self.original_data))
            } else {
                Some(next_key_layer)
            }
        } else {
            None
        }
    }
}

impl ExactSizeIterator for Sealer {
    fn len(&self) -> usize {
        self.key_generator.len()
    }
}

#[allow(dead_code)]
pub struct Unsealer {
    key_generator: KeyGenerator,
}

// FIXME/TODO
impl Unsealer {}

pub struct KeyGenerator {
    replica_id: Node,
    window_index: usize,
    config: Config,
    current_layer_index: usize,
    gpu: GPU,
}

impl KeyGenerator {
    fn new(config: Config, replica_id: Node, window_index: usize, gpu: GPU) -> Self {
        assert_eq!(config.n, gpu.leaf_count() * std::mem::size_of::<Node>());
        Self {
            replica_id,
            window_index,
            config,
            current_layer_index: 0, // Initial value of 0 means the current layer precedes any generated layer.
            gpu,
        }
    }
    fn layers_remaining(&self) -> usize {
        self.len() - self.current_layer_index
    }

    // Generate maske layer on GPU from seeds.
    fn generate_mask_layer(&mut self) -> Layer {
        self.gpu
            .generate_mask_layer(self.replica_id, self.window_index)
    }

    // Generate expander layer on GPU, using previous layer already loaded.
    fn generate_expander_layer(&mut self) -> Layer {
        self.gpu.generate_expander_layer(self.current_layer_index)
    }
    // Generate butterfly layer on GPU, using previous layer already loaded.
    fn generate_butterfly_layer(&mut self) -> Layer {
        self.gpu.generate_expander_layer(self.current_layer_index)
    }

    fn combine_layer(&mut self, layer: &Layer) -> Layer {
        self.gpu.combine_layer(layer)
    }
}

impl Iterator for KeyGenerator {
    type Item = Layer;

    fn next(&mut self) -> Option<Self::Item> {
        let last_index = self.config.num_expander_layers + self.config.num_butterfly_layers;

        // If current index is last, then we have already finished generating layers.
        if self.current_layer_index >= last_index {
            return None;
        }
        self.current_layer_index += 1;

        // First layer is mask layer.
        if self.current_layer_index == 1 {
            return Some(self.generate_mask_layer());
        }

        // When current index equals number of expander layers, we need to generate the last expander layer.
        // Before that, generate earlier expander layers.
        if self.current_layer_index <= self.config.num_expander_layers {
            return Some(self.generate_expander_layer());
        }

        // When current index equals last index (having been incremented since the first check),
        // we need to generate the last butterfly layer. Before that, generate earlier butterfly layers.
        if self.current_layer_index <= last_index {
            return Some(self.generate_butterfly_layer());
        };

        unreachable!();
    }
}

impl ExactSizeIterator for KeyGenerator {
    fn len(&self) -> usize {
        self.config.num_expander_layers + self.config.num_butterfly_layers
    }
}
