mod error;
mod gpu;
mod sources;

use error::*;
use ff::{Field, PrimeField};
pub use gpu::*;
use paired::bls12_381::{Fr, FrRepr};
use rand::{Rng, RngCore};

// TODO: Move these constants into configuration of GPU, Sealer, KeyGenerator, etc.
const COMBINE_BATCH_SIZE: usize = 500000;

#[derive(PartialEq, Debug, Clone, Copy)]
#[repr(transparent)]
pub struct Node(pub FrRepr);

impl Node {
    pub fn random<R: RngCore>(rng: &mut R) -> Self {
        Node(Fr::random(rng).into_repr())
    }
}

impl Default for Node {
    fn default() -> Self {
        Node(Fr::zero().into_repr())
    }
}

#[derive(PartialEq, Debug, Clone, Copy)]
pub struct Sha256Domain(pub [u8; 32]);

impl Default for Sha256Domain {
    fn default() -> Self {
        Self([0u8; 32])
    }
}

impl Sha256Domain {
    pub fn random<R: RngCore>(rng: &mut R) -> Self {
        Sha256Domain(rng.gen())
    }
}

#[derive(PartialEq, Debug, Clone)]
pub struct Layer(pub Vec<Node>);

impl Layer {
    pub fn random<R: RngCore>(rng: &mut R, node_count: usize) -> Self {
        Layer((0..node_count).map(|_| Node::random(rng)).collect())
    }
}

pub trait NarrowStackedExpander: Sized {
    fn new(config: Config) -> NSEResult<Self>;
    fn generate_mask_layer(
        &mut self,
        replica_id: Sha256Domain,
        window_index: usize,
    ) -> NSEResult<Layer>;
    fn generate_expander_layer(
        &mut self,
        replica_id: Sha256Domain,
        window_index: usize,
        layer_index: usize,
    ) -> NSEResult<Layer>;
    fn generate_butterfly_layer(
        &mut self,
        replica_id: Sha256Domain,
        window_index: usize,
        layer_index: usize,
    ) -> NSEResult<Layer>;
    // Combine functions need to get `&mut self`, as they modify internal state of GPU buffers
    fn combine_layer(&mut self, layer: &Layer, is_decode: bool) -> NSEResult<Layer> {
        Ok(Layer(self.combine_segment(0, &layer.0, is_decode)?))
    }
    fn combine_segment(
        &mut self,
        offset: usize,
        segment: &[Node],
        is_decode: bool,
    ) -> NSEResult<Vec<Node>>;
    fn combine_batch_size(&self) -> usize;
    fn leaf_count(&self) -> usize;
}

// NOTES:
// layers are 1-indexed,

/// The configuration parameters for NSE.
#[derive(Debug, Clone, Copy)]
pub struct Config {
    /// Batch hashing factor.
    pub k: u32,
    /// Number of nodes per window
    pub num_nodes_window: usize,
    /// Degree of the expander graph.
    pub degree_expander: usize,
    /// Degree of the butterfly graph.
    pub degree_butterfly: usize,
    /// Number of expander layers.
    pub num_expander_layers: usize, // 8
    /// Number of butterfly layers.
    pub num_butterfly_layers: usize, // 7
}

pub struct Sealer<'a> {
    original_data: Layer,
    key_generator: KeyGenerator<'a>,
}

impl<'a> Sealer<'a> {
    pub fn new(
        config: Config,
        replica_id: Sha256Domain,
        window_index: usize,
        original_data: Layer,
        gpu: &'a mut GPU,
    ) -> NSEResult<Self> {
        Ok(Self {
            original_data,
            key_generator: KeyGenerator::new(config, replica_id, window_index, gpu)?,
        })
    }
}

impl<'a> Iterator for Sealer<'a> {
    type Item = Layer;

    /// Returns successive layers, starting with mask layer, and ending with sealed replica layer.
    fn next(&mut self) -> Option<Self::Item> {
        if let Some(next_key_layer) = self.key_generator.next() {
            if self.key_generator.layers_remaining() == 0 {
                Some(
                    // TODO: Remove `unwrap()`, handle errors
                    self.key_generator
                        .combine_layer(&self.original_data, false)
                        .unwrap(),
                )
            } else {
                Some(next_key_layer)
            }
        } else {
            None
        }
    }
}

impl<'a> ExactSizeIterator for Sealer<'a> {
    fn len(&self) -> usize {
        self.key_generator.len()
    }
}

pub struct Unsealer<'a> {
    sealed_data: Layer,
    key_generator: KeyGenerator<'a>,
}

impl<'a> Unsealer<'a> {
    pub fn new(
        config: Config,
        replica_id: Sha256Domain,
        window_index: usize,
        sealed_data: Layer,
        gpu: &'a mut GPU,
    ) -> NSEResult<Self> {
        Ok(Self {
            sealed_data,
            key_generator: KeyGenerator::new(config, replica_id, window_index, gpu)?,
        })
    }
}

impl<'a> Iterator for Unsealer<'a> {
    type Item = Layer;

    /// Returns successive layers, starting with mask layer, and ending with sealed replica layer.
    fn next(&mut self) -> Option<Self::Item> {
        if let Some(next_key_layer) = self.key_generator.next() {
            if self.key_generator.layers_remaining() == 0 {
                Some(
                    // TODO: Remove `unwrap()`, handle errors
                    self.key_generator
                        .combine_layer(&self.sealed_data, true)
                        .unwrap(),
                )
            } else {
                Some(next_key_layer)
            }
        } else {
            None
        }
    }
}

impl<'a> ExactSizeIterator for Unsealer<'a> {
    fn len(&self) -> usize {
        self.key_generator.len()
    }
}

pub struct KeyGenerator<'a> {
    replica_id: Sha256Domain,
    window_index: usize,
    current_layer_index: usize,
    gpu: &'a mut GPU,
}

impl<'a> KeyGenerator<'a> {
    fn new(
        config: Config,
        replica_id: Sha256Domain,
        window_index: usize,
        gpu: &'a mut GPU,
    ) -> NSEResult<Self> {
        assert_eq!(config.num_nodes_window, gpu.leaf_count());
        Ok(Self {
            replica_id,
            window_index,
            current_layer_index: 0, // Initial value of 0 means the current layer precedes any generated layer.
            gpu,
        })
    }

    fn config(&self) -> Config {
        self.gpu.config
    }

    fn layers_remaining(&self) -> usize {
        self.len() - self.current_layer_index
    }

    // Generate maske layer on GPU from seeds.
    fn generate_mask_layer(&mut self) -> NSEResult<Layer> {
        self.gpu
            .generate_mask_layer(self.replica_id, self.window_index)
    }

    // Generate expander layer on GPU, using previous layer already loaded.
    fn generate_expander_layer(&mut self) -> NSEResult<Layer> {
        self.gpu.generate_expander_layer(
            self.replica_id,
            self.window_index,
            self.current_layer_index,
        )
    }
    // Generate butterfly layer on GPU, using previous layer already loaded.
    fn generate_butterfly_layer(&mut self) -> NSEResult<Layer> {
        self.gpu.generate_butterfly_layer(
            self.replica_id,
            self.window_index,
            self.current_layer_index,
        )
    }

    fn combine_layer(&mut self, layer: &Layer, is_decode: bool) -> NSEResult<Layer> {
        self.gpu.combine_layer(layer, is_decode)
    }
}

impl<'a> Iterator for KeyGenerator<'a> {
    type Item = Layer;

    fn next(&mut self) -> Option<Self::Item> {
        let last_index = self.config().num_expander_layers + self.config().num_butterfly_layers;

        // If current index is last, then we have already finished generating layers.
        if self.current_layer_index >= last_index {
            return None;
        }
        self.current_layer_index += 1;

        // First layer is mask layer.
        if self.current_layer_index == 1 {
            // TODO: Remove `unwrap()`, handle errors
            return Some(self.generate_mask_layer().unwrap());
        }

        // When current index equals number of expander layers, we need to generate the last expander layer.
        // Before that, generate earlier expander layers.
        if self.current_layer_index <= self.config().num_expander_layers {
            // TODO: Remove `unwrap()`, handle errors
            return Some(self.generate_expander_layer().unwrap());
        }

        // When current index equals last index (having been incremented since the first check),
        // we need to generate the last butterfly layer. Before that, generate earlier butterfly layers.
        if self.current_layer_index <= last_index {
            // TODO: Remove `unwrap()`, handle errors
            return Some(self.generate_butterfly_layer().unwrap());
        };

        unreachable!();
    }
}

impl<'a> ExactSizeIterator for KeyGenerator<'a> {
    fn len(&self) -> usize {
        self.config().num_expander_layers + self.config().num_butterfly_layers
    }
}

#[test]
fn test_sealer_unsealer_consistency() {
    use rand::thread_rng;

    let config = Config {
        k: 4,
        num_nodes_window: 1 << 10,
        degree_expander: 384,
        degree_butterfly: 16,
        num_expander_layers: 8,
        num_butterfly_layers: 7,
    };

    let mut rng = thread_rng();
    let original_data = Layer::random(&mut rng, config.num_nodes_window);
    let replica_id = Sha256Domain::random(&mut rng);
    let window_index: usize = rng.gen();

    let mut gpu = GPU::new(config).unwrap();

    let sealer = Sealer::new(
        config,
        replica_id,
        window_index,
        original_data.clone(),
        &mut gpu,
    )
    .unwrap();
    let mut sealed_layers = Vec::new();
    for l in sealer {
        sealed_layers.push(l);
    }

    let sealed_data = sealed_layers.last().unwrap().clone();

    let unsealer = Unsealer::new(config, replica_id, window_index, sealed_data, &mut gpu).unwrap();
    let mut unsealed_layers = Vec::new();
    for l in unsealer {
        unsealed_layers.push(l);
    }

    let unsealed_data = unsealed_layers.last().unwrap().clone();

    assert_eq!(unsealed_data, original_data);
}
