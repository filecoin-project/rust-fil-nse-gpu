mod error;
mod gpu;
mod pool;
mod scheduler;
mod sources;

pub use error::*;
use ff::{Field, PrimeField};
pub use gpu::*;
use neptune::tree_builder::TreeBuilderTrait;
use paired::bls12_381::{Fr, FrRepr};
pub use pool::*;
use rand::{Rng, RngCore};

// TODO: Move these constants into configuration of GPU, Sealer, KeyGenerator, etc.
const COMBINE_BATCH_SIZE: usize = 500000;

#[derive(PartialEq, Debug, Clone, Copy)]
#[repr(transparent)]
/// Nodes are always assumed to be in Montgomery Form.
pub struct Node(pub Fr);

pub const NODE_SIZE: usize = std::mem::size_of::<Node>();

impl Node {
    pub fn random<R: RngCore>(rng: &mut R) -> Self {
        Node(Fr::random(rng))
    }

    /// Convert a slice of `Node`s to a slice of `Fr`s.
    /// This conversion is accurate because `Node`s are in Montgomery Form.
    fn as_frs<'a>(nodes: &'a [Node]) -> &'a [Fr] {
        assert_eq!(
            std::mem::size_of::<Fr>(),
            std::mem::size_of::<Node>(),
            "Node should be zero-size wrapper around Fr representation"
        );

        unsafe { std::slice::from_raw_parts(nodes.as_ptr() as *const () as *const Fr, nodes.len()) }
    }

    /// Convert a slice of `Fr`s to a slice of `Node`s.
    /// This conversion is accurate because `Node`s are in Montgomery Form.
    fn from_frs<'a>(frs: &'a [Fr]) -> &'a [Node] {
        assert_eq!(
            std::mem::size_of::<Fr>(),
            std::mem::size_of::<Node>(),
            "Node should be zero-size wrapper around Fr representation"
        );

        unsafe { std::slice::from_raw_parts(frs.as_ptr() as *const () as *const Node, frs.len()) }
    }
}

impl Default for Node {
    fn default() -> Self {
        Node(Fr::zero())
    }
}

#[derive(PartialEq, Debug, Clone, Copy)]
pub struct ReplicaId(pub [u8; 32]);

impl Default for ReplicaId {
    fn default() -> Self {
        Self([0u8; 32])
    }
}

impl ReplicaId {
    pub fn random<R: RngCore>(rng: &mut R) -> Self {
        ReplicaId(rng.gen())
    }
}

#[derive(PartialEq, Debug, Clone, Default)]
pub struct Layer(pub Vec<Node>);

impl From<&Vec<u8>> for Layer {
    fn from(data: &Vec<u8>) -> Self {
        assert_eq!(std::mem::size_of::<FrRepr>(), NODE_SIZE);
        let mut nodes = Vec::with_capacity(data.len() / NODE_SIZE);
        let mut temp = [0u8; NODE_SIZE];
        for slice in data.chunks_exact(NODE_SIZE) {
            temp.copy_from_slice(&slice[..]);
            nodes.push(Node(
                Fr::from_repr(unsafe { std::mem::transmute::<[u8; NODE_SIZE], FrRepr>(temp) })
                    .unwrap(),
            ));
        }
        Layer(nodes)
    }
}

impl From<&Layer> for Vec<u8> {
    fn from(layer: &Layer) -> Self {
        assert_eq!(std::mem::size_of::<FrRepr>(), NODE_SIZE);
        let mut ret = Vec::with_capacity(layer.0.len() * NODE_SIZE);
        for n in layer.0.iter() {
            ret.extend(unsafe {
                std::slice::from_raw_parts(
                    &n.0.into_repr() as *const FrRepr as *const u8,
                    std::mem::size_of::<FrRepr>(),
                )
                .to_vec()
            });
        }
        ret
    }
}

#[derive(PartialEq, Debug, Clone)]
pub struct SealerInput {
    pub replica_id: ReplicaId,
    pub window_index: usize,
    pub original_data: Layer,
}

#[derive(PartialEq, Debug, Clone)]
pub struct LayerOutput {
    pub base: Layer,
    pub tree: Vec<Node>,
}

impl Layer {
    pub fn random<R: RngCore>(rng: &mut R, node_count: usize) -> Self {
        Layer((0..node_count).map(|_| Node::random(rng)).collect())
    }
}

pub trait NarrowStackedExpander: Sized {
    fn new(context: GPUContext, config: Config) -> NSEResult<Self>;
    fn generate_mask_layer(
        &mut self,
        replica_id: ReplicaId,
        window_index: usize,
    ) -> NSEResult<Layer>;
    fn generate_expander_layer(
        &mut self,
        replica_id: ReplicaId,
        window_index: usize,
        layer_index: usize,
    ) -> NSEResult<Layer>;
    fn generate_butterfly_layer(
        &mut self,
        replica_id: ReplicaId,
        window_index: usize,
        layer_index: usize,
    ) -> NSEResult<Layer>;
    fn finalize(&mut self) -> NSEResult<()>;
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
    build_trees: bool,
}

impl<'a> Sealer<'a> {
    pub fn new(
        config: Config,
        input: SealerInput,
        gpu: &'a mut GPU,
        build_trees: bool,
    ) -> NSEResult<Self> {
        Ok(Self {
            original_data: input.original_data,
            key_generator: KeyGenerator::new(config, input.replica_id, input.window_index, gpu)?,
            build_trees,
        })
    }

    pub fn seek(&mut self, target_layer_index: usize, target_layer_data: &Layer) -> NSEResult<()> {
        self.key_generator
            .seek(target_layer_index, target_layer_data)
    }

    pub fn new_from_layer(
        provided_layer_index: usize,
        provided_layer: &Layer,
        config: Config,
        input: SealerInput,
        gpu: &'a mut GPU,
        build_trees: bool,
    ) -> NSEResult<Self> {
        let mut sealer = Self::new(config, input, gpu, build_trees)?;
        sealer.seek(provided_layer_index, provided_layer)?;
        Ok(sealer)
    }
}

impl<'a> Iterator for Sealer<'a> {
    type Item = NSEResult<LayerOutput>;

    /// Returns successive layers, starting with mask layer, and ending with sealed replica layer.
    fn next(&mut self) -> Option<Self::Item> {
        if let Some(next_key_layer) = self.key_generator.next() {
            Some(|| -> NSEResult<LayerOutput> {
                let layer = if self.key_generator.layers_remaining() == 0 {
                    self.key_generator.combine_layer(&self.original_data, false)
                } else {
                    next_key_layer
                }?;
                if self.build_trees {
                    let tree_builder = self.key_generator.gpu.tree_builder().as_mut().unwrap(); // WARN: unwrap()
                    let frs = Node::as_frs(layer.0.as_slice());
                    let (_, fr_tree) = tree_builder.add_final_leaves(frs)?;
                    let tree = Node::from_frs(&fr_tree).to_vec();
                    Ok(LayerOutput { base: layer, tree })
                } else {
                    Ok(LayerOutput {
                        base: layer,
                        tree: Vec::new(), // Maybe change Vec<Node> to Option<Vec<Node>> and return None?
                    })
                }
            }())
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
    #[allow(dead_code)]
    key_generator: KeyGenerator<'a>,
}

impl<'a> Unsealer<'a> {
    pub fn new(
        config: Config,
        replica_id: ReplicaId,
        window_index: usize,
        gpu: &'a mut GPU,
    ) -> NSEResult<Self> {
        Ok(Self {
            key_generator: KeyGenerator::new(config, replica_id, window_index, gpu)?,
        })
    }

    #[allow(dead_code)]
    fn unseal_range(&mut self, offset: usize, sealed_data: &[Node]) -> NSEResult<Vec<Node>> {
        while let Some(layer) = self.key_generator.next() {
            layer?;
        }

        self.key_generator
            .combine_segment(offset, sealed_data, true)
    }

    #[allow(dead_code)]
    fn unseal_layer(&mut self, sealed: Layer) -> NSEResult<Layer> {
        Ok(Layer(self.unseal_range(0, &sealed.0)?))
    }
}

pub struct KeyGenerator<'a> {
    replica_id: ReplicaId,
    window_index: usize,
    current_layer_index: usize,
    gpu: &'a mut GPU,
}

impl<'a> KeyGenerator<'a> {
    fn new(
        config: Config,
        replica_id: ReplicaId,
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
    pub fn seek(&mut self, target_layer_index: usize, target_layer_data: &Layer) -> NSEResult<()> {
        self.current_layer_index = target_layer_index + 1;
        self.gpu.push_layer(&target_layer_data)
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

    fn finalize(&mut self) -> NSEResult<()> {
        self.gpu.finalize()
    }

    fn combine_segment(
        &mut self,
        offset: usize,
        segment: &[Node],
        is_decode: bool,
    ) -> NSEResult<Vec<Node>> {
        self.gpu.combine_segment(offset, segment, is_decode)
    }

    fn last_index(&self) -> usize {
        self.config().num_expander_layers + self.config().num_butterfly_layers
    }
}

impl<'a> Iterator for KeyGenerator<'a> {
    type Item = NSEResult<Layer>;

    fn next(&mut self) -> Option<Self::Item> {
        let last_index = self.last_index();

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
        if self.current_layer_index <= self.config().num_expander_layers {
            return Some(self.generate_expander_layer());
        }

        // When current index equals last index (having been incremented since the first check),
        // we need to generate the last butterfly layer. Before that, generate earlier butterfly layers.
        if self.current_layer_index <= last_index {
            return Some(self.generate_butterfly_layer().and_then(|l| {
                if self.current_layer_index == last_index {
                    self.finalize()?;
                }
                Ok(l)
            }));
        };

        unreachable!();
    }
}

impl<'a> ExactSizeIterator for KeyGenerator<'a> {
    fn len(&self) -> usize {
        self.config().num_expander_layers + self.config().num_butterfly_layers
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ff::PrimeField;
    use paired::bls12_381::{Fr, FrRepr};

    const TEST_CONFIG: Config = Config {
        k: 2,
        num_nodes_window: 512,
        degree_expander: 96,
        degree_butterfly: 4,
        num_expander_layers: 4,
        num_butterfly_layers: 3,
    };
    const TEST_WINDOW_INDEX: usize = 1234567890;
    const TEST_REPLICA_ID: ReplicaId = ReplicaId([123u8; 32]);

    pub fn incrementing_layer(start: usize, count: usize) -> Layer {
        Layer(
            (start..start + count)
                .map(|i| Node(Fr::from_str(&i.to_string()).unwrap()))
                .collect(),
        )
    }

    #[test]
    fn test_sealer() {
        let ctx =
            GPUContext::default(TEST_CONFIG, TreeOptions::Enabled { rows_to_discard: 2 }).unwrap();
        let mut gpu = GPU::new(ctx, TEST_CONFIG).unwrap();
        let original_data = incrementing_layer(123, TEST_CONFIG.num_nodes_window);
        let sealer = Sealer::new(
            TEST_CONFIG,
            SealerInput {
                replica_id: TEST_REPLICA_ID,
                window_index: TEST_WINDOW_INDEX,
                original_data: original_data.clone(),
            },
            &mut gpu,
            true,
        )
        .unwrap();

        let layer_index_to_restart = 3;
        let layers = sealer.map(|x| x).collect::<NSEResult<Vec<_>>>().unwrap();
        let roots = layers
            .iter()
            .map(|l| {
                assert_eq!(1, l.tree.len());
                l.tree[l.tree.len() - 1]
            })
            .collect::<Vec<_>>();

        let layer_to_restart = &layers[layer_index_to_restart].base;

        assert_eq!(
            roots[..7].to_vec(),
            vec![
                Node(
                    Fr::from_repr(FrRepr([
                        0xba0280418b23a3c4,
                        0x3d49ba6a2e216f46,
                        0x65c38ddf4cde76f3,
                        0x318dfbdd1cb3c701
                    ]))
                    .unwrap()
                ),
                Node(
                    Fr::from_repr(FrRepr([
                        0xaf6ffcb87a51a03f,
                        0x4d8bff6bcf6f1961,
                        0xb0d8e687e2d4c464,
                        0x51d4ffb9de3a8a18
                    ]))
                    .unwrap()
                ),
                Node(
                    Fr::from_repr(FrRepr([
                        0xfa3ff052b0ef4205,
                        0x285abfefd2861b21,
                        0xf884bc316ee69685,
                        0x4f39580cc5bafd0e
                    ]))
                    .unwrap()
                ),
                Node(
                    Fr::from_repr(FrRepr([
                        0xdb9d5e5df6796064,
                        0x1aba373bd41020a2,
                        0x114f262788304089,
                        0x0bf168dec07a3ac4
                    ]))
                    .unwrap()
                ),
                Node(
                    Fr::from_repr(FrRepr([
                        0xd0382f0647cc5bc3,
                        0x9a719d1d53da5f80,
                        0x00aa9e6aa538779d,
                        0x5187e91d102035af
                    ]))
                    .unwrap()
                ),
                Node(
                    Fr::from_repr(FrRepr([
                        0x646451246f10344b,
                        0x8d6e697c269041c0,
                        0xc5f1b27e2bdfdcbf,
                        0x50c97f19b8f0e3e2
                    ]))
                    .unwrap()
                ),
                Node(
                    Fr::from_repr(FrRepr([
                        0x8d66677d0d708e31,
                        0x4c34772776327789,
                        0xf86d396bc0bd4cf2,
                        0x64238b97c181baaf
                    ]))
                    .unwrap()
                )
            ]
        );

        let mut restarted_sealer = Sealer::new_from_layer(
            layer_index_to_restart,
            layer_to_restart,
            TEST_CONFIG,
            SealerInput {
                replica_id: TEST_REPLICA_ID,
                window_index: TEST_WINDOW_INDEX,
                original_data: original_data.clone(),
            },
            &mut gpu,
            true,
        )
        .unwrap();

        let mut restarted_roots = Vec::new();

        for r in &mut restarted_sealer {
            let l = r.unwrap();
            assert_eq!(1, l.tree.len());
            restarted_roots.push(l.tree[l.tree.len() - 1]);
        }

        assert_eq!(
            &roots[layer_index_to_restart + 1..],
            restarted_roots.as_slice()
        );

        let seek_target = 2;
        restarted_sealer
            .seek(seek_target, &layers[seek_target].base)
            .unwrap();

        let sought_roots = restarted_sealer
            .map(|r| {
                let l = r.unwrap();
                assert_eq!(1, l.tree.len());
                l.tree[l.tree.len() - 1]
            })
            .collect::<Vec<_>>();

        assert_eq!(&roots[seek_target + 1..], sought_roots.as_slice());
    }

    #[test]
    fn test_sealer_unsealer_consistency() {
        use rand::thread_rng;

        let mut rng = thread_rng();
        let original_data = Layer::random(&mut rng, TEST_CONFIG.num_nodes_window);
        let replica_id = ReplicaId::random(&mut rng);
        let window_index: usize = rng.gen();

        let ctx = GPUContext::default(TEST_CONFIG, TreeOptions::Disabled).unwrap();
        let mut gpu = GPU::new(ctx, TEST_CONFIG).unwrap();

        let sealer = Sealer::new(
            TEST_CONFIG,
            SealerInput {
                replica_id,
                window_index,
                original_data: original_data.clone(),
            },
            &mut gpu,
            false,
        )
        .unwrap();

        let sealed_data = sealer.last().unwrap().unwrap().base;

        let mut unsealer = Unsealer::new(TEST_CONFIG, replica_id, window_index, &mut gpu).unwrap();

        let unsealed_data = unsealer.unseal_layer(sealed_data.clone()).unwrap();

        assert_eq!(unsealed_data, original_data);

        let unsealed_data2 = unsealer.unseal_layer(sealed_data.clone()).unwrap();
        assert_eq!(original_data, unsealed_data2);

        let chunk_size = 50;
        let mut offset = 0;
        sealed_data.0.chunks(chunk_size).for_each(|chunk| {
            let unsealed = unsealer.unseal_range(offset, &chunk).unwrap();
            let end = offset + chunk.len();

            assert_eq!(&original_data.0[offset..end], unsealed.as_slice());
            offset = end;
        })
    }
}
