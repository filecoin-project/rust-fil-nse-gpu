#[cfg(test)]
mod tests {
    use ff::{Field, PrimeField};
    use merkletree::store::StoreConfig;
    use paired::bls12_381::{Fr, FrRepr};
    use rand::{thread_rng, Rng};
    use rust_fil_nse_gpu::*;
    use storage_proofs::cache_key::CacheKey;
    use storage_proofs::hasher::poseidon;
    use storage_proofs::merkle::split_config;
    use storage_proofs::merkle::OctLCMerkleTree;
    use storage_proofs::porep::nse;

    pub const TEST_CONFIG: Config = Config {
        k: 2,
        num_nodes_window: 512, // Must be 2^(3*x) for 8-ary merkle trees
        degree_expander: 96,
        degree_butterfly: 4,
        num_expander_layers: 4,
        num_butterfly_layers: 3,
    };

    fn to_cpu_config(conf: Config) -> nse::Config {
        nse::Config {
            k: conf.k,
            num_nodes_window: conf.num_nodes_window,
            degree_expander: conf.degree_expander,
            degree_butterfly: conf.degree_butterfly,
            num_expander_layers: conf.num_expander_layers,
            num_butterfly_layers: conf.num_butterfly_layers,
            sector_size: 0,
        }
    }

    fn to_poseidon_domain(sha256: Sha256Domain) -> poseidon::PoseidonDomain {
        unsafe { std::mem::transmute::<Sha256Domain, poseidon::PoseidonDomain>(sha256) }
    }

    fn node_to_poseidon_domain(node: Node) -> poseidon::PoseidonDomain {
        unsafe { std::mem::transmute::<FrRepr, poseidon::PoseidonDomain>(node.0.into_repr()) }
    }

    fn accumulate(l: &Vec<Node>) -> Node {
        let mut acc = Fr::zero();
        for n in l.iter() {
            acc.add_assign(&n.0);
        }
        Node(acc)
    }

    fn u8_limbs_of<T>(value: T) -> Vec<u8> {
        unsafe {
            std::slice::from_raw_parts(&value as *const T as *const u8, std::mem::size_of::<T>())
                .to_vec()
        }
    }

    fn layer_to_vec_u8(l: &Layer) -> Vec<u8> {
        let mut ret = Vec::new();
        for n in l.0.iter() {
            ret.extend(u8_limbs_of(n.0.into_repr()));
        }
        ret
    }

    fn vec_u8_to_layer(v: &Vec<u8>) -> Layer {
        let mut nodes = Vec::new();
        for slice in v.chunks_exact(32) {
            let mut slc = [0u8; 32];
            slc.copy_from_slice(&slice[..]);
            nodes.push(Node(
                Fr::from_repr(unsafe { std::mem::transmute::<[u8; 32], FrRepr>(slc) }).unwrap(),
            ));
        }
        Layer(nodes)
    }

    #[test]
    fn test_expander_compatibility() {
        let mut rng = thread_rng();
        let mut gpu = GPU::new(TEST_CONFIG).unwrap();

        for _ in 0..10 {
            let prev_layer = Layer::random(&mut rng, TEST_CONFIG.num_nodes_window);
            let replica_id = Sha256Domain::random(&mut rng);
            let window_index: usize = rng.gen();
            let layer_index = 2;

            gpu.push_layer(&prev_layer).unwrap();
            let gpu_output = gpu
                .generate_expander_layer(replica_id, window_index, layer_index)
                .unwrap();

            let layer_a = layer_to_vec_u8(&prev_layer);
            let mut layer_b = layer_a.clone();
            nse::expander_layer(
                &to_cpu_config(TEST_CONFIG),
                window_index as u32,
                &to_poseidon_domain(replica_id),
                layer_index as u32,
                &layer_a,
                &mut layer_b,
            )
            .unwrap();
            let cpu_output = vec_u8_to_layer(&layer_b);

            assert_eq!(accumulate(&cpu_output.0), accumulate(&gpu_output.0));
        }
    }

    #[test]
    fn test_butterfly_compatibility() {
        let mut rng = thread_rng();
        let mut gpu = GPU::new(TEST_CONFIG).unwrap();

        for _ in 0..10 {
            let prev_layer = Layer::random(&mut rng, TEST_CONFIG.num_nodes_window);
            let replica_id = Sha256Domain::random(&mut rng);
            let window_index: usize = rng.gen();
            let layer_index = 5;

            gpu.push_layer(&prev_layer).unwrap();
            let gpu_output = gpu
                .generate_butterfly_layer(replica_id, window_index, layer_index)
                .unwrap();

            let layer_a = layer_to_vec_u8(&prev_layer);
            let mut layer_b = layer_a.clone();
            nse::butterfly_layer(
                &to_cpu_config(TEST_CONFIG),
                window_index as u32,
                &to_poseidon_domain(replica_id),
                layer_index as u32,
                &layer_a,
                &mut layer_b,
            )
            .unwrap();
            let cpu_output = vec_u8_to_layer(&layer_b);

            assert_eq!(accumulate(&cpu_output.0), accumulate(&gpu_output.0));
        }
    }

    #[test]
    fn test_sealer_compatibility() {
        let mut rng = thread_rng();
        let mut gpu = GPU::new(TEST_CONFIG).unwrap();

        for _ in 0..10 {
            let data = Layer::random(&mut rng, TEST_CONFIG.num_nodes_window);
            let replica_id = Sha256Domain::random(&mut rng);
            let window_index: usize = rng.gen();
            let sealer = Sealer::new(
                TEST_CONFIG,
                replica_id,
                window_index,
                data.clone(),
                &mut gpu,
                true,
                2,
            )
            .unwrap();

            let gpu_layers = sealer.map(|r| r.unwrap()).collect::<Vec<_>>();
            let gpu_output = gpu_layers.iter().last().unwrap().base.clone();
            let gpu_roots = gpu_layers.iter().map(|l| {
                assert_eq!(l.tree.len(), 1);
                node_to_poseidon_domain(l.tree[0])
            });

            let cpu_config = to_cpu_config(TEST_CONFIG);
            let cache_dir = tempfile::tempdir().unwrap();
            let store_config = StoreConfig::new(
                cache_dir.path(),
                CacheKey::CommDTree.to_string(),
                StoreConfig::default_cached_above_base_layer(
                    cpu_config.num_nodes_window as usize,
                    8,
                ),
            );
            let store_configs =
                split_config(store_config.clone(), cpu_config.num_layers()).unwrap();
            let mut cpu_output = layer_to_vec_u8(&data);
            let (cpu_trees, _) =
                nse::encode_with_trees::<OctLCMerkleTree<poseidon::PoseidonHasher>>(
                    &cpu_config,
                    store_configs,
                    window_index as u32,
                    &to_poseidon_domain(replica_id),
                    &mut cpu_output,
                )
                .unwrap();
            let cpu_output = vec_u8_to_layer(&cpu_output);
            let cpu_roots = cpu_trees.iter().map(|t| t.root());

            assert_eq!(gpu_output, cpu_output);
            for (gpu_root, cpu_root) in gpu_roots.zip(cpu_roots) {
                assert_eq!(gpu_root, cpu_root);
            }
        }
    }
}
