#[cfg(test)]
mod tests {
    use ff::{Field, PrimeField};
    use paired::bls12_381::{Fr, FrRepr};
    use rand::{thread_rng, Rng};
    use rust_fil_nse_gpu::*;
    use storage_proofs::hasher::sha256;
    use storage_proofs::porep::nse;

    pub const TEST_CONFIG: Config = Config {
        k: 4,
        num_nodes_window: 1024,
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
                &sha256::Sha256Domain::from(replica_id.0),
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
                &sha256::Sha256Domain::from(replica_id.0),
                layer_index as u32,
                &layer_a,
                &mut layer_b,
            )
            .unwrap();
            let cpu_output = vec_u8_to_layer(&layer_b);

            assert_eq!(accumulate(&cpu_output.0), accumulate(&gpu_output.0));
        }
    }
}
