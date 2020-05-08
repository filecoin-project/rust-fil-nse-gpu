use rand::thread_rng;
use rust_fil_nse_gpu::*;
use std::time::Instant;

fn main() {
    let config = Config {
        k: 8,
        num_nodes_window: 1 << 27,
        degree_expander: 384,
        degree_butterfly: 16,
        num_expander_layers: 8,
        num_butterfly_layers: 7,
    };
    let replica_id = Sha256Domain([123u8; 32]);
    let window_index = 1234567890;

    let mut rng = thread_rng();

    let original_data = Layer(
        (0..config.num_nodes_window)
            .map(|_| Node::random(&mut rng))
            .collect(),
    );

    let gpu = GPU::new(config).unwrap();
    let sealer = Sealer::new(config, replica_id, window_index, original_data, gpu).unwrap();

    let mut before = Instant::now();
    for _ in sealer {
        let dur =
            before.elapsed().as_secs() * 1000 as u64 + before.elapsed().subsec_millis() as u64;
        println!("Layer took: {}ms", dur);
        before = Instant::now();
    }
}
