use rand::{thread_rng, Rng};
use rust_fil_nse_gpu::*;
use std::time::Instant;
use structopt::StructOpt;

macro_rules! timer {
    ($e:expr, $samples:expr) => {{
        let before = Instant::now();
        for _ in 0..$samples {
            $e;
        }
        (before.elapsed().as_secs() * 1000 as u64 + before.elapsed().subsec_millis() as u64)
            / ($samples as u64)
    }};
}

fn bench_mask(conf: Config, samples: usize) -> u64 {
    let mut rng = thread_rng();
    let replica_id = Sha256Domain::random(&mut rng);
    let window_index: usize = rng.gen();
    let mut gpu = GPU::new(conf).unwrap();
    timer!(
        gpu.generate_mask_layer(replica_id, window_index).unwrap(),
        samples
    )
}

fn bench_expander(conf: Config, samples: usize) -> u64 {
    let mut rng = thread_rng();
    let replica_id = Sha256Domain::random(&mut rng);
    let window_index: usize = rng.gen();
    let mut gpu = GPU::new(conf).unwrap();
    gpu.generate_mask_layer(replica_id, window_index).unwrap();
    timer!(
        gpu.generate_expander_layer(replica_id, window_index, rng.gen())
            .unwrap(),
        samples
    )
}

fn bench_butterfly(conf: Config, samples: usize) -> u64 {
    let mut rng = thread_rng();
    let replica_id = Sha256Domain::random(&mut rng);
    let window_index: usize = rng.gen();
    let mut gpu = GPU::new(conf).unwrap();
    gpu.generate_mask_layer(replica_id, window_index).unwrap();
    timer!(
        gpu.generate_butterfly_layer(replica_id, window_index, rng.gen())
            .unwrap(),
        samples
    )
}

fn bench_combine(conf: Config, samples: usize) -> u64 {
    let mut rng = thread_rng();
    let replica_id = Sha256Domain::random(&mut rng);
    let window_index: usize = rng.gen();
    let data = Layer::random(&mut rng, conf.num_nodes_window);
    let mut gpu = GPU::new(conf).unwrap();
    gpu.generate_mask_layer(replica_id, window_index).unwrap();
    timer!(gpu.combine_layer(&data, false).unwrap(), samples)
}

#[derive(Debug, StructOpt, Clone, Copy)]
#[structopt(name = "NSE Bench", about = "Benchmarking NSE operations on GPU.")]
struct Opts {
    #[structopt(short = "k", default_value = "8")]
    k: u32,
    #[structopt(long = "num-nodes-window", default_value = "524288")]
    num_nodes_window: usize,
    #[structopt(long = "degree-expander", default_value = "384")]
    degree_expander: usize,
    #[structopt(long = "degree-butterfly", default_value = "16")]
    degree_butterfly: usize,
    #[structopt(long = "num-expander-layers", default_value = "8")]
    num_expander_layers: usize,
    #[structopt(long = "num-butterfly-layers", default_value = "7")]
    num_butterfly_layers: usize,
    #[structopt(long = "samples", default_value = "10")]
    samples: usize,
}

impl From<Opts> for Config {
    fn from(cli: Opts) -> Self {
        Config {
            k: cli.k,
            num_nodes_window: cli.num_nodes_window,
            degree_expander: cli.degree_expander,
            degree_butterfly: cli.degree_butterfly,
            num_expander_layers: cli.num_expander_layers,
            num_butterfly_layers: cli.num_butterfly_layers,
        }
    }
}

fn main() {
    let opts = Opts::from_args();
    println!("Options: {:?}", opts);

    let config: Config = Config::from(opts);
    println!("Mask: {}ms", bench_mask(config, opts.samples));
    println!("Expander: {}ms", bench_expander(config, opts.samples));
    println!("Butterfly: {}ms", bench_butterfly(config, opts.samples));
    println!("Combine: {}ms", bench_combine(config, opts.samples));
}
