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

fn bench_mask(gpu: &mut GPU, samples: usize) -> u64 {
    let mut rng = thread_rng();
    let replica_id = Sha256Domain::random(&mut rng);
    let window_index: usize = rng.gen();
    timer!(
        gpu.generate_mask_layer(replica_id, window_index).unwrap(),
        samples
    )
}

fn bench_expander(gpu: &mut GPU, samples: usize) -> u64 {
    let mut rng = thread_rng();
    let replica_id = Sha256Domain::random(&mut rng);
    let window_index: usize = rng.gen();
    gpu.generate_mask_layer(replica_id, window_index).unwrap();
    timer!(
        gpu.generate_expander_layer(replica_id, window_index, rng.gen())
            .unwrap(),
        samples
    )
}

fn bench_butterfly(gpu: &mut GPU, samples: usize) -> u64 {
    let mut rng = thread_rng();
    let replica_id = Sha256Domain::random(&mut rng);
    let window_index: usize = rng.gen();
    gpu.generate_mask_layer(replica_id, window_index).unwrap();
    timer!(
        gpu.generate_butterfly_layer(replica_id, window_index, rng.gen())
            .unwrap(),
        samples
    )
}

fn bench_combine(gpu: &mut GPU, samples: usize) -> u64 {
    let mut rng = thread_rng();
    let replica_id = Sha256Domain::random(&mut rng);
    let window_index: usize = rng.gen();
    let data = Layer::random(&mut rng, gpu.leaf_count());
    gpu.generate_mask_layer(replica_id, window_index).unwrap();
    timer!(gpu.combine_layer(&data, false).unwrap(), samples)
}

fn bench_sealer(gpu: &mut GPU, samples: usize, build_trees: bool) -> u64 {
    let mut rng = thread_rng();
    let replica_id = Sha256Domain::random(&mut rng);
    let window_index: usize = rng.gen();
    let data = Layer::random(&mut rng, gpu.leaf_count());
    timer!(
        {
            let sealer = Sealer::new(
                gpu.config,
                replica_id,
                window_index,
                data.clone(),
                gpu,
                build_trees,
                2,
            )
            .unwrap();
            for _ in sealer {}
        },
        samples
    )
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
    #[structopt(long = "trees")]
    build_trees: bool,
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
    env_logger::init();

    let opts = Opts::from_args();
    println!("Options: {:?}", opts);

    let config: Config = Config::from(opts);
    let mut gpu = GPU::new(config).unwrap();

    println!("Mask: {}ms", bench_mask(&mut gpu, opts.samples));
    println!("Expander: {}ms", bench_expander(&mut gpu, opts.samples));
    println!("Butterfly: {}ms", bench_butterfly(&mut gpu, opts.samples));
    println!("Combine: {}ms", bench_combine(&mut gpu, opts.samples));
    println!(
        "Sealer: {}ms",
        bench_sealer(&mut gpu, opts.samples, opts.build_trees)
    );
}
