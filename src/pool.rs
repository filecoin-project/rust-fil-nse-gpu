use super::scheduler::schedule;
use crate::NarrowStackedExpander;
use crate::{Config, GPUContext, LayerOutput, NSEResult, Sealer, SealerInput, TreeOptions, GPU};
use log::*;
use rust_gpu_tools::opencl as cl;
use std::sync::mpsc;
use std::sync::{Arc, Mutex};

pub struct SealerPool {
    devices: Vec<cl::Device>,
    config: Config,
    tree_options: TreeOptions,
}

impl SealerPool {
    pub fn new(
        devices: Vec<cl::Device>,
        config: Config,
        tree_options: TreeOptions,
    ) -> NSEResult<Self> {
        info!("Creating a sealer pool of {} devices.", devices.len());

        Ok(SealerPool {
            devices,
            config,
            tree_options,
        })
    }

    /// Gets a SealerInput and returns a receiving output channel as soon as a free GPU is found.
    /// Blocks if all GPUs are busy.
    pub fn seal_on_gpu(&mut self, inp: SealerInput) -> mpsc::Receiver<NSEResult<LayerOutput>> {
        let (tx, rx): (
            mpsc::Sender<NSEResult<LayerOutput>>,
            mpsc::Receiver<NSEResult<LayerOutput>>,
        ) = mpsc::channel();
        let tx = Arc::new(Mutex::new(tx));
        let config = self.config.clone();
        let tree_options = self.tree_options.clone();
        schedule(&self.devices, move |dev| {
            let tree_enabled =
                if let TreeOptions::Enabled { rows_to_discard: _ } = tree_options.clone() {
                    true
                } else {
                    false
                };
            let tx_result = Arc::clone(&tx);
            if let Err(e) = move || -> NSEResult<()> {
                let tx = tx.lock().unwrap();
                let ctx = GPUContext::new(dev.clone(), config.clone(), tree_options.clone())?;
                let mut gpu = GPU::new(ctx, config.clone())?;
                let sealer = Sealer::new(config.clone(), inp, &mut gpu, tree_enabled)?;
                for output in sealer {
                    // If receiving channel is dead
                    if tx.send(output).is_err() {
                        break;
                    }
                }
                Ok(())
            }() {
                let _ = tx_result.lock().unwrap().send(Err(e));
            }
        });
        return rx;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::*;
    use rand::{thread_rng, Rng};

    const TEST_CONFIG: Config = Config {
        k: 2,
        num_nodes_window: 512,
        degree_expander: 96,
        degree_butterfly: 4,
        num_expander_layers: 4,
        num_butterfly_layers: 3,
    };

    #[test]
    fn test_sealer_pool() {
        const NUM_RUNS: usize = 10;
        let mut rng = thread_rng();

        let inputs: Vec<SealerInput> = (0..NUM_RUNS)
            .map(|_| SealerInput {
                replica_id: ReplicaId::random(&mut rng),
                window_index: rng.gen(),
                original_data: Layer::random(&mut rng, TEST_CONFIG.num_nodes_window),
            })
            .collect();

        let pool_outputs = {
            let mut pool = SealerPool::new(
                cl::Device::all().unwrap(),
                TEST_CONFIG,
                TreeOptions::Enabled { rows_to_discard: 2 },
            )
            .unwrap();
            let pool_output_channels = inputs
                .iter()
                .map(|inp| pool.seal_on_gpu(inp.clone()))
                .collect::<Vec<_>>();
            pool_output_channels
                .into_iter()
                .map(|c| c.iter().collect::<NSEResult<Vec<_>>>().unwrap())
                .collect::<Vec<_>>()
        };

        let ctx =
            GPUContext::default(TEST_CONFIG, TreeOptions::Enabled { rows_to_discard: 2 }).unwrap();
        let mut gpu = GPU::new(ctx, TEST_CONFIG).unwrap();
        let normal_outputs = inputs
            .iter()
            .map(|inp| {
                Sealer::new(TEST_CONFIG, inp.clone(), &mut gpu, true)
                    .unwrap()
                    .map(|o| o.unwrap())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        assert_eq!(pool_outputs, normal_outputs);
    }
}
