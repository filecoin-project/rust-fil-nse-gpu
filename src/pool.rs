use crate::NarrowStackedExpander;
use crate::{Config, GPUContext, LayerOutput, NSEResult, Sealer, SealerInput, TreeOptions, GPU};
use log::*;
use ocl::Device;
use std::sync::mpsc;
use std::sync::{Arc, Condvar, Mutex};
use std::thread;
use std::time::Duration;

struct SealerWorker {
    died: bool,
    busy: Arc<Mutex<bool>>,
    channel: mpsc::Sender<(SealerInput, mpsc::Sender<NSEResult<LayerOutput>>)>,
}

pub struct SealerPool {
    lock: Mutex<()>,
    cond: Arc<Condvar>,
    workers: Vec<SealerWorker>,
}

impl SealerPool {
    pub fn new(devices: Vec<Device>, config: Config, tree_options: TreeOptions) -> NSEResult<Self> {
        info!("Creating a sealer pool of {} devices.", devices.len());

        let mut workers = Vec::new();
        let cond = Arc::new(Condvar::new());

        let tree_enabled = if let TreeOptions::Enabled { rows_to_discard: _ } = tree_options {
            true
        } else {
            false
        };

        for (i, dev) in devices.into_iter().enumerate() {
            info!("Creating Sealer-Worker on device[{}]: {}", i, dev.name()?);

            let (fn_tx, fn_rx): (
                mpsc::Sender<(SealerInput, mpsc::Sender<NSEResult<LayerOutput>>)>,
                mpsc::Receiver<(SealerInput, mpsc::Sender<NSEResult<LayerOutput>>)>,
            ) = mpsc::channel();

            let busy = Arc::new(Mutex::new(false));
            workers.push(SealerWorker {
                channel: fn_tx,
                busy: Arc::clone(&busy),
                died: false,
            });

            let cond = Arc::clone(&cond);
            thread::spawn(move || {
                match GPUContext::new(dev, config.clone(), tree_options.clone())
                    .and_then(|ctx| GPU::new(ctx, config.clone()))
                {
                    Ok(mut gpu) => {
                        info!(
                            "Device[{}]: GPU context initialized, waiting for inputs...",
                            i
                        );

                        for (inp, sender) in fn_rx.into_iter() {
                            info!("Device[{}]: New sealing request!", i);
                            let mut busy = busy.lock().unwrap();
                            match Sealer::new(config.clone(), inp, &mut gpu, tree_enabled) {
                                Ok(sealer) => {
                                    for output in sealer {
                                        // If receiving channel is dead
                                        if sender.send(output).is_err() {
                                            error!("Device[{}]: Requester died!", i);
                                            break;
                                        }
                                    }
                                }
                                Err(e) => {
                                    error!("Device[{}]: Cannot create sealer! Error: {}", i, e);
                                }
                            }
                            *busy = false;
                            cond.notify_all(); // Notify that one GPU is not busy anymore
                            info!("Device[{}]: Sealing finished, waiting for inputs...", i);
                        }
                    }
                    Err(e) => {
                        error!("Device[{}]: Cannot create GPU context! Error: {}", i, e);
                    }
                }
                warn!("Device[{}]: Worker died.", i);
                cond.notify_all(); // Notify when worker dies
            });
        }
        Ok(SealerPool {
            workers,
            lock: Mutex::new(()),
            cond,
        })
    }

    /// Gets a SealerInput and returns a receiving output channel as soon as a free GPU is found.
    /// Blocks if all GPUs are busy.
    pub fn seal_on_gpu(&mut self, inp: SealerInput) -> mpsc::Receiver<NSEResult<LayerOutput>> {
        const TIMEOUT: Duration = Duration::from_millis(5000);

        // Lock until a free GPU is found
        let mut lock = self.lock.lock().unwrap();

        loop {
            // Try finding a free GPU
            for worker in self.workers.iter_mut().filter(|w| !w.died) {
                // Check if GPU is free
                match worker.busy.try_lock() {
                    Ok(mut busy) => {
                        if !*busy {
                            *busy = true;
                            // A free GPU found! Create a communication channel and pass inputs
                            let (tx, rx): (
                                mpsc::Sender<NSEResult<LayerOutput>>,
                                mpsc::Receiver<NSEResult<LayerOutput>>,
                            ) = mpsc::channel();
                            if worker.channel.send((inp.clone(), tx)).is_err() {
                                warn!("Dead worker found! Marking as dead...");
                                worker.died = true;
                                continue;
                            }
                            return rx;
                        }
                    }
                    Err(_) => {}
                }
            }

            if self.workers.iter().filter(|w| !w.died).count() == 0 {
                panic!("No workers exist!");
            }

            // No free GPUs found, wait for a GPU to notify us
            info!("Waiting for a free GPU...");
            lock = self.cond.wait_timeout(lock, TIMEOUT).unwrap().0;
        }
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
                utils::all_devices().unwrap(),
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
