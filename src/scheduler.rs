use lazy_static::lazy_static;
use rust_gpu_tools::opencl as cl;
use rust_gpu_tools::scheduler as sch;
use std::sync::Mutex;
use std::time::Duration;

const PATH: &str = "/tmp/gpus";
const POLL_INTERVAL: Duration = Duration::from_millis(1000);

lazy_static! {
    static ref SCHEDULER: Mutex<sch::Scheduler::<cl::Device>> = {
        Mutex::new(
            sch::Scheduler::<cl::Device>::new_with_poll_interval(PATH.into(), POLL_INTERVAL)
                .expect("Failed to create scheduler"),
        )
    };
    static ref HANDLE: Mutex<Option<sch::SchedulerHandle>> = Mutex::new(None);
}

pub fn schedule<F>(devs: &Vec<cl::Device>, f: F)
where
    F: FnOnce(&cl::Device) + Send + Sync + 'static,
{
    let mut handle = HANDLE.lock().unwrap();
    if handle.is_none() {
        *handle = Some(sch::Scheduler::start(&*SCHEDULER).unwrap());
    }

    SCHEDULER.lock().unwrap().schedule(0, "NSE", devs, f);
}
