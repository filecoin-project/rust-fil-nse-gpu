use ff::PrimeField;

static NSE_SRC: &str = include_str!("cl/nse.cl");

pub fn generate_nse_program<F: PrimeField>() -> String {
    format!("{}\n{}", ff_cl_gen::field::<F>("F"), NSE_SRC.to_string())
}
