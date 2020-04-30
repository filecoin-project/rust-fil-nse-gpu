use ff::PrimeField;
use itertools::join;

#[allow(dead_code)]
static NSE_SRC: &str = include_str!("cl/nse.cl");
#[allow(dead_code)]
static SHA256_SRC: &str = include_str!("cl/hash/sha256.cl");

#[allow(dead_code)]
pub fn generate_nse_program<F: PrimeField>() -> String {
    const FIELD_NAME: &str = "Field";
    join(
        &[
            ff_cl_gen::field::<F>(FIELD_NAME),
            SHA256_SRC.to_string(),
            NSE_SRC.to_string(),
        ],
        "\n",
    )
}
