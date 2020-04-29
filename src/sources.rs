use ff::PrimeField;
use itertools::join;

static NSE_SRC: &str = include_str!("cl/nse.cl");
static SHA256_SRC: &str = include_str!("cl/hash/sha256.cl");

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
