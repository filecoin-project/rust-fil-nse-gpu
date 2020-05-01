use itertools::join;
use paired::bls12_381::Fr;

#[allow(dead_code)]
static NSE_SRC: &str = include_str!("cl/nse.cl");
#[allow(dead_code)]
static SHA256_SRC: &str = include_str!("cl/hash/sha256.cl");

#[allow(dead_code)]
pub fn generate_nse_program() -> String {
    join(
        &[
            ff_cl_gen::field::<Fr>("Fr"),
            SHA256_SRC.to_string(),
            NSE_SRC.to_string(),
        ],
        "\n",
    )
}
