use itertools::join;
use paired::bls12_381::Fr;

static SHA256_SRC: &str = include_str!("cl/hash/sha256.cl");
static COMMON_SRC: &str = include_str!("cl/common.cl");
static MASK_SRC: &str = include_str!("cl/mask.cl");
static EXPANDER_SRC: &str = include_str!("cl/expander.cl");
static BUTTERFLY_SRC: &str = include_str!("cl/butterfly.cl");
static COMBINE_SRC: &str = include_str!("cl/combine.cl");

pub fn generate_nse_program() -> String {
    join(
        &[
            ff_cl_gen::field::<Fr>("Fr"),
            SHA256_SRC.to_string(),
            COMMON_SRC.to_string(),
            MASK_SRC.to_string(),
            EXPANDER_SRC.to_string(),
            BUTTERFLY_SRC.to_string(),
            COMBINE_SRC.to_string(),
        ],
        "\n",
    )
}
