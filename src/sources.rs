use super::Config;
use itertools::join;
use paired::bls12_381::Fr;

static SHA256_SRC: &str = include_str!("cl/hash/sha256.cl");
static COMMON_SRC: &str = include_str!("cl/common.cl");
static MASK_SRC: &str = include_str!("cl/mask.cl");
static EXPANDER_SRC: &str = include_str!("cl/expander.cl");
static BUTTERFLY_SRC: &str = include_str!("cl/butterfly.cl");
static COMBINE_SRC: &str = include_str!("cl/combine.cl");

fn config(conf: Config) -> String {
    format!(
        "#define N ({})
         #define K ({})
         #define LOG2_K ({})
         #define DEGREE_EXPANDER ({})
         #define DEGREE_BUTTERFLY ({})
         #define LOG2_DEGREE_BUTTERFLY ({})
         #define NUM_EXPANDER_LAYERS ({})
         #define NUM_BUTTERFLY_LAYERS ({})
         #define BIT_SIZE ({})\n",
        conf.n,
        conf.k,
        (conf.k as f64).log2() as u32,
        conf.degree_expander,
        conf.degree_butterfly,
        (conf.degree_butterfly as f64).log2() as u32,
        conf.num_expander_layers,
        conf.num_butterfly_layers,
        (conf.n as f64 / conf.k as f64).log2() as u32
    )
}

pub fn generate_nse_program(conf: Config) -> String {
    join(
        &[
            config(conf),
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
