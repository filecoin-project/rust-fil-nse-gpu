#define BITS_PER_BYTE (8)
#define NUM_LAYERS (NUM_EXPANDER_LAYERS + NUM_BUTTERFLY_LAYERS)
#define MODULO_N_MASK (N - 1)

typedef struct {
  uint vals[8];
} replica_id;

__kernel void to_montgomery(__global Fr *buffer) {
  uint node = get_global_id(0);
  buffer[node] = Fr_mont(buffer[node]);
}

__kernel void generate_montgomery(__global Fr *input,
                                  __global Fr *output) {
  uint node = get_global_id(0);
  output[node] = Fr_mont(input[node]);
}

__kernel void generate_ordinary(__global Fr *input,
                                __global Fr *output) {
  uint node = get_global_id(0);
  output[node] = Fr_unmont(input[node]);
}

uint reverse_bytes(uint a) {
  return (a & 0x000000ff) << 24 | (a & 0x0000ff00) << 8 |
         (a & 0x00ff0000) >> 8 | (a & 0xff000000) >> 24;
}

sha256_block hash_prefix(uint layer_index, ulong node_absolute_index, replica_id id) {
  sha256_block data = sha256_ZERO;
  data.vals[0] = layer_index;
  // `node_absolute_index` is encoded in big-endian order (Based on the cpu impl),
  // which means (Based on the assumption that Nvidia/AMD devices are little-endian)
  // we should assume the supplied ulong `node_absolute_index`'s lower part is
  // `(node_absolute_index >> 32)` and higher part is `node_absolute_index & 0xffffffff`
  data.vals[1] = node_absolute_index >> 32;
  data.vals[2] = node_absolute_index;
  for(uint i = 0; i < 8; i++) {
    data.vals[8 + i] = reverse_bytes(id.vals[i]);
  }
  return data;
}

sha256_block Fr_to_sha256_block(Fr a, Fr b) {
  sha256_block data;
  for(uint i = 0; i < Fr_LIMBS; i++) {
    data.vals[2 * i] = reverse_bytes(a.val[i] & 0xffffffff);
    data.vals[2 * i + 1] = reverse_bytes(a.val[i] >> 32);
    data.vals[2 * (i + Fr_LIMBS)] = reverse_bytes(b.val[i] & 0xffffffff);
    data.vals[2 * (i + Fr_LIMBS) + 1] = reverse_bytes(b.val[i] >> 32);
  }
  return data;
}

Fr sha256_domain_to_Fr(sha256_domain state) {
  Fr f;
  for(uint i = 0; i < Fr_LIMBS; i++)
    f.val[i] = ((limb)reverse_bytes(state.vals[2 * i + 1]) << 32) + reverse_bytes(state.vals[2 * i]);
  // Zeroing out last two bits
  f.val[Fr_LIMBS - 1] <<= 2;
  f.val[Fr_LIMBS - 1] >>= 2;
  return f;
}
