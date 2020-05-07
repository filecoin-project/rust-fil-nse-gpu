#define BITS_PER_BYTE (8)
#define NUM_LAYERS (NUM_EXPANDER_LAYERS + NUM_BUTTERFLY_LAYERS)
#define MODULO_N_MASK (N - 1)

uint reverse_bytes(uint a) {
  return (a & 0x000000ff) << 24 | (a & 0x0000ff00) << 8 |
         (a & 0x00ff0000) >> 8 | (a & 0xff000000) >> 24;
}

sha256_block hash_prefix(uint layer_index, uint node_index, uint window_index, sha256_domain replica_id) {
  sha256_block data = sha256_ZERO;
  data.vals[0] = layer_index;
  data.vals[1] = node_index;
  data.vals[2] = window_index;
  for(uint i = 0; i < 8; i++) {
    data.vals[8 + i] = replica_id.vals[i];
  }
  return data;
}

sha256_block Fr_to_sha256_block(Fr a, Fr b) {
  sha256_block data;
  a = Fr_unmont(a);
  b = Fr_unmont(b);
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
  f = Fr_mont(f);
  return f;
}
