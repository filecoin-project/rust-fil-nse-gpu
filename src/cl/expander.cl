#define BYTE_SIZE (BIT_SIZE / 8)
#define STREAM_HASH_COUNT (DEGREE_EXPANDER * BIT_SIZE / sha256_BITS)

// We are going to generate and store the entire bit-stream per node
// before running the expander algorithm. Is this a good idea?
// (bit-stream per node is around ~1KB in size)
typedef struct {
  sha256_domain bit_source[STREAM_HASH_COUNT];
} bit_stream;

bit_stream gen_stream(uint node) {
  bit_stream stream;
  sha256_block data = sha256_ZERO;
  data.vals[0] = node;
  for(uint i = 0; i < STREAM_HASH_COUNT; i++) {
    data.vals[1] = i;
    stream.bit_source[i] = sha256(data);
  }
  return stream;
}

uchar get_byte(bit_stream *stream, uint i) {
  i = (i & 0xfffffffc) + (3 - (i & 3)); // Change endianness
  return ((uchar*)stream)[i];
}

// Get `i`th chunk of bitstream (chunks are `BIT_SIZE` long)
// I.e. get `i`th *non-expanded* parent of node
// Result is in the range `[0, 2^BIT_SIZE)`
uint get_parent(bit_stream *stream, uint i) {
  uint ret = 0;
  for(uint j = 0; j < BYTE_SIZE; j++) {
    uint bt = get_byte(stream, i * BYTE_SIZE + j);
    ret |= (bt << (j * BITS_PER_BYTE));
  }
  return ret;
}

// Returns `i`th *expanded* parent of node
// `i` is in the range `[0, K * EXPANDED_DEGREE)`
uint get_expanded_parent(bit_stream *stream, uint i) {

  // `i`th expanded parent of node is equal with:
  // `i / K`th non-expanded parent of node plus `i % K`
  uint x = i >> LOG2_K; // i / K
  uint offset = i & (K - 1); // i % K

  // Return Parent_x(node) * K + offset
  return get_parent(stream, x) * K + offset;
}

__kernel void generate_expander(__global Fr *input,
                                __global Fr *output,
                                sha256_domain replica_id,
                                uint window_index,
                                uint layer_index) {

  uint node = get_global_id(0); // Nodes are processed in parallel

  bit_stream stream = gen_stream(node); // 1152 Bytes ~ 1KB

  sha256_domain state = sha256_INIT;
  state = sha256_update(state, hash_prefix(layer_index, node, window_index, replica_id));

  for(uint i = 0; i < DEGREE_EXPANDER / 2; i++) {
    uint i_1 = i * 2;
    uint i_2 = i * 2 + 1;

    Fr x_1 = Fr_ZERO;
    Fr x_2 = Fr_ZERO;

    for(uint j = 0; j < K; j++) {
      uint parent_1 = get_expanded_parent(&stream, i_1 + j * DEGREE_EXPANDER);
      uint parent_2 = get_expanded_parent(&stream, i_2 + j * DEGREE_EXPANDER);

      x_1 = Fr_add(x_1, input[parent_1]);
      x_2 = Fr_add(x_2, input[parent_2]);
    }

    state = sha256_update(state, Fr_to_sha256_block(x_1, x_2));
  }

  state = sha256_finish(state, DEGREE_EXPANDER / 2);

  output[node] = sha256_domain_to_Fr(state);
}
