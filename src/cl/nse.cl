#define DEGREE_EXPANDER (384)
#define K (8)
#define BIT_SIZE (24)
#define HASH_SIZE (256)
#define STREAM_HASH_COUNT (DEGREE_EXPANDER * BIT_SIZE / HASH_SIZE) // 36

// We are going to generate and store the entire bit-stream per node
// before running the expander algorithm. Is this a good idea?
// (bit-stream per node is  around ~1KB in size)
typedef struct {
  sha256_state bit_source[STREAM_HASH_COUNT];
} bit_stream;

bit_stream gen_stream(uint node) {
  bit_stream stream;
  sha256_data data = sha256_ZERO;
  data.vals[0] = node;
  for(uint i = 0; i < STREAM_HASH_COUNT; i++) {
    data.vals[1] = i;
    stream.bit_source[i] = sha256(data);
  }
  return stream;
}

// Get `i`th chunk of bitstream (chunks are `BIT_SIZE` long)
// Result is in the range `[0, 2^BIT_SIZE)`
uint get_chunk(bit_stream *stream, uint i) {
  // TODO: Return `stream[i * BIT_SIZE..(i + 1) * BIT_SIZE]`
  return 0;
}

// Returns `i`th *expanded* parent of node
// `i` is in the range `[0, K * EXPANDED_DEGREE)`
uint parent(bit_stream *stream, uint node, uint i) {

  // `i`th expanded parent of node is equal with:
  // `i / K`th non-expanded parent of node plus `i % K`
  uint x = i / K;
  uint offset = i % K; // Or `i - K * x` if faster

  // Return Parent_x(node) * K + offset
  return get_chunk(stream, x) * K + offset;
}

FIELD cast_to_field(sha256_state data) {
  // TODO: Implement this
  return FIELD_ZERO;
}

sha256_data cast_to_data(FIELD field) {
  // TODO: Implement this
  return sha256_ZERO;
}

// `current` and `next` layers consists of `n` Fr elements
// Fr elements are 32bit long, and N is 4Gib / 32
// Which means size of a layer is 4Gib
__kernel void expander(__global FIELD *current,
                       __global FIELD *next,
                       uint n) {
  uint node = get_global_id(0); // Nodes are processed in parallel
  bit_stream stream = gen_stream(node); // 1152 Bytes ~ 1KB

  sha256_state digest = sha256_INIT;
  for(uint i = 0; i < DEGREE_EXPANDER; i++) {
    FIELD x_i = FIELD_ZERO;
    for(uint j = 0; j < K; j++) {
      x_i = FIELD_add(x_i, current[parent(&stream,
                                       node,
                                       i + j * DEGREE_EXPANDER)]);
    }
    digest = sha256_update(digest, cast_to_data(x_i));
  }

  next[node] = cast_to_field(digest);
}
