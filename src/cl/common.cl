#define DEGREE_EXPANDER (384)
#define K (8)
#define BIT_SIZE (24)
#define BITS_PER_BYTE (8)
#define BYTE_SIZE (BIT_SIZE / BITS_PER_BYTE)
#define HASH_SIZE (256)
#define STREAM_HASH_COUNT (DEGREE_EXPANDER * BIT_SIZE / HASH_SIZE) // 36

// We are going to generate and store the entire bit-stream per node
// before running the expander algorithm. Is this a good idea?
// (bit-stream per node is around ~1KB in size)
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

uchar get_byte(bit_stream *stream, uint i) {
  uint index = (i / 4) * 4 + (4 - 1 - (i % 4)); // Change endianness
  return ((uchar*)stream)[index];
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
  uint x = i / K;
  uint offset = i % K; // Or `i - K * x` if faster

  // Return Parent_x(node) * K + offset
  return get_parent(stream, x) * K + offset;
}
