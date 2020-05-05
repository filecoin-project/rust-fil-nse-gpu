typedef struct {
  uint vals[8];
} sha256_state;

typedef struct {
  uint vals[16];
} sha256_data;

#define sha256_ZERO ((sha256_data){{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}})
#define sha256_INIT ((sha256_state){{0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19}})

__constant uint SHA256_K[64] =
{
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
    0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
    0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
    0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
    0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
    0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

#define ChI(x, y, z) ( z ^ (x & ( y ^ z)) )
#define MajI(x, y, z) ( (x & y) | (z & (x | y)) )

#define S0I(x) ((uint)rotate((uint)x,(uint)30) ^ (uint)rotate((uint)x,(uint)19) ^ (uint)rotate((uint)x,(uint)10))
#define S1I(x) ((uint)rotate((uint)x,(uint)26) ^ (uint)rotate((uint)x,(uint)21) ^ (uint)rotate((uint)x,(uint)7))
#define s0I(x) ((uint)rotate((uint)x,(uint)25) ^ (uint)rotate((uint)x,(uint)14) ^ (x>>3))
#define s1I(x) ((uint)rotate((uint)x,(uint)15) ^ (uint)rotate((uint)x,(uint)13) ^ (x>>10))

sha256_state sha256_update(sha256_state state, sha256_data data)
{
	uint W00,W01,W02,W03,W04,W05,W06,W07;
	uint W08,W09,W10,W11,W12,W13,W14,W15;
	uint T0,T1,T2,T3,T4,T5,T6,T7;

	T0 = state.vals[0]; T1 = state.vals[1];
	T2 = state.vals[2]; T3 = state.vals[3];
	T4 = state.vals[4]; T5 = state.vals[5];
	T6 = state.vals[6]; T7 = state.vals[7];

	T7 += S1I( T4 ) + ChI( T4, T5, T6 ) + SHA256_K[0] + ( (W00 = data.vals[0]) );
	T3 += T7;
	T7 += S0I( T0 ) + MajI( T0, T1, T2 );

	T6 += S1I( T3 ) + ChI( T3, T4, T5 ) + SHA256_K[1] + ( (W01 = data.vals[1]) );
	T2 += T6;
	T6 += S0I( T7 ) + MajI( T7, T0, T1 );

	T5 += S1I( T2 ) + ChI( T2, T3, T4 ) + SHA256_K[2] + ( (W02 = data.vals[2]) );
	T1 += T5;
	T5 += S0I( T6 ) + MajI( T6, T7, T0 );

	T4 += S1I( T1 ) + ChI( T1, T2, T3 ) + SHA256_K[3] + ( (W03 = data.vals[3]) );
	T0 += T4;
	T4 += S0I( T5 ) + MajI( T5, T6, T7 );

	T3 += S1I( T0 ) + ChI( T0, T1, T2 ) + SHA256_K[4] + ( (W04 = data.vals[4]) );
	T7 += T3;
	T3 += S0I( T4 ) + MajI( T4, T5, T6 );

	T2 += S1I( T7 ) + ChI( T7, T0, T1 ) + SHA256_K[5] + ( (W05 = data.vals[5]) );
	T6 += T2;
	T2 += S0I( T3 ) + MajI( T3, T4, T5 );

	T1 += S1I( T6 ) + ChI( T6, T7, T0 ) + SHA256_K[6] + ( (W06 = data.vals[6]) );
	T5 += T1;
	T1 += S0I( T2 ) + MajI( T2, T3, T4 );

	T0 += S1I( T5 ) + ChI( T5, T6, T7 ) + SHA256_K[7] + ( (W07 = data.vals[7]) );
	T4 += T0;
	T0 += S0I( T1 ) + MajI( T1, T2, T3 );

	T7 += S1I( T4 ) + ChI( T4, T5, T6 ) + SHA256_K[8] + ( (W08 = data.vals[8]) );
	T3 += T7;
	T7 += S0I( T0 ) + MajI( T0, T1, T2 );

	T6 += S1I( T3 ) + ChI( T3, T4, T5 ) + SHA256_K[9] + ( (W09 = data.vals[9]) );
	T2 += T6;
	T6 += S0I( T7 ) + MajI( T7, T0, T1 );

	T5 += S1I( T2 ) + ChI( T2, T3, T4 ) + SHA256_K[10] + ( (W10 = data.vals[10]) );
	T1 += T5;
	T5 += S0I( T6 ) + MajI( T6, T7, T0 );

	T4 += S1I( T1 ) + ChI( T1, T2, T3 ) + SHA256_K[11] + ( (W11 = data.vals[11]) );
	T0 += T4;
	T4 += S0I( T5 ) + MajI( T5, T6, T7 );

	T3 += S1I( T0 ) + ChI( T0, T1, T2 ) + SHA256_K[12] + ( (W12 = data.vals[12]) );
	T7 += T3;
	T3 += S0I( T4 ) + MajI( T4, T5, T6 );

	T2 += S1I( T7 ) + ChI( T7, T0, T1 ) + SHA256_K[13] + ( (W13 = data.vals[13]) );
	T6 += T2;
	T2 += S0I( T3 ) + MajI( T3, T4, T5 );

	T1 += S1I( T6 ) + ChI( T6, T7, T0 ) + SHA256_K[14] + ( (W14 = data.vals[14]) );
	T5 += T1;
	T1 += S0I( T2 ) + MajI( T2, T3, T4 );

	T0 += S1I( T5 ) + ChI( T5, T6, T7 ) + SHA256_K[15] + ( (W15 = data.vals[15]) );
	T4 += T0;
	T0 += S0I( T1 ) + MajI( T1, T2, T3 );



	T7 += S1I( T4 ) + ChI( T4, T5, T6 ) + SHA256_K[16] + ( (W00 += s1I( W14 ) + W09 + s0I( W01 ) ) );
	T3 += T7;
	T7 += S0I( T0 ) + MajI( T0, T1, T2 );

	T6 += S1I( T3 ) + ChI( T3, T4, T5 ) + SHA256_K[17] + ( (W01 += s1I( W15 ) + W10 + s0I( W02 ) ) );
	T2 += T6;
	T6 += S0I( T7 ) + MajI( T7, T0, T1 );

	T5 += S1I( T2 ) + ChI( T2, T3, T4 ) + SHA256_K[18] + ( (W02 += s1I( W00 ) + W11 + s0I( W03 ) ) );
	T1 += T5;
	T5 += S0I( T6 ) + MajI( T6, T7, T0 );

	T4 += S1I( T1 ) + ChI( T1, T2, T3 ) + SHA256_K[19] + ( (W03 += s1I( W01 ) + W12 + s0I( W04 ) ) );
	T0 += T4;
	T4 += S0I( T5 ) + MajI( T5, T6, T7 );

	T3 += S1I( T0 ) + ChI( T0, T1, T2 ) + SHA256_K[20] + ( (W04 += s1I( W02 ) + W13 + s0I( W05 ) ) );
	T7 += T3;
	T3 += S0I( T4 ) + MajI( T4, T5, T6 );

	T2 += S1I( T7 ) + ChI( T7, T0, T1 ) + SHA256_K[21] + ( (W05 += s1I( W03 ) + W14 + s0I( W06 ) ) );
	T6 += T2;
	T2 += S0I( T3 ) + MajI( T3, T4, T5 );

	T1 += S1I( T6 ) + ChI( T6, T7, T0 ) + SHA256_K[22] + ( (W06 += s1I( W04 ) + W15 + s0I( W07 ) ) );
	T5 += T1;
	T1 += S0I( T2 ) + MajI( T2, T3, T4 );

	T0 += S1I( T5 ) + ChI( T5, T6, T7 ) + SHA256_K[23] + ( (W07 += s1I( W05 ) + W00 + s0I( W08 ) ) );
	T4 += T0;
	T0 += S0I( T1 ) + MajI( T1, T2, T3 );

	T7 += S1I( T4 ) + ChI( T4, T5, T6 ) + SHA256_K[24] + ( (W08 += s1I( W06 ) + W01 + s0I( W09 ) ) );
	T3 += T7;
	T7 += S0I( T0 ) + MajI( T0, T1, T2 );

	T6 += S1I( T3 ) + ChI( T3, T4, T5 ) + SHA256_K[25] + ( (W09 += s1I( W07 ) + W02 + s0I( W10 ) ) );
	T2 += T6;
	T6 += S0I( T7 ) + MajI( T7, T0, T1 );

	T5 += S1I( T2 ) + ChI( T2, T3, T4 ) + SHA256_K[26] + ( (W10 += s1I( W08 ) + W03 + s0I( W11 ) ) );
	T1 += T5;
	T5 += S0I( T6 ) + MajI( T6, T7, T0 );

	T4 += S1I( T1 ) + ChI( T1, T2, T3 ) + SHA256_K[27] + ( (W11 += s1I( W09 ) + W04 + s0I( W12 ) ) );
	T0 += T4;
	T4 += S0I( T5 ) + MajI( T5, T6, T7 );

	T3 += S1I( T0 ) + ChI( T0, T1, T2 ) + SHA256_K[28] + ( (W12 += s1I( W10 ) + W05 + s0I( W13 ) ) );
	T7 += T3;
	T3 += S0I( T4 ) + MajI( T4, T5, T6 );

	T2 += S1I( T7 ) + ChI( T7, T0, T1 ) + SHA256_K[29] + ( (W13 += s1I( W11 ) + W06 + s0I( W14 ) ) );
	T6 += T2;
	T2 += S0I( T3 ) + MajI( T3, T4, T5 );

	T1 += S1I( T6 ) + ChI( T6, T7, T0 ) + SHA256_K[30] + ( (W14 += s1I( W12 ) + W07 + s0I( W15 ) ) );
	T5 += T1;
	T1 += S0I( T2 ) + MajI( T2, T3, T4 );

	T0 += S1I( T5 ) + ChI( T5, T6, T7 ) + SHA256_K[31] + ( (W15 += s1I( W13 ) + W08 + s0I( W00 ) ) );
	T4 += T0;
	T0 += S0I( T1 ) + MajI( T1, T2, T3 );




	T7 += S1I( T4 ) + ChI( T4, T5, T6 ) + SHA256_K[32] + ( (W00 += s1I( W14 ) + W09 + s0I( W01 ) ) );
	T3 += T7;
	T7 += S0I( T0 ) + MajI( T0, T1, T2 );

	T6 += S1I( T3 ) + ChI( T3, T4, T5 ) + SHA256_K[33] + ( (W01 += s1I( W15 ) + W10 + s0I( W02 ) ) );
	T2 += T6;
	T6 += S0I( T7 ) + MajI( T7, T0, T1 );

	T5 += S1I( T2 ) + ChI( T2, T3, T4 ) + SHA256_K[34] + ( (W02 += s1I( W00 ) + W11 + s0I( W03 ) ) );
	T1 += T5;
	T5 += S0I( T6 ) + MajI( T6, T7, T0 );

	T4 += S1I( T1 ) + ChI( T1, T2, T3 ) + SHA256_K[35] + ( (W03 += s1I( W01 ) + W12 + s0I( W04 ) ) );
	T0 += T4;
	T4 += S0I( T5 ) + MajI( T5, T6, T7 );

	T3 += S1I( T0 ) + ChI( T0, T1, T2 ) + SHA256_K[36] + ( (W04 += s1I( W02 ) + W13 + s0I( W05 ) ) );
	T7 += T3;
	T3 += S0I( T4 ) + MajI( T4, T5, T6 );

	T2 += S1I( T7 ) + ChI( T7, T0, T1 ) + SHA256_K[37] + ( (W05 += s1I( W03 ) + W14 + s0I( W06 ) ) );
	T6 += T2;
	T2 += S0I( T3 ) + MajI( T3, T4, T5 );

	T1 += S1I( T6 ) + ChI( T6, T7, T0 ) + SHA256_K[38] + ( (W06 += s1I( W04 ) + W15 + s0I( W07 ) ) );
	T5 += T1;
	T1 += S0I( T2 ) + MajI( T2, T3, T4 );

	T0 += S1I( T5 ) + ChI( T5, T6, T7 ) + SHA256_K[39] + ( (W07 += s1I( W05 ) + W00 + s0I( W08 ) ) );
	T4 += T0;
	T0 += S0I( T1 ) + MajI( T1, T2, T3 );

	T7 += S1I( T4 ) + ChI( T4, T5, T6 ) + SHA256_K[40] + ( (W08 += s1I( W06 ) + W01 + s0I( W09 ) ) );
	T3 += T7;
	T7 += S0I( T0 ) + MajI( T0, T1, T2 );

	T6 += S1I( T3 ) + ChI( T3, T4, T5 ) + SHA256_K[41] + ( (W09 += s1I( W07 ) + W02 + s0I( W10 ) ) );
	T2 += T6;
	T6 += S0I( T7 ) + MajI( T7, T0, T1 );

	T5 += S1I( T2 ) + ChI( T2, T3, T4 ) + SHA256_K[42] + ( (W10 += s1I( W08 ) + W03 + s0I( W11 ) ) );
	T1 += T5;
	T5 += S0I( T6 ) + MajI( T6, T7, T0 );

	T4 += S1I( T1 ) + ChI( T1, T2, T3 ) + SHA256_K[43] + ( (W11 += s1I( W09 ) + W04 + s0I( W12 ) ) );
	T0 += T4;
	T4 += S0I( T5 ) + MajI( T5, T6, T7 );

	T3 += S1I( T0 ) + ChI( T0, T1, T2 ) + SHA256_K[44] + ( (W12 += s1I( W10 ) + W05 + s0I( W13 ) ) );
	T7 += T3;
	T3 += S0I( T4 ) + MajI( T4, T5, T6 );

	T2 += S1I( T7 ) + ChI( T7, T0, T1 ) + SHA256_K[45] + ( (W13 += s1I( W11 ) + W06 + s0I( W14 ) ) );
	T6 += T2;
	T2 += S0I( T3 ) + MajI( T3, T4, T5 );

	T1 += S1I( T6 ) + ChI( T6, T7, T0 ) + SHA256_K[46] + ( (W14 += s1I( W12 ) + W07 + s0I( W15 ) ) );
	T5 += T1;
	T1 += S0I( T2 ) + MajI( T2, T3, T4 );

	T0 += S1I( T5 ) + ChI( T5, T6, T7 ) + SHA256_K[47] + ( (W15 += s1I( W13 ) + W08 + s0I( W00 ) ) );
	T4 += T0;
	T0 += S0I( T1 ) + MajI( T1, T2, T3 );




	T7 += S1I( T4 ) + ChI( T4, T5, T6 ) + SHA256_K[48] + ( (W00 += s1I( W14 ) + W09 + s0I( W01 ) ) );
	T3 += T7;
	T7 += S0I( T0 ) + MajI( T0, T1, T2 );

	T6 += S1I( T3 ) + ChI( T3, T4, T5 ) + SHA256_K[49] + ( (W01 += s1I( W15 ) + W10 + s0I( W02 ) ) );
	T2 += T6;
	T6 += S0I( T7 ) + MajI( T7, T0, T1 );

	T5 += S1I( T2 ) + ChI( T2, T3, T4 ) + SHA256_K[50] + ( (W02 += s1I( W00 ) + W11 + s0I( W03 ) ) );
	T1 += T5;
	T5 += S0I( T6 ) + MajI( T6, T7, T0 );

	T4 += S1I( T1 ) + ChI( T1, T2, T3 ) + SHA256_K[51] + ( (W03 += s1I( W01 ) + W12 + s0I( W04 ) ) );
	T0 += T4;
	T4 += S0I( T5 ) + MajI( T5, T6, T7 );

	T3 += S1I( T0 ) + ChI( T0, T1, T2 ) + SHA256_K[52] + ( (W04 += s1I( W02 ) + W13 + s0I( W05 ) ) );
	T7 += T3;
	T3 += S0I( T4 ) + MajI( T4, T5, T6 );

	T2 += S1I( T7 ) + ChI( T7, T0, T1 ) + SHA256_K[53] + ( (W05 += s1I( W03 ) + W14 + s0I( W06 ) ) );
	T6 += T2;
	T2 += S0I( T3 ) + MajI( T3, T4, T5 );

	T1 += S1I( T6 ) + ChI( T6, T7, T0 ) + SHA256_K[54] + ( (W06 += s1I( W04 ) + W15 + s0I( W07 ) ) );
	T5 += T1;
	T1 += S0I( T2 ) + MajI( T2, T3, T4 );

	T0 += S1I( T5 ) + ChI( T5, T6, T7 ) + SHA256_K[55] + ( (W07 += s1I( W05 ) + W00 + s0I( W08 ) ) );
	T4 += T0;
	T0 += S0I( T1 ) + MajI( T1, T2, T3 );

	T7 += S1I( T4 ) + ChI( T4, T5, T6 ) + SHA256_K[56] + ( (W08 += s1I( W06 ) + W01 + s0I( W09 ) ) );
	T3 += T7;
	T7 += S0I( T0 ) + MajI( T0, T1, T2 );

	T6 += S1I( T3 ) + ChI( T3, T4, T5 ) + SHA256_K[57] + ( (W09 += s1I( W07 ) + W02 + s0I( W10 ) ) );
	T2 += T6;
	T6 += S0I( T7 ) + MajI( T7, T0, T1 );

	T5 += S1I( T2 ) + ChI( T2, T3, T4 ) + SHA256_K[58] + ( (W10 += s1I( W08 ) + W03 + s0I( W11 ) ) );
	T1 += T5;
	T5 += S0I( T6 ) + MajI( T6, T7, T0 );

	T4 += S1I( T1 ) + ChI( T1, T2, T3 ) + SHA256_K[59] + ( (W11 += s1I( W09 ) + W04 + s0I( W12 ) ) );
	T0 += T4;
	T4 += S0I( T5 ) + MajI( T5, T6, T7 );

	T3 += S1I( T0 ) + ChI( T0, T1, T2 ) + SHA256_K[60] + ( (W12 += s1I( W10 ) + W05 + s0I( W13 ) ) );
	T7 += T3;
	T3 += S0I( T4 ) + MajI( T4, T5, T6 );

	T2 += S1I( T7 ) + ChI( T7, T0, T1 ) + SHA256_K[61] + ( (W13 += s1I( W11 ) + W06 + s0I( W14 ) ) );
	T6 += T2;
	T2 += S0I( T3 ) + MajI( T3, T4, T5 );

	T1 += S1I( T6 ) + ChI( T6, T7, T0 ) + SHA256_K[62] + ( (W14 += s1I( W12 ) + W07 + s0I( W15 ) ) );
	T5 += T1;
	T1 += S0I( T2 ) + MajI( T2, T3, T4 );

	T0 += S1I( T5 ) + ChI( T5, T6, T7 ) + SHA256_K[63] + ( (W15 += s1I( W13 ) + W08 + s0I( W00 ) ) );
	T4 += T0;
	T0 += S0I( T1 ) + MajI( T1, T2, T3 );

	state.vals[0] += T0;
	state.vals[1] += T1;
  state.vals[2] += T2;
  state.vals[3] += T3;
  state.vals[4] += T4;
  state.vals[5] += T5;
  state.vals[6] += T6;
  state.vals[7] += T7;

  return state;
}

sha256_state sha256(sha256_data data) {
  sha256_data padding = sha256_ZERO;
  padding.vals[0] = 0x80000000;
  padding.vals[15] = 512;
  return sha256_update(sha256_update(sha256_INIT, data), padding);
}

__kernel void sha256_test(__global uint *data, __global uint *digest) {
  sha256_data dat = sha256_ZERO;
  for(uint i = 0; i < 16; i++)
    dat.vals[i] = data[i];
  sha256_state s = sha256(dat);
  for(uint i = 0; i < 8; i++)
    digest[i] = s.vals[i];
}
