// perlin_noise.glsl - Shared noise functions for terrain generation
// Minecraft 1.8.9 compatible noise implementation
//
// Usage: #include "perlin_noise.glsl"
// Requires: Permutation table buffer bound at binding 3 (configurable via PERM_BINDING)

#ifndef PERLIN_NOISE_GLSL
#define PERLIN_NOISE_GLSL

// Default permutation table binding - override before including if needed
#ifndef PERM_BINDING
#define PERM_BINDING 3
#endif

// Minecraft 1.8.9 permutation table (Ken Perlin's original)
// This exact sequence ensures seed compatibility with Java Minecraft
const int PERM[256] = int[256](
    151, 160, 137,  91,  90,  15, 131,  13, 201,  95,  96,  53, 194, 233,   7, 225,
    140,  36, 103,  30,  69, 142,   8,  99,  37, 240,  21,  10,  23, 190,   6, 148,
    247, 120, 234,  75,   0,  26, 197,  62,  94, 252, 219, 203, 117,  35,  11,  32,
     57, 177,  33,  88, 237, 149,  56,  87, 174,  20, 125, 136, 171, 168,  68, 175,
     74, 165,  71, 134, 139,  48,  27, 166,  77, 146, 158, 231,  83, 111, 229, 122,
     60, 211, 133, 230, 220, 105,  92,  41,  55,  46, 245,  40, 244, 102, 143,  54,
     65,  25,  63, 161,   1, 216,  80,  73, 209,  76, 132, 187, 208,  89,  18, 169,
    200, 196, 135, 130, 116, 188, 159,  86, 164, 100, 109, 198, 173, 186,   3,  64,
     52, 217, 226, 250, 124, 123,   5, 202,  38, 147, 118, 126, 255,  82,  85, 212,
    207, 206,  59, 227,  47,  16,  58,  17, 182, 189,  28,  42, 223, 183, 170, 213,
    119, 248, 152,   2,  44, 154, 163,  70, 221, 153, 101, 155, 167,  43, 172,   9,
    129,  22,  39, 253,  19,  98, 108, 110,  79, 113, 224, 232, 178, 185, 112, 104,
    218, 246,  97, 228, 251,  34, 242, 193, 238, 210, 144,  12, 191, 179, 162, 241,
     81,  51, 145, 235, 249,  14, 239, 107,  49, 192, 214,  31, 181, 199, 106, 157,
    184,  84, 204, 176, 115, 121,  50,  45, 127,   4, 150, 254, 138, 236, 205,  93,
    222, 114,  67,  29,  24,  72, 243, 141, 128, 195,  78,  66, 215,  61, 156, 180
);

// 3D gradient vectors (12 directions) - standard Perlin gradients
const vec3 GRAD3[12] = vec3[12](
    vec3( 1,  1,  0), vec3(-1,  1,  0), vec3( 1, -1,  0), vec3(-1, -1,  0),
    vec3( 1,  0,  1), vec3(-1,  0,  1), vec3( 1,  0, -1), vec3(-1,  0, -1),
    vec3( 0,  1,  1), vec3( 0, -1,  1), vec3( 0,  1, -1), vec3( 0, -1, -1)
);

// 2D gradient vectors
const vec2 GRAD2[8] = vec2[8](
    vec2( 1,  0), vec2(-1,  0), vec2( 0,  1), vec2( 0, -1),
    vec2( 1,  1), vec2(-1,  1), vec2( 1, -1), vec2(-1, -1)
);

// Simplex skew factors
const float F2 = 0.5 * (sqrt(3.0) - 1.0);  // 0.366025403784
const float G2 = (3.0 - sqrt(3.0)) / 6.0;  // 0.211324865405
const float F3 = 1.0 / 3.0;
const float G3 = 1.0 / 6.0;

// ============================================================================
// Helper functions
// ============================================================================

// Permutation lookup with seed offset
int perm_lookup(int idx, float seed) {
    int offset = int(seed * 256.0) & 255;
    return PERM[(idx + offset) & 255];
}

// Double permutation for better seed mixing
int perm2(int x, int y, float seed) {
    return perm_lookup(perm_lookup(x, seed) + y, seed);
}

// Triple permutation
int perm3(int x, int y, int z, float seed) {
    return perm_lookup(perm2(x, y, seed) + z, seed);
}

// Quintic interpolation curve (smoother than cubic, no second derivative discontinuity)
float fade(float t) {
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
}

// Gradient function for 2D
float grad2(int hash, float x, float y) {
    int h = hash & 7;
    return dot(GRAD2[h], vec2(x, y));
}

// Gradient function for 3D using gradient vectors
float grad3(int hash, float x, float y, float z) {
    return dot(GRAD3[hash % 12], vec3(x, y, z));
}

// Classic gradient function (Ken Perlin's original)
float grad_classic(int hash, float x, float y, float z) {
    int h = hash & 15;
    float u = h < 8 ? x : y;
    float v = h < 4 ? y : (h == 12 || h == 14 ? x : z);
    return ((h & 1) == 0 ? u : -u) + ((h & 2) == 0 ? v : -v);
}

// Hash function for simplex noise (fast integer hash)
int hash_simplex(int x, int y, float seed) {
    int n = x + y * 57 + int(seed * 131.0);
    n = (n << 13) ^ n;
    return (n * (n * n * 15731 + 789221) + 1376312589) & 0x7fffffff;
}

// ============================================================================
// 2D Perlin Noise
// ============================================================================

// Standard 2D Perlin noise with seed
// Returns value in range [-1, 1]
float perlin_2d(vec2 p, float seed) {
    int X = int(floor(p.x)) & 255;
    int Y = int(floor(p.y)) & 255;

    float x = fract(p.x);
    float y = fract(p.y);

    float u = fade(x);
    float v = fade(y);

    int A  = perm_lookup(X, seed) + Y;
    int B  = perm_lookup(X + 1, seed) + Y;
    int AA = perm_lookup(A, seed);
    int AB = perm_lookup(A + 1, seed);
    int BA = perm_lookup(B, seed);
    int BB = perm_lookup(B + 1, seed);

    float n00 = grad2(AA, x,     y);
    float n10 = grad2(BA, x - 1, y);
    float n01 = grad2(AB, x,     y - 1);
    float n11 = grad2(BB, x - 1, y - 1);

    float nx0 = mix(n00, n10, u);
    float nx1 = mix(n01, n11, u);

    return mix(nx0, nx1, v);
}

// ============================================================================
// 3D Perlin Noise
// ============================================================================

// Standard 3D Perlin noise with seed
// Returns value in range [-1, 1]
float perlin_3d(vec3 p, float seed) {
    int X = int(floor(p.x)) & 255;
    int Y = int(floor(p.y)) & 255;
    int Z = int(floor(p.z)) & 255;

    float x = fract(p.x);
    float y = fract(p.y);
    float z = fract(p.z);

    float u = fade(x);
    float v = fade(y);
    float w = fade(z);

    int A  = perm_lookup(X, seed) + Y;
    int AA = perm_lookup(A, seed) + Z;
    int AB = perm_lookup(A + 1, seed) + Z;
    int B  = perm_lookup(X + 1, seed) + Y;
    int BA = perm_lookup(B, seed) + Z;
    int BB = perm_lookup(B + 1, seed) + Z;

    int AAA = perm_lookup(AA, seed);
    int BAA = perm_lookup(BA, seed);
    int ABA = perm_lookup(AB, seed);
    int BBA = perm_lookup(BB, seed);
    int AAB = perm_lookup(AA + 1, seed);
    int BAB = perm_lookup(BA + 1, seed);
    int ABB = perm_lookup(AB + 1, seed);
    int BBB = perm_lookup(BB + 1, seed);

    float n000 = grad3(AAA, x,     y,     z);
    float n100 = grad3(BAA, x - 1, y,     z);
    float n010 = grad3(ABA, x,     y - 1, z);
    float n110 = grad3(BBA, x - 1, y - 1, z);
    float n001 = grad3(AAB, x,     y,     z - 1);
    float n101 = grad3(BAB, x - 1, y,     z - 1);
    float n011 = grad3(ABB, x,     y - 1, z - 1);
    float n111 = grad3(BBB, x - 1, y - 1, z - 1);

    float nx00 = mix(n000, n100, u);
    float nx10 = mix(n010, n110, u);
    float nx01 = mix(n001, n101, u);
    float nx11 = mix(n011, n111, u);

    float nxy0 = mix(nx00, nx10, v);
    float nxy1 = mix(nx01, nx11, v);

    return mix(nxy0, nxy1, w);
}

// ============================================================================
// Octave (FBM) Noise
// ============================================================================

// 2D octave noise (fractional Brownian motion)
// persistence: amplitude multiplier per octave (typically 0.5)
// Returns value in range [-1, 1]
float octave_noise_2d(vec2 p, int octaves, float persistence, float seed) {
    float total = 0.0;
    float amplitude = 1.0;
    float frequency = 1.0;
    float maxValue = 0.0;

    for (int i = 0; i < octaves; i++) {
        total += perlin_2d(p * frequency, seed + float(i) * 100.0) * amplitude;
        maxValue += amplitude;
        amplitude *= persistence;
        frequency *= 2.0;
    }

    return total / maxValue;
}

// 3D octave noise (fractional Brownian motion)
// persistence: amplitude multiplier per octave (typically 0.5)
// Returns value in range [-1, 1]
float octave_noise_3d(vec3 p, int octaves, float persistence, float seed) {
    float total = 0.0;
    float amplitude = 1.0;
    float frequency = 1.0;
    float maxValue = 0.0;

    for (int i = 0; i < octaves; i++) {
        total += perlin_3d(p * frequency, seed + float(i) * 100.0) * amplitude;
        maxValue += amplitude;
        amplitude *= persistence;
        frequency *= 2.0;
    }

    return total / maxValue;
}

// ============================================================================
// 2D Simplex Noise (faster alternative to Perlin)
// ============================================================================

// 2D Simplex noise with seed
// Returns value in range [-1, 1]
float simplex_2d(vec2 p, float seed) {
    // Skew input space to determine which simplex cell we're in
    float s = (p.x + p.y) * F2;
    int i = int(floor(p.x + s));
    int j = int(floor(p.y + s));

    // Unskew to get cell origin
    float t = float(i + j) * G2;
    vec2 origin = vec2(float(i) - t, float(j) - t);
    vec2 d0 = p - origin;

    // Determine which simplex we're in (upper or lower triangle)
    int i1, j1;
    if (d0.x > d0.y) {
        i1 = 1; j1 = 0;  // Lower triangle
    } else {
        i1 = 0; j1 = 1;  // Upper triangle
    }

    // Offsets for middle and last corner
    vec2 d1 = d0 - vec2(float(i1), float(j1)) + G2;
    vec2 d2 = d0 - 1.0 + 2.0 * G2;

    // Hash coordinates for gradient selection
    int ii = i & 255;
    int jj = j & 255;

    int gi0 = perm2(ii,      jj,      seed) % 8;
    int gi1 = perm2(ii + i1, jj + j1, seed) % 8;
    int gi2 = perm2(ii + 1,  jj + 1,  seed) % 8;

    // Calculate contribution from three corners
    float n0 = 0.0, n1 = 0.0, n2 = 0.0;

    float t0 = 0.5 - d0.x * d0.x - d0.y * d0.y;
    if (t0 >= 0.0) {
        t0 *= t0;
        n0 = t0 * t0 * dot(GRAD2[gi0], d0);
    }

    float t1 = 0.5 - d1.x * d1.x - d1.y * d1.y;
    if (t1 >= 0.0) {
        t1 *= t1;
        n1 = t1 * t1 * dot(GRAD2[gi1], d1);
    }

    float t2 = 0.5 - d2.x * d2.x - d2.y * d2.y;
    if (t2 >= 0.0) {
        t2 *= t2;
        n2 = t2 * t2 * dot(GRAD2[gi2], d2);
    }

    // Scale to [-1, 1]
    return 70.0 * (n0 + n1 + n2);
}

// ============================================================================
// 2D Worley (Cellular) Noise - for caves
// ============================================================================

// 2D Worley noise with seed
// Returns distance to nearest feature point (value in range [0, 1])
float worley_2d(vec2 p, float seed) {
    vec2 cell = floor(p);
    vec2 frac = fract(p);

    float minDist = 1.0;

    // Check 3x3 neighborhood of cells
    for (int j = -1; j <= 1; j++) {
        for (int i = -1; i <= 1; i++) {
            vec2 neighbor = vec2(float(i), float(j));
            vec2 cellCoord = cell + neighbor;

            // Hash to get feature point position within cell
            int h = perm2(int(cellCoord.x) & 255, int(cellCoord.y) & 255, seed);

            // Feature point position (0-1 within cell)
            float px = float(perm_lookup(h, seed)) / 255.0;
            float py = float(perm_lookup(h + 1, seed)) / 255.0;
            vec2 featurePoint = neighbor + vec2(px, py);

            // Distance from p to feature point
            vec2 diff = featurePoint - frac;
            float dist = length(diff);

            minDist = min(minDist, dist);
        }
    }

    return minDist;
}

// Worley noise returning F1 and F2 distances (useful for cave shapes)
vec2 worley_2d_f1f2(vec2 p, float seed) {
    vec2 cell = floor(p);
    vec2 frac = fract(p);

    float f1 = 1.0;  // Nearest
    float f2 = 1.0;  // Second nearest

    for (int j = -1; j <= 1; j++) {
        for (int i = -1; i <= 1; i++) {
            vec2 neighbor = vec2(float(i), float(j));
            vec2 cellCoord = cell + neighbor;

            int h = perm2(int(cellCoord.x) & 255, int(cellCoord.y) & 255, seed);
            float px = float(perm_lookup(h, seed)) / 255.0;
            float py = float(perm_lookup(h + 1, seed)) / 255.0;
            vec2 featurePoint = neighbor + vec2(px, py);

            float dist = length(featurePoint - frac);

            if (dist < f1) {
                f2 = f1;
                f1 = dist;
            } else if (dist < f2) {
                f2 = dist;
            }
        }
    }

    return vec2(f1, f2);
}

// ============================================================================
// Utility functions for terrain generation
// ============================================================================

// Ridged noise - absolute value creates ridges
float ridged_noise_2d(vec2 p, int octaves, float persistence, float seed) {
    float total = 0.0;
    float amplitude = 1.0;
    float frequency = 1.0;
    float maxValue = 0.0;

    for (int i = 0; i < octaves; i++) {
        float n = perlin_2d(p * frequency, seed + float(i) * 100.0);
        n = 1.0 - abs(n);  // Invert absolute value for ridges
        n = n * n;         // Square for sharper ridges
        total += n * amplitude;
        maxValue += amplitude;
        amplitude *= persistence;
        frequency *= 2.0;
    }

    return total / maxValue;
}

// Billowed noise - like clouds
float billowed_noise_2d(vec2 p, int octaves, float persistence, float seed) {
    float total = 0.0;
    float amplitude = 1.0;
    float frequency = 1.0;
    float maxValue = 0.0;

    for (int i = 0; i < octaves; i++) {
        float n = abs(perlin_2d(p * frequency, seed + float(i) * 100.0));
        total += n * amplitude;
        maxValue += amplitude;
        amplitude *= persistence;
        frequency *= 2.0;
    }

    return total / maxValue;
}

// Domain warping - distort noise coordinates with noise
float warped_noise_2d(vec2 p, int octaves, float warpStrength, float seed) {
    // First pass: get warp offset
    float warpX = octave_noise_2d(p + vec2(0.0, 0.0), octaves, 0.5, seed);
    float warpY = octave_noise_2d(p + vec2(5.2, 1.3), octaves, 0.5, seed + 50.0);

    // Second pass: sample with warped coordinates
    vec2 warped = p + vec2(warpX, warpY) * warpStrength;
    return octave_noise_2d(warped, octaves, 0.5, seed + 100.0);
}

// Turbulence - sum of absolute values (good for fire, smoke)
float turbulence_2d(vec2 p, int octaves, float seed) {
    float total = 0.0;
    float amplitude = 1.0;
    float frequency = 1.0;
    float maxValue = 0.0;

    for (int i = 0; i < octaves; i++) {
        total += abs(perlin_2d(p * frequency, seed + float(i) * 100.0)) * amplitude;
        maxValue += amplitude;
        amplitude *= 0.5;
        frequency *= 2.0;
    }

    return total / maxValue;
}

// ============================================================================
// Minecraft-specific terrain noise
// ============================================================================

// MC 1.8.9 terrain base heightmap noise
// Returns height offset from base level
float mc_terrain_noise(vec2 worldXZ, float seed) {
    // Base continental shape
    float continental = octave_noise_2d(worldXZ * 0.0005, 4, 0.5, seed) * 64.0;

    // Medium detail
    float terrain = octave_noise_2d(worldXZ * 0.004, 4, 0.5, seed + 1000.0) * 16.0;

    // Fine detail
    float detail = octave_noise_2d(worldXZ * 0.02, 2, 0.5, seed + 2000.0) * 4.0;

    // Micro detail
    float micro = octave_noise_2d(worldXZ * 0.1, 2, 0.5, seed + 3000.0) * 1.0;

    return continental + terrain + detail + micro;
}

// MC 1.8.9 cave noise
// Returns 1.0 if solid, 0.0 if cave
float mc_cave_noise(vec3 worldPos, float seed) {
    // Main cheese caves
    float cheese = octave_noise_3d(worldPos * 0.05, 3, 0.5, seed);
    float cheeseThreshold = 0.55 + (worldPos.y / 64.0) * 0.1;  // Caves less common higher up

    // Spaghetti caves (worm-like tunnels)
    float worm1 = perlin_3d(worldPos * 0.02, seed + 5000.0);
    float worm2 = perlin_3d(worldPos * 0.02 + vec3(100, 0, 100), seed + 5001.0);
    float spaghetti = 1.0 - (worm1 * worm1 + worm2 * worm2);

    // Combine: if either carves, it's a cave
    if (cheese > cheeseThreshold || spaghetti > 0.95) {
        return 0.0;  // Cave
    }
    return 1.0;  // Solid
}

// MC 1.8.9 ore distribution noise
// Returns ore density factor (0-1) at position
float mc_ore_noise(vec3 worldPos, float baseRarity, float seed) {
    float noise = perlin_3d(worldPos * 0.15, seed);
    return smoothstep(baseRarity, 1.0, noise);
}

#endif // PERLIN_NOISE_GLSL
