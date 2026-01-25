"""
Terrain test data generator for verifying Minecraft terrain generation.

Generates test vectors using Java-compatible random number generation
to verify noise functions, biome placement, and structure positions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator


class JavaRandom:
    """
    Java's java.util.Random implementation.

    Uses the same linear congruential generator (LCG) as Java for
    bit-exact reproducibility of Minecraft's RNG.
    """

    MULTIPLIER = 0x5DEECE66D
    ADDEND = 0xB
    MASK = (1 << 48) - 1

    def __init__(self, seed: int) -> None:
        self.seed = (seed ^ self.MULTIPLIER) & self.MASK

    def _next(self, bits: int) -> int:
        self.seed = (self.seed * self.MULTIPLIER + self.ADDEND) & self.MASK
        return self.seed >> (48 - bits)

    def next_int(self, bound: int | None = None) -> int:
        if bound is None:
            return self._next(32) - (1 << 31) if self._next(32) >= (1 << 31) else self._next(32)
        if bound <= 0:
            raise ValueError("bound must be positive")

        # Java's algorithm for nextInt(bound)
        if (bound & -bound) == bound:  # Power of 2
            return (bound * self._next(31)) >> 31

        bits = self._next(31)
        val = bits % bound
        while bits - val + (bound - 1) < 0:
            bits = self._next(31)
            val = bits % bound
        return val

    def next_int_32(self) -> int:
        """Return signed 32-bit integer like Java."""
        v = self._next(32)
        return v if v < (1 << 31) else v - (1 << 32)

    def next_long(self) -> int:
        """Return signed 64-bit integer like Java."""
        high = self._next(32)
        low = self._next(32)
        val = (high << 32) + low
        if val >= (1 << 63):
            val -= 1 << 64
        return val

    def next_float(self) -> float:
        return self._next(24) / (1 << 24)

    def next_double(self) -> float:
        return ((self._next(26) << 27) + self._next(27)) / (1 << 53)

    def next_gaussian(self) -> float:
        """Box-Muller transform for Gaussian distribution."""
        import math

        while True:
            v1 = 2 * self.next_double() - 1
            v2 = 2 * self.next_double() - 1
            s = v1 * v1 + v2 * v2
            if s < 1 and s != 0:
                break
        multiplier = math.sqrt(-2 * math.log(s) / s)
        return v1 * multiplier

    def set_seed(self, seed: int) -> None:
        self.seed = (seed ^ self.MULTIPLIER) & self.MASK


class SimplexNoise:
    """
    Simplex noise implementation matching Minecraft's terrain generation.

    Based on Ken Perlin's improved noise algorithm as implemented in
    Minecraft's net.minecraft.world.gen.SimplexNoise class.
    """

    SQRT_3 = 1.7320508075688772
    F2 = 0.5 * (SQRT_3 - 1)
    G2 = (3 - SQRT_3) / 6

    GRAD3 = [
        (1, 1, 0),
        (-1, 1, 0),
        (1, -1, 0),
        (-1, -1, 0),
        (1, 0, 1),
        (-1, 0, 1),
        (1, 0, -1),
        (-1, 0, -1),
        (0, 1, 1),
        (0, -1, 1),
        (0, 1, -1),
        (0, -1, -1),
    ]

    def __init__(self, rand: JavaRandom) -> None:
        self.perm = list(range(256))
        self.x_offset = rand.next_double() * 256
        self.y_offset = rand.next_double() * 256
        self.z_offset = rand.next_double() * 256

        # Fisher-Yates shuffle using Java random
        for i in range(255, 0, -1):
            j = rand.next_int(i + 1)
            self.perm[i], self.perm[j] = self.perm[j], self.perm[i]

        # Extend permutation table
        self.perm = self.perm + self.perm

    def _grad(self, hash_val: int, x: float, y: float) -> float:
        h = hash_val & 7
        u = x if h < 4 else y
        v = y if h < 4 else x
        return (u if (h & 1) == 0 else -u) + (v if (h & 2) == 0 else -v)

    def noise_2d(self, x: float, y: float) -> float:
        """2D simplex noise."""
        x = x + self.x_offset
        y = y + self.y_offset

        s = (x + y) * self.F2
        i = int(x + s) if x + s >= 0 else int(x + s) - 1
        j = int(y + s) if y + s >= 0 else int(y + s) - 1

        t = (i + j) * self.G2
        x0 = x - (i - t)
        y0 = y - (j - t)

        i1, j1 = (1, 0) if x0 > y0 else (0, 1)

        x1 = x0 - i1 + self.G2
        y1 = y0 - j1 + self.G2
        x2 = x0 - 1 + 2 * self.G2
        y2 = y0 - 1 + 2 * self.G2

        ii = i & 255
        jj = j & 255

        n0 = n1 = n2 = 0.0

        t0 = 0.5 - x0 * x0 - y0 * y0
        if t0 >= 0:
            t0 *= t0
            gi0 = self.perm[ii + self.perm[jj]] % 12
            n0 = t0 * t0 * self._grad(gi0, x0, y0)

        t1 = 0.5 - x1 * x1 - y1 * y1
        if t1 >= 0:
            t1 *= t1
            gi1 = self.perm[ii + i1 + self.perm[jj + j1]] % 12
            n1 = t1 * t1 * self._grad(gi1, x1, y1)

        t2 = 0.5 - x2 * x2 - y2 * y2
        if t2 >= 0:
            t2 *= t2
            gi2 = self.perm[ii + 1 + self.perm[jj + 1]] % 12
            n2 = t2 * t2 * self._grad(gi2, x2, y2)

        return 70.0 * (n0 + n1 + n2)


class PerlinNoise:
    """
    Perlin noise implementation matching Minecraft's ImprovedNoise.
    """

    def __init__(self, rand: JavaRandom) -> None:
        self.perm = list(range(256))
        self.x_offset = rand.next_double() * 256
        self.y_offset = rand.next_double() * 256
        self.z_offset = rand.next_double() * 256

        for i in range(255, 0, -1):
            j = rand.next_int(i + 1)
            self.perm[i], self.perm[j] = self.perm[j], self.perm[i]

        self.perm = self.perm + self.perm

    @staticmethod
    def _fade(t: float) -> float:
        return t * t * t * (t * (t * 6 - 15) + 10)

    @staticmethod
    def _lerp(t: float, a: float, b: float) -> float:
        return a + t * (b - a)

    @staticmethod
    def _grad(hash_val: int, x: float, y: float, z: float) -> float:
        h = hash_val & 15
        u = x if h < 8 else y
        v = y if h < 4 else (x if h == 12 or h == 14 else z)
        return (u if (h & 1) == 0 else -u) + (v if (h & 2) == 0 else -v)

    def noise_3d(self, x: float, y: float, z: float) -> float:
        x = x + self.x_offset
        y = y + self.y_offset
        z = z + self.z_offset

        xi = int(x) if x >= 0 else int(x) - 1
        yi = int(y) if y >= 0 else int(y) - 1
        zi = int(z) if z >= 0 else int(z) - 1

        xf = x - xi
        yf = y - yi
        zf = z - zi

        xi &= 255
        yi &= 255
        zi &= 255

        u = self._fade(xf)
        v = self._fade(yf)
        w = self._fade(zf)

        p = self.perm
        a = p[xi] + yi
        aa = p[a] + zi
        ab = p[a + 1] + zi
        b = p[xi + 1] + yi
        ba = p[b] + zi
        bb = p[b + 1] + zi

        return self._lerp(
            w,
            self._lerp(
                v,
                self._lerp(u, self._grad(p[aa], xf, yf, zf), self._grad(p[ba], xf - 1, yf, zf)),
                self._lerp(
                    u, self._grad(p[ab], xf, yf - 1, zf), self._grad(p[bb], xf - 1, yf - 1, zf)
                ),
            ),
            self._lerp(
                v,
                self._lerp(
                    u,
                    self._grad(p[aa + 1], xf, yf, zf - 1),
                    self._grad(p[ba + 1], xf - 1, yf, zf - 1),
                ),
                self._lerp(
                    u,
                    self._grad(p[ab + 1], xf, yf - 1, zf - 1),
                    self._grad(p[bb + 1], xf - 1, yf - 1, zf - 1),
                ),
            ),
        )


class OctaveNoise:
    """Multi-octave noise combining multiple PerlinNoise instances."""

    def __init__(self, rand: JavaRandom, octaves: int) -> None:
        self.octaves = octaves
        self.generators = [PerlinNoise(rand) for _ in range(octaves)]

    def noise_3d(self, x: float, y: float, z: float) -> float:
        result = 0.0
        scale = 1.0
        for gen in self.generators:
            result += gen.noise_3d(x * scale, y * scale, z * scale) / scale
            scale *= 2.0
        return result


@dataclass
class NoiseTestVector:
    """Test vector for noise function verification."""

    seed: int
    x: float
    y: float
    z: float
    expected_simplex_2d: float | None
    expected_perlin_3d: float | None


@dataclass
class BiomeTestVector:
    """Test vector for biome placement verification."""

    seed: int
    x: int
    z: int
    expected_biome_id: int
    expected_temperature: float
    expected_humidity: float


def generate_noise_test_vectors(
    seed: int, count: int = 10000, coord_range: float = 1000.0
) -> Iterator[NoiseTestVector]:
    """
    Generate test vectors for noise function verification.

    Args:
        seed: World seed
        count: Number of test vectors
        coord_range: Range for random coordinates

    Yields:
        NoiseTestVector instances with computed expected values
    """
    coord_rand = JavaRandom(seed ^ 0xDEADBEEF)
    noise_rand = JavaRandom(seed)

    simplex = SimplexNoise(noise_rand)
    perlin = PerlinNoise(JavaRandom(seed))

    for _ in range(count):
        x = (coord_rand.next_double() * 2 - 1) * coord_range
        y = (coord_rand.next_double() * 2 - 1) * coord_range
        z = (coord_rand.next_double() * 2 - 1) * coord_range

        yield NoiseTestVector(
            seed=seed,
            x=x,
            y=y,
            z=z,
            expected_simplex_2d=simplex.noise_2d(x, z),
            expected_perlin_3d=perlin.noise_3d(x, y, z),
        )


def generate_biome_test_grid(
    seed: int,
    min_x: int = -500,
    max_x: int = 500,
    min_z: int = -500,
    max_z: int = 500,
    step: int = 16,
) -> Iterator[BiomeTestVector]:
    """
    Generate a grid of biome test points.

    Uses Minecraft's biome generation algorithm which combines
    temperature and humidity noise to determine biome placement.

    Args:
        seed: World seed
        min_x, max_x: X coordinate range
        min_z, max_z: Z coordinate range
        step: Grid step size (typically chunk size)

    Yields:
        BiomeTestVector instances
    """
    temp_rand = JavaRandom(seed * 9871)
    humid_rand = JavaRandom(seed * 39811)

    temp_noise = OctaveNoise(temp_rand, 4)
    humid_noise = OctaveNoise(humid_rand, 4)

    for x in range(min_x, max_x + 1, step):
        for z in range(min_z, max_z + 1, step):
            temp = temp_noise.noise_3d(x / 8.0, 0.0, z / 8.0)
            humid = humid_noise.noise_3d(x / 8.0, 0.0, z / 8.0)

            # Normalize to 0-1 range
            temp = (temp + 1) / 2
            humid = (humid + 1) / 2

            # Simple biome selection (actual MC is more complex)
            biome_id = _select_biome(temp, humid)

            yield BiomeTestVector(
                seed=seed,
                x=x,
                z=z,
                expected_biome_id=biome_id,
                expected_temperature=temp,
                expected_humidity=humid,
            )


def _select_biome(temperature: float, humidity: float) -> int:
    """
    Select biome ID based on temperature and humidity.

    Simplified version of Minecraft's biome selection.
    Real implementation uses BiomeGenBase lookups.

    Returns:
        Biome ID
    """
    # Biome IDs from Minecraft
    OCEAN = 0
    PLAINS = 1
    DESERT = 2
    FOREST = 4
    TAIGA = 5
    SWAMP = 6
    JUNGLE = 21
    ICE_PLAINS = 12

    if temperature < 0.2:
        return ICE_PLAINS if humidity < 0.5 else TAIGA
    elif temperature < 0.5:
        return FOREST if humidity > 0.5 else PLAINS
    elif temperature < 0.8:
        return SWAMP if humidity > 0.7 else (JUNGLE if humidity > 0.5 else PLAINS)
    else:
        return DESERT if humidity < 0.3 else PLAINS


def save_test_vectors(
    vectors: Iterator[NoiseTestVector] | Iterator[BiomeTestVector], filepath: str
) -> int:
    """
    Save test vectors to binary file for efficient loading.

    Args:
        vectors: Iterator of test vectors
        filepath: Output file path

    Returns:
        Number of vectors written
    """
    import json

    count = 0
    data = []

    for vec in vectors:
        if isinstance(vec, NoiseTestVector):
            data.append(
                {
                    "seed": vec.seed,
                    "x": vec.x,
                    "y": vec.y,
                    "z": vec.z,
                    "simplex_2d": vec.expected_simplex_2d,
                    "perlin_3d": vec.expected_perlin_3d,
                }
            )
        else:
            data.append(
                {
                    "seed": vec.seed,
                    "x": vec.x,
                    "z": vec.z,
                    "biome_id": vec.expected_biome_id,
                    "temperature": vec.expected_temperature,
                    "humidity": vec.expected_humidity,
                }
            )
        count += 1

    with open(filepath, "w") as f:
        json.dump(data, f)

    return count


if __name__ == "__main__":
    import sys

    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 12345

    print(f"Generating noise test vectors for seed {seed}...")
    noise_count = save_test_vectors(
        generate_noise_test_vectors(seed, count=10000), f"noise_vectors_{seed}.json"
    )
    print(f"  Wrote {noise_count} noise vectors")

    print(f"Generating biome test grid for seed {seed}...")
    biome_count = save_test_vectors(generate_biome_test_grid(seed), f"biome_vectors_{seed}.json")
    print(f"  Wrote {biome_count} biome vectors")
