import os
import struct

OUTPUT_FILE = "input.bin"
MAGIC = b"FOI32v1\0"
CHUNK_BYTES = 4 * 1024 * 1024


def write_random_int32_block(file, size):
    file.write(struct.pack("<Q", size))

    bytes_remaining = size * 4
    while bytes_remaining > 0:
        chunk_size = min(CHUNK_BYTES, bytes_remaining)
        file.write(os.urandom(chunk_size))
        bytes_remaining -= chunk_size


def main():
    sizes = [10_000, 100_000, 1_000_000, 10_000_000, 100_000_000]

    with open(OUTPUT_FILE, "wb") as file:
        file.write(MAGIC)
        file.write(struct.pack("<Q", len(sizes)))

        for size in sizes:
            write_random_int32_block(file, size)


if __name__ == "__main__":
    main()
