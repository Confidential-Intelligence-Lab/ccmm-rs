#!/usr/bin/env python3
"""R16.5.4: compare UHD cross-tile operand alignment."""

TILE_ROWS = 135
TILE_COLS = 15
LOCAL_ROWS = 16
LOCAL_COLS = 256
TILE_SLOTS = LOCAL_ROWS * LOCAL_COLS

def groups(length, radices):
    assert __import__("math").prod(radices) == length
    stride = 1
    for radix in radices:
        block = stride * radix
        for base in range(0, length, block):
            for offset in range(stride):
                yield [
                    base + offset + digit * stride
                    for digit in range(radix)
                ]
        stride = block

def locate(tile_row, tile_col, packing):
    tile = tile_row * TILE_COLS + tile_col

    if packing == "compact":
        return tile // 2, (tile % 2) * TILE_SLOTS

    if packing == "row_aligned":
        return tile_row * 8 + tile_col // 2, (tile_col % 2) * TILE_SLOTS

    raise ValueError(packing)

def classify(locations):
    cts = [ct for ct, _ in locations]
    slots = [slot for _, slot in locations]

    if len(set(cts)) == 1:
        return "same"
    if len(set(slots)) == 1:
        return "aligned"
    return "realignment"

def count(packing, axis):
    counts = {"same": 0, "aligned": 0, "realignment": 0}

    if axis == "horizontal":
        butterfly_groups = list(groups(TILE_COLS, [3, 5]))
        for tile_row in range(TILE_ROWS):
            for group in butterfly_groups:
                locations = [
                    locate(tile_row, tile_col, packing)
                    for tile_col in group
                ]
                counts[classify(locations)] += LOCAL_ROWS * LOCAL_COLS

    else:
        butterfly_groups = list(groups(TILE_ROWS, [3, 3, 3, 5]))
        for tile_col in range(TILE_COLS):
            for group in butterfly_groups:
                locations = [
                    locate(tile_row, tile_col, packing)
                    for tile_row in group
                ]
                counts[classify(locations)] += LOCAL_ROWS * LOCAL_COLS

    return counts

def main():
    for packing in ("compact", "row_aligned"):
        for axis in ("horizontal", "vertical"):
            result = count(packing, axis)
            total = sum(result.values())

            expected = 4_423_680 if axis == "horizontal" else 9_953_280
            assert total == expected

            print(
                f"R16_5_4_PACKING={packing} AXIS={axis} "
                f"same={result['same']} "
                f"aligned={result['aligned']} "
                f"realignment={result['realignment']} "
                f"total={total}"
            )

    print("R16_5_4_PACKING_COMPARISON=PASS")

if __name__ == "__main__":
    main()
