// Copyright © 2026 Apple Inc.

import Foundation

public enum GGUFQuantizationError: Error, Equatable, LocalizedError {
    case unsupportedShape([Int])
    case incompatibleShape([Int])
    case invalidDataSize(expected: Int, actual: Int)

    public var errorDescription: String? {
        switch self {
        case .unsupportedShape(let shape):
            return "GGUF quantized tensor shape must be at least 2D, got \(shape)."
        case .incompatibleShape(let shape):
            return
                "GGUF Q4_K tensor shape must have a last dimension divisible by 256, got \(shape)."
        case .invalidDataSize(let expected, let actual):
            return "GGUF Q4_K tensor data has \(actual) bytes, expected \(expected)."
        }
    }
}

public struct GGUFAffineQuantizedBuffers: Equatable, Sendable {
    public var weightData: Data
    public var scalesData: Data
    public var biasesData: Data
    public var weightShape: [Int]
    public var scalesShape: [Int]
    public var groupSize: Int
    public var bits: Int
}

public struct GGUFAffineQuantizedArrays {
    public var weight: MLXArray
    public var scales: MLXArray
    public var biases: MLXArray
    public var groupSize: Int
    public var bits: Int
}

public enum GGUFKQuantization: Sendable {
    case q8_0
    case q2K
    case q3K
    case q4K
    case q5K
    case q6K
    case q8K

    public var blockSize: Int {
        switch self {
        case .q8_0: 34
        case .q2K: 84
        case .q3K: 110
        case .q4K: 144
        case .q5K: 176
        case .q6K: 210
        case .q8K: 292
        }
    }

    public var valuesPerBlock: Int {
        switch self {
        case .q8_0: 32
        case .q2K, .q3K, .q4K, .q5K, .q6K, .q8K: 256
        }
    }

    public var groupSize: Int {
        switch self {
        case .q8_0, .q4K, .q5K: 32
        case .q2K, .q3K, .q6K, .q8K: 16
        }
    }

    public var bits: Int {
        switch self {
        case .q8_0: 8
        case .q2K: 2
        case .q3K: 3
        case .q4K: 4
        case .q5K: 5
        case .q6K: 6
        case .q8K: 8
        }
    }

    fileprivate var groupsPerBlock: Int { valuesPerBlock / groupSize }
    fileprivate var weightBytesPerBlock: Int { valuesPerBlock * bits / 8 }
    fileprivate var scaleBytesPerBlock: Int { groupsPerBlock * MemoryLayout<Float16>.size }
}

public func ggufKDataByteCount(format: GGUFKQuantization, shape: [Int]) throws -> Int {
    try ggufKDataByteCount(
        shape: shape, valuesPerBlock: format.valuesPerBlock, blockSize: format.blockSize)
}

public func ggufKToAffineQuantizedBuffers(
    _ data: Data, format: GGUFKQuantization, shape: [Int]
) throws -> GGUFAffineQuantizedBuffers {
    try data.withUnsafeBytes { rawBuffer in
        try ggufKToAffineQuantizedBuffers(
            rawBuffer.bindMemory(to: UInt8.self), format: format, shape: shape)
    }
}

public func ggufKToAffineQuantizedBuffers(
    _ bytes: UnsafeBufferPointer<UInt8>, format: GGUFKQuantization, shape: [Int]
) throws -> GGUFAffineQuantizedBuffers {
    let expectedDataSize = try ggufKDataByteCount(format: format, shape: shape)
    guard bytes.count == expectedDataSize else {
        throw GGUFQuantizationError.invalidDataSize(expected: expectedDataSize, actual: bytes.count)
    }

    let blockCount = expectedDataSize / format.blockSize
    var buffers = makeAffineBuffers(shape: shape, format: format)

    buffers.weightData.withUnsafeMutableBytes { weightRawBuffer in
        buffers.scalesData.withUnsafeMutableBytes { scalesRawBuffer in
            buffers.biasesData.withUnsafeMutableBytes { biasesRawBuffer in
                let weightBytes = weightRawBuffer.bindMemory(to: UInt8.self)
                let scaleBytes = scalesRawBuffer.bindMemory(to: UInt8.self)
                let biasBytes = biasesRawBuffer.bindMemory(to: UInt8.self)
                var values = [UInt8](repeating: 0, count: format.valuesPerBlock)

                for blockIndex in 0 ..< blockCount {
                    let blockOffset = blockIndex * format.blockSize
                    let weightBlockOffset = blockIndex * format.weightBytesPerBlock
                    let scaleBlockOffset = blockIndex * format.scaleBytesPerBlock

                    switch format {
                    case .q8_0:
                        convertQ8_0Block(
                            bytes,
                            blockOffset: blockOffset,
                            weightBytes: weightBytes,
                            weightBlockOffset: weightBlockOffset,
                            scaleBytes: scaleBytes,
                            biasBytes: biasBytes,
                            scaleBlockOffset: scaleBlockOffset)
                    case .q2K:
                        convertQ2KBlock(
                            bytes,
                            blockOffset: blockOffset,
                            values: &values,
                            weightBytes: weightBytes,
                            weightBlockOffset: weightBlockOffset,
                            scaleBytes: scaleBytes,
                            biasBytes: biasBytes,
                            scaleBlockOffset: scaleBlockOffset)
                    case .q3K:
                        convertQ3KBlock(
                            bytes,
                            blockOffset: blockOffset,
                            values: &values,
                            weightBytes: weightBytes,
                            weightBlockOffset: weightBlockOffset,
                            scaleBytes: scaleBytes,
                            biasBytes: biasBytes,
                            scaleBlockOffset: scaleBlockOffset)
                    case .q4K:
                        convertQ4KBlock(
                            bytes,
                            blockOffset: blockOffset,
                            weightBytes: weightBytes,
                            weightBlockOffset: weightBlockOffset,
                            scaleBytes: scaleBytes,
                            biasBytes: biasBytes,
                            scaleBlockOffset: scaleBlockOffset)
                    case .q5K:
                        convertQ5KBlock(
                            bytes,
                            blockOffset: blockOffset,
                            values: &values,
                            weightBytes: weightBytes,
                            weightBlockOffset: weightBlockOffset,
                            scaleBytes: scaleBytes,
                            biasBytes: biasBytes,
                            scaleBlockOffset: scaleBlockOffset)
                    case .q6K:
                        convertQ6KBlock(
                            bytes,
                            blockOffset: blockOffset,
                            values: &values,
                            weightBytes: weightBytes,
                            weightBlockOffset: weightBlockOffset,
                            scaleBytes: scaleBytes,
                            biasBytes: biasBytes,
                            scaleBlockOffset: scaleBlockOffset)
                    case .q8K:
                        convertQ8KBlock(
                            bytes,
                            blockOffset: blockOffset,
                            weightBytes: weightBytes,
                            weightBlockOffset: weightBlockOffset,
                            scaleBytes: scaleBytes,
                            biasBytes: biasBytes,
                            scaleBlockOffset: scaleBlockOffset)
                    }
                }
            }
        }
    }

    return buffers
}

public func ggufKToAffineQuantizedArrays(
    _ data: Data, format: GGUFKQuantization, shape: [Int]
) throws -> GGUFAffineQuantizedArrays {
    ggufAffineQuantizedArrays(
        from: try ggufKToAffineQuantizedBuffers(data, format: format, shape: shape))
}

public func ggufKToAffineQuantizedArrays(
    _ bytes: UnsafeBufferPointer<UInt8>, format: GGUFKQuantization, shape: [Int]
) throws -> GGUFAffineQuantizedArrays {
    ggufAffineQuantizedArrays(
        from: try ggufKToAffineQuantizedBuffers(bytes, format: format, shape: shape))
}

private func convertQ4KBlock(
    _ bytes: UnsafeBufferPointer<UInt8>,
    blockOffset: Int,
    weightBytes: UnsafeMutableBufferPointer<UInt8>,
    weightBlockOffset: Int,
    scaleBytes: UnsafeMutableBufferPointer<UInt8>,
    biasBytes: UnsafeMutableBufferPointer<UInt8>,
    scaleBlockOffset: Int
) {
    let d = Float(Float16(bitPattern: readUInt16(bytes, blockOffset)))
    let dmin = Float(Float16(bitPattern: readUInt16(bytes, blockOffset + 2)))
    let scalesOffset = blockOffset + 4
    let quantsOffset = blockOffset + 16

    for group in 0 ..< 8 {
        writePackedQ4KGroup(
            bytes,
            quantsOffset: quantsOffset,
            group: group,
            to: weightBytes,
            at: weightBlockOffset + group * 16)

        let (scale, minimum) = readQ4KScaleMin(bytes, scalesOffset: scalesOffset, group: group)
        writeLittleEndian(
            Float16(d * Float(scale)).bitPattern,
            to: scaleBytes,
            at: scaleBlockOffset + group * 2)
        writeLittleEndian(
            Float16(-dmin * Float(minimum)).bitPattern,
            to: biasBytes,
            at: scaleBlockOffset + group * 2)
    }
}

private func convertQ5KBlock(
    _ bytes: UnsafeBufferPointer<UInt8>,
    blockOffset: Int,
    values: inout [UInt8],
    weightBytes: UnsafeMutableBufferPointer<UInt8>,
    weightBlockOffset: Int,
    scaleBytes: UnsafeMutableBufferPointer<UInt8>,
    biasBytes: UnsafeMutableBufferPointer<UInt8>,
    scaleBlockOffset: Int
) {
    let d = Float(Float16(bitPattern: readUInt16(bytes, blockOffset)))
    let dmin = Float(Float16(bitPattern: readUInt16(bytes, blockOffset + 2)))
    let scalesOffset = blockOffset + 4
    let qhOffset = blockOffset + 16
    let qsOffset = blockOffset + 48

    for pair in 0 ..< 4 {
        let highBitLow = UInt8(1 << (pair * 2))
        let highBitHigh = UInt8(1 << (pair * 2 + 1))
        for l in 0 ..< 32 {
            let packed = bytes[qsOffset + pair * 32 + l]
            values[pair * 64 + l] =
                (packed & 0x0f) + ((bytes[qhOffset + l] & highBitLow) != 0 ? 16 : 0)
            values[pair * 64 + 32 + l] =
                (packed >> 4) + ((bytes[qhOffset + l] & highBitHigh) != 0 ? 16 : 0)
        }
    }

    for group in 0 ..< 8 {
        writePackedQ5KGroup(
            values, group: group, to: weightBytes, at: weightBlockOffset + group * 20)

        let (scaleMultiplier, minimum) = readQ4KScaleMin(
            bytes, scalesOffset: scalesOffset, group: group)
        let scale = d * Float(scaleMultiplier)
        writeLittleEndian(
            Float16(scale).bitPattern, to: scaleBytes, at: scaleBlockOffset + group * 2)
        writeLittleEndian(
            Float16(-dmin * Float(minimum)).bitPattern,
            to: biasBytes,
            at: scaleBlockOffset + group * 2)
    }
}

private func convertQ2KBlock(
    _ bytes: UnsafeBufferPointer<UInt8>,
    blockOffset: Int,
    values: inout [UInt8],
    weightBytes: UnsafeMutableBufferPointer<UInt8>,
    weightBlockOffset: Int,
    scaleBytes: UnsafeMutableBufferPointer<UInt8>,
    biasBytes: UnsafeMutableBufferPointer<UInt8>,
    scaleBlockOffset: Int
) {
    let d = Float(Float16(bitPattern: readUInt16(bytes, blockOffset + 80)))
    let dmin = Float(Float16(bitPattern: readUInt16(bytes, blockOffset + 82)))
    var iscale = 0
    var qOffset = blockOffset + 16

    for half in 0 ..< 2 {
        var shift = 0
        for _ in 0 ..< 4 {
            let group0 = iscale
            for l in 0 ..< 16 {
                values[half * 128 + (group0 % 8) * 16 + l] =
                    (bytes[qOffset + l] >> shift) & 0x03
            }
            iscale += 1

            let group1 = iscale
            for l in 0 ..< 16 {
                values[half * 128 + (group1 % 8) * 16 + l] =
                    (bytes[qOffset + 16 + l] >> shift) & 0x03
            }
            iscale += 1
            shift += 2
        }
        qOffset += 32
    }

    for group in 0 ..< 16 {
        writePackedQ2KGroup(
            values, group: group, to: weightBytes, at: weightBlockOffset + group * 4)

        let sc = bytes[blockOffset + group]
        let scale = d * Float(sc & 0x0f)
        writeLittleEndian(
            Float16(scale).bitPattern, to: scaleBytes, at: scaleBlockOffset + group * 2)
        writeLittleEndian(
            Float16(-dmin * Float(sc >> 4)).bitPattern,
            to: biasBytes,
            at: scaleBlockOffset + group * 2)
    }
}

private func convertQ3KBlock(
    _ bytes: UnsafeBufferPointer<UInt8>,
    blockOffset: Int,
    values: inout [UInt8],
    weightBytes: UnsafeMutableBufferPointer<UInt8>,
    weightBlockOffset: Int,
    scaleBytes: UnsafeMutableBufferPointer<UInt8>,
    biasBytes: UnsafeMutableBufferPointer<UInt8>,
    scaleBlockOffset: Int
) {
    let d = Float(Float16(bitPattern: readUInt16(bytes, blockOffset + 108)))
    let qsOffset = blockOffset + 32
    let scalesOffset = blockOffset + 96
    var group = 0
    var mask: UInt8 = 1

    for half in 0 ..< 2 {
        var shift = 0
        for _ in 0 ..< 4 {
            for l in 0 ..< 16 {
                let low = (bytes[qsOffset + half * 32 + l] >> shift) & 0x03
                values[group * 16 + l] = low + ((bytes[blockOffset + l] & mask) != 0 ? 4 : 0)
            }
            group += 1

            for l in 0 ..< 16 {
                let low = (bytes[qsOffset + half * 32 + 16 + l] >> shift) & 0x03
                values[group * 16 + l] = low + ((bytes[blockOffset + 16 + l] & mask) != 0 ? 4 : 0)
            }
            group += 1
            shift += 2
            mask <<= 1
        }
    }

    for group in 0 ..< 16 {
        writePackedQ3KGroup(
            values, group: group, to: weightBytes, at: weightBlockOffset + group * 6)

        let scale =
            d * Float(Int(readQ3KScale(bytes, scalesOffset: scalesOffset, group: group)) - 32)
        writeLittleEndian(
            Float16(scale).bitPattern, to: scaleBytes, at: scaleBlockOffset + group * 2)
        writeLittleEndian(
            Float16(-4 * scale).bitPattern, to: biasBytes, at: scaleBlockOffset + group * 2)
    }
}

private func convertQ6KBlock(
    _ bytes: UnsafeBufferPointer<UInt8>,
    blockOffset: Int,
    values: inout [UInt8],
    weightBytes: UnsafeMutableBufferPointer<UInt8>,
    weightBlockOffset: Int,
    scaleBytes: UnsafeMutableBufferPointer<UInt8>,
    biasBytes: UnsafeMutableBufferPointer<UInt8>,
    scaleBlockOffset: Int
) {
    let d = Float(Float16(bitPattern: readUInt16(bytes, blockOffset + 208)))

    for half in 0 ..< 2 {
        let valueOffset = half * 128
        let qlOffset = blockOffset + half * 64
        let qhOffset = blockOffset + 128 + half * 32

        for l in 0 ..< 32 {
            values[valueOffset + l] =
                (bytes[qlOffset + l] & 0x0f)
                | (((bytes[qhOffset + l] >> 0) & 0x03) << 4)
            values[valueOffset + l + 32] =
                (bytes[qlOffset + l + 32] & 0x0f)
                | (((bytes[qhOffset + l] >> 2) & 0x03) << 4)
            values[valueOffset + l + 64] =
                (bytes[qlOffset + l] >> 4)
                | (((bytes[qhOffset + l] >> 4) & 0x03) << 4)
            values[valueOffset + l + 96] =
                (bytes[qlOffset + l + 32] >> 4)
                | (((bytes[qhOffset + l] >> 6) & 0x03) << 4)
        }
    }

    for group in 0 ..< 16 {
        writePackedQ6KGroup(
            values, group: group, to: weightBytes, at: weightBlockOffset + group * 12)

        let scaleMultiplier = Int8(bitPattern: bytes[blockOffset + 192 + group])
        let scale = d * Float(scaleMultiplier)
        writeLittleEndian(
            Float16(scale).bitPattern, to: scaleBytes, at: scaleBlockOffset + group * 2)
        writeLittleEndian(
            Float16(-32 * scale).bitPattern, to: biasBytes, at: scaleBlockOffset + group * 2)
    }
}

private func convertQ8KBlock(
    _ bytes: UnsafeBufferPointer<UInt8>,
    blockOffset: Int,
    weightBytes: UnsafeMutableBufferPointer<UInt8>,
    weightBlockOffset: Int,
    scaleBytes: UnsafeMutableBufferPointer<UInt8>,
    biasBytes: UnsafeMutableBufferPointer<UInt8>,
    scaleBlockOffset: Int
) {
    let d = readFloat32(bytes, blockOffset)

    for group in 0 ..< 16 {
        writePackedQ8KGroup(
            bytes,
            quantsOffset: blockOffset + 4,
            group: group,
            to: weightBytes,
            at: weightBlockOffset + group * 16)
        writeLittleEndian(
            Float16(d).bitPattern, to: scaleBytes, at: scaleBlockOffset + group * 2)
        writeLittleEndian(
            Float16(-128 * d).bitPattern, to: biasBytes, at: scaleBlockOffset + group * 2)
    }
}

private func convertQ8_0Block(
    _ bytes: UnsafeBufferPointer<UInt8>,
    blockOffset: Int,
    weightBytes: UnsafeMutableBufferPointer<UInt8>,
    weightBlockOffset: Int,
    scaleBytes: UnsafeMutableBufferPointer<UInt8>,
    biasBytes: UnsafeMutableBufferPointer<UInt8>,
    scaleBlockOffset: Int
) {
    let d = readUInt16(bytes, blockOffset)

    for index in 0 ..< 32 {
        let signed = Int(Int8(bitPattern: bytes[blockOffset + 2 + index])) + 128
        weightBytes[weightBlockOffset + index] = UInt8(signed)
    }
    writeLittleEndian(d, to: scaleBytes, at: scaleBlockOffset)
    let scale = Float(Float16(bitPattern: d))
    writeLittleEndian(Float16(-128 * scale).bitPattern, to: biasBytes, at: scaleBlockOffset)
}

private func readUInt16(_ bytes: UnsafeBufferPointer<UInt8>, _ offset: Int) -> UInt16 {
    UInt16(bytes[offset]) | (UInt16(bytes[offset + 1]) << 8)
}

private func readFloat32(_ bytes: UnsafeBufferPointer<UInt8>, _ offset: Int) -> Float {
    let b0 = UInt32(bytes[offset])
    let b1 = UInt32(bytes[offset + 1]) << 8
    let b2 = UInt32(bytes[offset + 2]) << 16
    let b3 = UInt32(bytes[offset + 3]) << 24
    return Float(bitPattern: b0 | b1 | b2 | b3)
}

private func ggufKDataByteCount(shape: [Int], valuesPerBlock: Int, blockSize: Int) throws -> Int {
    guard shape.count >= 2 else {
        throw GGUFQuantizationError.unsupportedShape(shape)
    }
    guard let columns = shape.last, columns % valuesPerBlock == 0 else {
        throw GGUFQuantizationError.incompatibleShape(shape)
    }

    let rows = shape.dropLast().reduce(1, *)
    let blocksPerRow = columns / valuesPerBlock
    let blockCount = rows * blocksPerRow
    return blockCount * blockSize
}

private func makeAffineBuffers(shape: [Int], format: GGUFKQuantization)
    -> GGUFAffineQuantizedBuffers
{
    let columns = shape[shape.count - 1]
    let rows = shape.dropLast().reduce(1, *)
    var weightShape = shape
    weightShape[weightShape.count - 1] = columns * format.bits / 32
    var scalesShape = shape
    scalesShape[scalesShape.count - 1] = columns / format.groupSize

    return GGUFAffineQuantizedBuffers(
        weightData: Data(count: rows * columns * format.bits / 8),
        scalesData: Data(count: rows * columns / format.groupSize * MemoryLayout<Float16>.size),
        biasesData: Data(count: rows * columns / format.groupSize * MemoryLayout<Float16>.size),
        weightShape: weightShape,
        scalesShape: scalesShape,
        groupSize: format.groupSize,
        bits: format.bits)
}

private func ggufAffineQuantizedArrays(from buffers: GGUFAffineQuantizedBuffers)
    -> GGUFAffineQuantizedArrays
{
    GGUFAffineQuantizedArrays(
        weight: MLXArray(buffers.weightData, buffers.weightShape, dtype: .uint32),
        scales: MLXArray(buffers.scalesData, buffers.scalesShape, dtype: .float16),
        biases: MLXArray(buffers.biasesData, buffers.scalesShape, dtype: .float16),
        groupSize: buffers.groupSize,
        bits: buffers.bits)
}

private func readQ4KScaleMin(
    _ bytes: UnsafeBufferPointer<UInt8>, scalesOffset: Int, group: Int
) -> (scale: UInt8, minimum: UInt8) {
    if group < 4 {
        return (
            bytes[scalesOffset + group] & 63,
            bytes[scalesOffset + group + 4] & 63
        )
    } else {
        return (
            (bytes[scalesOffset + group + 4] & 0x0f)
                | ((bytes[scalesOffset + group - 4] >> 6) << 4),
            (bytes[scalesOffset + group + 4] >> 4)
                | ((bytes[scalesOffset + group] >> 6) << 4)
        )
    }
}

private func readQ3KScale(
    _ bytes: UnsafeBufferPointer<UInt8>, scalesOffset: Int, group: Int
) -> UInt8 {
    let low: UInt8
    if group < 8 {
        low = bytes[scalesOffset + group] & 0x0f
    } else {
        low = bytes[scalesOffset + group - 8] >> 4
    }
    let high = (bytes[scalesOffset + 8 + group % 4] >> UInt8((group / 4) * 2)) & 0x03
    return low | (high << 4)
}

private func writePackedQ4KGroup(
    _ bytes: UnsafeBufferPointer<UInt8>,
    quantsOffset: Int,
    group: Int,
    to output: UnsafeMutableBufferPointer<UInt8>,
    at outputOffset: Int
) {
    let pair = group / 2
    let highNibble = group % 2 == 1
    let pairOffset = quantsOffset + pair * 32
    var writeOffset = outputOffset
    for valuePair in stride(from: 0, to: 32, by: 2) {
        let first = q4KNibble(bytes[pairOffset + valuePair], high: highNibble)
        let second = q4KNibble(bytes[pairOffset + valuePair + 1], high: highNibble)
        output[writeOffset] = first | (second << 4)
        writeOffset += 1
    }
}

private func writePackedQ6KGroup(
    _ values: [UInt8],
    group: Int,
    to output: UnsafeMutableBufferPointer<UInt8>,
    at outputOffset: Int
) {
    let groupSize = 16
    let start = group * groupSize
    var writeOffset = outputOffset
    for index in stride(from: start, to: start + groupSize, by: 4) {
        let packed =
            UInt32(values[index])
            | (UInt32(values[index + 1]) << 6)
            | (UInt32(values[index + 2]) << 12)
            | (UInt32(values[index + 3]) << 18)
        output[writeOffset] = UInt8(truncatingIfNeeded: packed)
        output[writeOffset + 1] = UInt8(truncatingIfNeeded: packed >> 8)
        output[writeOffset + 2] = UInt8(truncatingIfNeeded: packed >> 16)
        writeOffset += 3
    }
}

private func writePackedQ2KGroup(
    _ values: [UInt8],
    group: Int,
    to output: UnsafeMutableBufferPointer<UInt8>,
    at outputOffset: Int
) {
    let start = group * 16
    for index in 0 ..< 4 {
        let valueOffset = start + index * 4
        output[outputOffset + index] =
            (values[valueOffset] & 0x03)
            | ((values[valueOffset + 1] & 0x03) << 2)
            | ((values[valueOffset + 2] & 0x03) << 4)
            | ((values[valueOffset + 3] & 0x03) << 6)
    }
}

private func writePackedQ3KGroup(
    _ values: [UInt8],
    group: Int,
    to output: UnsafeMutableBufferPointer<UInt8>,
    at outputOffset: Int
) {
    let start = group * 16
    for chunk in 0 ..< 2 {
        let valueOffset = start + chunk * 8
        let packed =
            UInt32(values[valueOffset] & 0x07)
            | (UInt32(values[valueOffset + 1] & 0x07) << 3)
            | (UInt32(values[valueOffset + 2] & 0x07) << 6)
            | (UInt32(values[valueOffset + 3] & 0x07) << 9)
            | (UInt32(values[valueOffset + 4] & 0x07) << 12)
            | (UInt32(values[valueOffset + 5] & 0x07) << 15)
            | (UInt32(values[valueOffset + 6] & 0x07) << 18)
            | (UInt32(values[valueOffset + 7] & 0x07) << 21)
        let writeOffset = outputOffset + chunk * 3
        output[writeOffset] = UInt8(truncatingIfNeeded: packed)
        output[writeOffset + 1] = UInt8(truncatingIfNeeded: packed >> 8)
        output[writeOffset + 2] = UInt8(truncatingIfNeeded: packed >> 16)
    }
}

private func writePackedQ5KGroup(
    _ values: [UInt8],
    group: Int,
    to output: UnsafeMutableBufferPointer<UInt8>,
    at outputOffset: Int
) {
    let start = group * 32
    for chunk in 0 ..< 4 {
        let valueOffset = start + chunk * 8
        let packed =
            UInt64(values[valueOffset] & 0x1f)
            | (UInt64(values[valueOffset + 1] & 0x1f) << 5)
            | (UInt64(values[valueOffset + 2] & 0x1f) << 10)
            | (UInt64(values[valueOffset + 3] & 0x1f) << 15)
            | (UInt64(values[valueOffset + 4] & 0x1f) << 20)
            | (UInt64(values[valueOffset + 5] & 0x1f) << 25)
            | (UInt64(values[valueOffset + 6] & 0x1f) << 30)
            | (UInt64(values[valueOffset + 7] & 0x1f) << 35)
        let writeOffset = outputOffset + chunk * 5
        output[writeOffset] = UInt8(truncatingIfNeeded: packed)
        output[writeOffset + 1] = UInt8(truncatingIfNeeded: packed >> 8)
        output[writeOffset + 2] = UInt8(truncatingIfNeeded: packed >> 16)
        output[writeOffset + 3] = UInt8(truncatingIfNeeded: packed >> 24)
        output[writeOffset + 4] = UInt8(truncatingIfNeeded: packed >> 32)
    }
}

private func writePackedQ8KGroup(
    _ bytes: UnsafeBufferPointer<UInt8>,
    quantsOffset: Int,
    group: Int,
    to output: UnsafeMutableBufferPointer<UInt8>,
    at outputOffset: Int
) {
    let groupSize = 16
    let start = group * groupSize
    for index in 0 ..< groupSize {
        let signed = Int(Int8(bitPattern: bytes[quantsOffset + start + index])) + 128
        output[outputOffset + index] = UInt8(signed)
    }
}

private func writeLittleEndian(
    _ value: UInt16,
    to output: UnsafeMutableBufferPointer<UInt8>,
    at offset: Int
) {
    output[offset] = UInt8(truncatingIfNeeded: value)
    output[offset + 1] = UInt8(truncatingIfNeeded: value >> 8)
}

private func q4KNibble(_ byte: UInt8, high: Bool) -> UInt8 {
    high ? byte >> 4 : byte & 0x0f
}

extension Data {
    fileprivate mutating func appendLittleEndian(_ value: Float16) {
        appendLittleEndian(value.bitPattern)
    }

    fileprivate mutating func appendLittleEndian(_ value: UInt16) {
        append(UInt8(truncatingIfNeeded: value))
        append(UInt8(truncatingIfNeeded: value >> 8))
    }

    fileprivate mutating func appendLittleEndian(_ value: UInt32) {
        append(UInt8(truncatingIfNeeded: value))
        append(UInt8(truncatingIfNeeded: value >> 8))
        append(UInt8(truncatingIfNeeded: value >> 16))
        append(UInt8(truncatingIfNeeded: value >> 24))
    }
}
