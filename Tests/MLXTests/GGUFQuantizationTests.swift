import Foundation
import MLX
import XCTest

@testable import MLX

final class GGUFQuantizationTests: XCTestCase {
    func testQ4KConversionProducesMLXAffineShapes() throws {
        let data = makeQ4KBlock()

        XCTAssertEqual(try ggufKDataByteCount(format: .q4K, shape: [1, 256]), 144)

        let converted = try ggufKToAffineQuantizedBuffers(data, format: .q4K, shape: [1, 256])

        XCTAssertEqual(converted.weightShape, [1, 32])
        XCTAssertEqual(converted.scalesShape, [1, 8])
        XCTAssertEqual(converted.groupSize, 32)
        XCTAssertEqual(converted.bits, 4)
        XCTAssertEqual(converted.weightData.count, 128)
        XCTAssertEqual(converted.scalesData.count, 16)
        XCTAssertEqual(converted.biasesData.count, 16)
    }

    func testQ4KConversionUnpacksScalesMinsAndRepackagesGroups() throws {
        let data = makeQ4KBlock()

        let converted = try ggufKToAffineQuantizedBuffers(data, format: .q4K, shape: [1, 256])

        let packedGroup =
            Data([0x10, 0x32, 0x54, 0x76, 0x98, 0xba, 0xdc, 0xfe])
            + Data([0x10, 0x32, 0x54, 0x76, 0x98, 0xba, 0xdc, 0xfe])
        XCTAssertEqual(converted.weightData.prefix(16), packedGroup)
        XCTAssertEqual(
            converted.weightData.dropFirst(16).prefix(16),
            packedGroup)
        XCTAssertEqual(float16Values(converted.scalesData), [2, 4, 6, 8, 10, 12, 14, 16])
        XCTAssertEqual(
            float16Values(converted.biasesData), [-18, -20, -22, -24, -26, -28, -30, -32])
    }

    func testQ4KConversionDequantizesToGGUFReferenceValues() throws {
        let data = makeQ4KBlock()
        let converted = try ggufKToAffineQuantizedArrays(data, format: .q4K, shape: [1, 256])

        let actual = dequantized(
            converted.weight,
            scales: converted.scales,
            biases: converted.biases,
            groupSize: converted.groupSize,
            bits: converted.bits,
            dtype: .float32)
        eval(actual)

        XCTAssertEqual(actual.asArray(Float.self), referenceQ4KDequantized(data))
    }

    func testQ6KConversionDequantizesToGGUFReferenceValues() throws {
        let data = makeQ6KBlock()
        let converted = try ggufKToAffineQuantizedArrays(data, format: .q6K, shape: [1, 256])

        let actual = dequantized(
            converted.weight,
            scales: converted.scales,
            biases: converted.biases,
            groupSize: converted.groupSize,
            bits: converted.bits,
            dtype: .float32)
        eval(actual)

        XCTAssertEqual(actual.asArray(Float.self), referenceQ6KDequantized(data))
    }

    func testRemainingKFormatByteCountsAndShapes() throws {
        XCTAssertEqual(try ggufKDataByteCount(format: .q8_0, shape: [1, 256]), 272)
        XCTAssertEqual(try ggufKDataByteCount(format: .q5K, shape: [1, 256]), 176)
        XCTAssertEqual(try ggufKDataByteCount(format: .q2K, shape: [1, 256]), 84)
        XCTAssertEqual(try ggufKDataByteCount(format: .q3K, shape: [1, 256]), 110)
        XCTAssertEqual(try ggufKDataByteCount(format: .q8K, shape: [1, 256]), 292)

        let q8_0 = try ggufKToAffineQuantizedBuffers(
            makeQ8_0Blocks(seed: 0), format: .q8_0, shape: [1, 256])
        XCTAssertEqual(q8_0.weightShape, [1, 64])
        XCTAssertEqual(q8_0.scalesShape, [1, 8])
        XCTAssertEqual(q8_0.groupSize, 32)
        XCTAssertEqual(q8_0.bits, 8)

        let q5 = try ggufKToAffineQuantizedBuffers(
            makeQ5KBlock(seed: 0), format: .q5K, shape: [1, 256])
        XCTAssertEqual(q5.weightShape, [1, 40])
        XCTAssertEqual(q5.scalesShape, [1, 8])
        XCTAssertEqual(q5.groupSize, 32)
        XCTAssertEqual(q5.bits, 5)

        let q2 = try ggufKToAffineQuantizedBuffers(
            makeQ2KBlock(seed: 0), format: .q2K, shape: [1, 256])
        XCTAssertEqual(q2.weightShape, [1, 16])
        XCTAssertEqual(q2.scalesShape, [1, 16])
        XCTAssertEqual(q2.groupSize, 16)
        XCTAssertEqual(q2.bits, 2)

        let q3 = try ggufKToAffineQuantizedBuffers(
            makeQ3KBlock(seed: 0), format: .q3K, shape: [1, 256])
        XCTAssertEqual(q3.weightShape, [1, 24])
        XCTAssertEqual(q3.scalesShape, [1, 16])
        XCTAssertEqual(q3.groupSize, 16)
        XCTAssertEqual(q3.bits, 3)

        let q8 = try ggufKToAffineQuantizedBuffers(
            makeQ8KBlock(seed: 0), format: .q8K, shape: [1, 256])
        XCTAssertEqual(q8.weightShape, [1, 64])
        XCTAssertEqual(q8.scalesShape, [1, 16])
        XCTAssertEqual(q8.groupSize, 16)
        XCTAssertEqual(q8.bits, 8)
    }

    func testRemainingKFormatConversionsRejectIncompatibleInputs() {
        for format in [GGUFKQuantization.q8_0, .q5K, .q2K, .q3K, .q8K] {
            XCTAssertThrowsError(try ggufKToAffineQuantizedBuffers(Data(), format: format, shape: [256]))
            XCTAssertThrowsError(try ggufKToAffineQuantizedBuffers(Data(), format: format, shape: [1, 128]))
            XCTAssertThrowsError(try ggufKToAffineQuantizedBuffers(Data(), format: format, shape: [1, 256]))
        }
    }

    func testRemainingKFormatConversionsDequantizeToGGUFReferenceValues() throws {
        let q5 = makeQ5KBlock(seed: 0)
        XCTAssertEqual(
            try dequantizedValues(ggufKToAffineQuantizedArrays(q5, format: .q5K, shape: [1, 256])),
            referenceQ5KDequantized(q5))

        let q2 = makeQ2KBlock(seed: 1)
        XCTAssertEqual(
            try dequantizedValues(ggufKToAffineQuantizedArrays(q2, format: .q2K, shape: [1, 256])),
            referenceQ2KDequantized(q2))

        let q3 = makeQ3KBlock(seed: 2)
        XCTAssertEqual(
            try dequantizedValues(ggufKToAffineQuantizedArrays(q3, format: .q3K, shape: [1, 256])),
            referenceQ3KDequantized(q3))

        let q8 = makeQ8KBlock(seed: 3)
        XCTAssertEqual(
            try dequantizedValues(ggufKToAffineQuantizedArrays(q8, format: .q8K, shape: [1, 256])),
            referenceQ8KDequantized(q8))
    }

    func testQ5KAndQ3KConversionsHandleMultipleRowsAndBlocks() throws {
        let q5 =
            makeQ5KBlock(seed: 0) + makeQ5KBlock(seed: 3) + makeQ5KBlock(seed: 6)
            + makeQ5KBlock(seed: 9)
        XCTAssertEqual(
            try dequantizedValues(ggufKToAffineQuantizedArrays(q5, format: .q5K, shape: [2, 512])),
            referenceQ5KDequantized(q5))

        let q3 =
            makeQ3KBlock(seed: 0) + makeQ3KBlock(seed: 5) + makeQ3KBlock(seed: 10)
            + makeQ3KBlock(seed: 15)
        XCTAssertEqual(
            try dequantizedValues(ggufKToAffineQuantizedArrays(q3, format: .q3K, shape: [2, 512])),
            referenceQ3KDequantized(q3))
    }

    func testQ4KBufferConversionAcceptsSliceWithoutCopying() throws {
        var data = Data([0xaa, 0xbb])
        data.append(makeQ4KBlock())
        data.append(contentsOf: [0xcc, 0xdd])

        let converted = try data.withUnsafeBytes { rawBuffer in
            let bytes = rawBuffer.bindMemory(to: UInt8.self)
            return try ggufKToAffineQuantizedBuffers(
                UnsafeBufferPointer(rebasing: bytes[2 ..< (2 + 144)]),
                format: .q4K,
                shape: [1, 256])
        }

        XCTAssertEqual(converted.weightShape, [1, 32])
        XCTAssertEqual(converted.scalesShape, [1, 8])
        XCTAssertEqual(converted.weightData.count, 128)
        XCTAssertEqual(float16Values(converted.scalesData), [2, 4, 6, 8, 10, 12, 14, 16])
    }

    func testQ4KConversionRejectsIncompatibleInputs() {
        XCTAssertThrowsError(try ggufKToAffineQuantizedBuffers(Data(), format: .q4K, shape: [256]))
        XCTAssertThrowsError(try ggufKToAffineQuantizedBuffers(Data(), format: .q4K, shape: [1, 128]))
        XCTAssertThrowsError(try ggufKToAffineQuantizedBuffers(Data(), format: .q4K, shape: [1, 256]))
    }

    private func makeQ4KBlock() -> Data {
        var data = Data()
        data.appendLittleEndian(Float16(2).bitPattern)
        data.appendLittleEndian(Float16(2).bitPattern)

        var scales = [UInt8](repeating: 0, count: 12)
        for group in 0 ..< 8 {
            writeScaleMin(
                group: group, scale: UInt8(group + 1), minimum: UInt8(group + 9), into: &scales)
        }
        data.append(contentsOf: scales)

        for pair in 0 ..< 4 {
            for value in 0 ..< 32 {
                let low = UInt8(value % 16)
                let high = UInt8((value + pair) % 16)
                data.append(low | (high << 4))
            }
        }

        XCTAssertEqual(data.count, 144)
        return data
    }

    private func makeQ6KBlock() -> Data {
        let values = (0 ..< 256).map { UInt8($0 % 64) }
        var data = Data(count: 210)

        data.withUnsafeMutableBytes { rawBuffer in
            let bytes = rawBuffer.bindMemory(to: UInt8.self)

            for half in 0 ..< 2 {
                let valueOffset = half * 128
                let qlOffset = half * 64
                let qhOffset = 128 + half * 32

                for l in 0 ..< 32 {
                    let q1 = values[valueOffset + l]
                    let q2 = values[valueOffset + l + 32]
                    let q3 = values[valueOffset + l + 64]
                    let q4 = values[valueOffset + l + 96]
                    bytes[qlOffset + l] = (q1 & 0x0f) | ((q3 & 0x0f) << 4)
                    bytes[qlOffset + l + 32] = (q2 & 0x0f) | ((q4 & 0x0f) << 4)
                    bytes[qhOffset + l] =
                        ((q1 >> 4) & 0x03)
                        | (((q2 >> 4) & 0x03) << 2)
                        | (((q3 >> 4) & 0x03) << 4)
                        | (((q4 >> 4) & 0x03) << 6)
                }
            }

            for group in 0 ..< 16 {
                bytes[192 + group] = UInt8(bitPattern: Int8(group - 8))
            }
            let d = Float16(0.5).bitPattern
            bytes[208] = UInt8(truncatingIfNeeded: d)
            bytes[209] = UInt8(truncatingIfNeeded: d >> 8)
        }

        XCTAssertEqual(data.count, 210)
        _ = values
        return data
    }

    private func makeQ5KBlock(seed: Int) -> Data {
        var data = Data(count: 176)
        data.withUnsafeMutableBytes { rawBuffer in
            let bytes = rawBuffer.bindMemory(to: UInt8.self)
            writeFloat16(0.25, to: bytes, at: 0)
            writeFloat16(0.5, to: bytes, at: 2)
            for group in 0 ..< 8 {
                writeScaleMin(
                    group: group,
                    scale: UInt8((group + seed) % 63 + 1),
                    minimum: UInt8((group * 3 + seed) % 63),
                    into: bytes,
                    at: 4)
            }
            for pair in 0 ..< 4 {
                for l in 0 ..< 32 {
                    let low = UInt8((pair * 64 + l + seed) % 32)
                    let high = UInt8((pair * 64 + 32 + l + seed) % 32)
                    bytes[48 + pair * 32 + l] = (low & 0x0f) | ((high & 0x0f) << 4)
                    if low >= 16 {
                        bytes[16 + l] |= UInt8(1 << (pair * 2))
                    }
                    if high >= 16 {
                        bytes[16 + l] |= UInt8(1 << (pair * 2 + 1))
                    }
                }
            }
        }
        return data
    }

    private func makeQ8_0Blocks(seed: Int) -> Data {
        var data = Data()
        for block in 0 ..< 8 {
            data.appendLittleEndian(Float16(Float(block + 1) * 0.125).bitPattern)
            for index in 0 ..< 32 {
                data.append(UInt8(bitPattern: Int8((index + block + seed) % 127 - 63)))
            }
        }
        return data
    }

    private func makeQ2KBlock(seed: Int) -> Data {
        var data = Data(count: 84)
        data.withUnsafeMutableBytes { rawBuffer in
            let bytes = rawBuffer.bindMemory(to: UInt8.self)
            for group in 0 ..< 16 {
                bytes[group] = UInt8(((group + seed) % 15 + 1) | (((group * 2 + seed) % 15) << 4))
            }
            for half in 0 ..< 2 {
                for j in 0 ..< 4 {
                    let shift = j * 2
                    for l in 0 ..< 16 {
                        let group0 = half * 8 + j * 2
                        let group1 = group0 + 1
                        bytes[16 + half * 32 + l] |= UInt8((l + group0 + seed) % 4) << shift
                        bytes[16 + half * 32 + 16 + l] |= UInt8((l + group1 + seed) % 4) << shift
                    }
                }
            }
            writeFloat16(0.75, to: bytes, at: 80)
            writeFloat16(0.25, to: bytes, at: 82)
        }
        return data
    }

    private func makeQ3KBlock(seed: Int) -> Data {
        var data = Data(count: 110)
        data.withUnsafeMutableBytes { rawBuffer in
            let bytes = rawBuffer.bindMemory(to: UInt8.self)
            for group in 0 ..< 16 {
                writeQ3KScale(UInt8((group * 5 + seed) % 63 + 1), group: group, into: bytes, at: 96)
            }
            for group in 0 ..< 16 {
                let shift = (group / 2) % 4 * 2
                let qsBase = 32 + (group / 8) * 32 + ((group % 8) / 2) * 16
                let mask = UInt8(1 << (group / 2))
                let maskOffset = group % 2 == 0 ? 0 : 16
                for l in 0 ..< 16 {
                    let value = Int8((l + group + seed) % 8) - 4
                    let unsigned = UInt8(bitPattern: value + 4)
                    bytes[qsBase + l] |= (unsigned & 0x03) << shift
                    if value >= 0 {
                        bytes[maskOffset + l] |= mask
                    }
                }
            }
            writeFloat16(0.125, to: bytes, at: 108)
        }
        return data
    }

    private func makeQ8KBlock(seed: Int) -> Data {
        var data = Data(count: 292)
        data.withUnsafeMutableBytes { rawBuffer in
            let bytes = rawBuffer.bindMemory(to: UInt8.self)
            writeFloat32(0.125, to: bytes, at: 0)
            for index in 0 ..< 256 {
                bytes[4 + index] = UInt8(bitPattern: Int8((index + seed) % 127 - 63))
            }
        }
        return data
    }

    private func referenceQ4KDequantized(_ data: Data) -> [Float] {
        let d = Float(float16Value(data, at: 0))
        let dmin = Float(float16Value(data, at: 2))
        let scalesOffset = 4
        let quantsOffset = 16
        var output = [Float]()
        output.reserveCapacity(256)

        for groupPair in 0 ..< 4 {
            let pairOffset = quantsOffset + groupPair * 32
            let (scale1, min1) = readScaleMin(
                data, scalesOffset: scalesOffset, group: groupPair * 2)
            let (scale2, min2) = readScaleMin(
                data, scalesOffset: scalesOffset, group: groupPair * 2 + 1)

            for l in 0 ..< 32 {
                output.append(
                    d * Float(scale1) * Float(data[pairOffset + l] & 0x0f) - dmin * Float(min1))
            }
            for l in 0 ..< 32 {
                output.append(
                    d * Float(scale2) * Float(data[pairOffset + l] >> 4) - dmin * Float(min2))
            }
        }

        return output
    }

    private func referenceQ6KDequantized(_ data: Data) -> [Float] {
        let d = Float(float16Value(data, at: 208))
        var output = [Float](repeating: 0, count: 256)

        for half in 0 ..< 2 {
            let outputOffset = half * 128
            let qlOffset = half * 64
            let qhOffset = 128 + half * 32
            let scalesOffset = 192 + half * 8

            for l in 0 ..< 32 {
                let iscale = l / 16
                let q1 =
                    Int((data[qlOffset + l] & 0x0f) | (((data[qhOffset + l] >> 0) & 0x03) << 4))
                    - 32
                let q2 =
                    Int(
                        (data[qlOffset + l + 32] & 0x0f) | (((data[qhOffset + l] >> 2) & 0x03) << 4)
                    ) - 32
                let q3 =
                    Int((data[qlOffset + l] >> 4) | (((data[qhOffset + l] >> 4) & 0x03) << 4)) - 32
                let q4 =
                    Int((data[qlOffset + l + 32] >> 4) | (((data[qhOffset + l] >> 6) & 0x03) << 4))
                    - 32
                output[outputOffset + l] =
                    d * Float(Int8(bitPattern: data[scalesOffset + iscale + 0])) * Float(q1)
                output[outputOffset + l + 32] =
                    d * Float(Int8(bitPattern: data[scalesOffset + iscale + 2])) * Float(q2)
                output[outputOffset + l + 64] =
                    d * Float(Int8(bitPattern: data[scalesOffset + iscale + 4])) * Float(q3)
                output[outputOffset + l + 96] =
                    d * Float(Int8(bitPattern: data[scalesOffset + iscale + 6])) * Float(q4)
            }
        }

        return output
    }

    private func referenceQ5KDequantized(_ data: Data) -> [Float] {
        var output = [Float]()
        output.reserveCapacity(data.count / 176 * 256)
        for blockOffset in stride(from: 0, to: data.count, by: 176) {
            let d = Float(float16Value(data, at: blockOffset))
            let dmin = Float(float16Value(data, at: blockOffset + 2))
            var iscale = 0
            var u1: UInt8 = 1
            var u2: UInt8 = 2
            for j in stride(from: 0, to: 256, by: 64) {
                let (scale1, min1) = readScaleMin(
                    data, scalesOffset: blockOffset + 4, group: iscale)
                let (scale2, min2) = readScaleMin(
                    data, scalesOffset: blockOffset + 4, group: iscale + 1)
                for l in 0 ..< 32 {
                    let q =
                        (data[blockOffset + 48 + j / 2 + l] & 0x0f)
                        + ((data[blockOffset + 16 + l] & u1) != 0 ? 16 : 0)
                    output.append(d * Float(scale1) * Float(q) - dmin * Float(min1))
                }
                for l in 0 ..< 32 {
                    let q =
                        (data[blockOffset + 48 + j / 2 + l] >> 4)
                        + ((data[blockOffset + 16 + l] & u2) != 0 ? 16 : 0)
                    output.append(d * Float(scale2) * Float(q) - dmin * Float(min2))
                }
                iscale += 2
                u1 <<= 2
                u2 <<= 2
            }
        }
        return output
    }

    private func referenceQ2KDequantized(_ data: Data) -> [Float] {
        var output = [Float]()
        output.reserveCapacity(data.count / 84 * 256)
        for blockOffset in stride(from: 0, to: data.count, by: 84) {
            let d = Float(float16Value(data, at: blockOffset + 80))
            let dmin = Float(float16Value(data, at: blockOffset + 82))
            var iscale = 0
            var qOffset = blockOffset + 16
            for _ in 0 ..< 2 {
                var shift = 0
                for _ in 0 ..< 4 {
                    var sc = data[blockOffset + iscale]
                    var scale = d * Float(sc & 0x0f)
                    var minimum = dmin * Float(sc >> 4)
                    for l in 0 ..< 16 {
                        output.append(scale * Float((data[qOffset + l] >> shift) & 0x03) - minimum)
                    }
                    iscale += 1
                    sc = data[blockOffset + iscale]
                    scale = d * Float(sc & 0x0f)
                    minimum = dmin * Float(sc >> 4)
                    for l in 0 ..< 16 {
                        output.append(
                            scale * Float((data[qOffset + 16 + l] >> shift) & 0x03) - minimum)
                    }
                    iscale += 1
                    shift += 2
                }
                qOffset += 32
            }
        }
        return output
    }

    private func referenceQ3KDequantized(_ data: Data) -> [Float] {
        var output = [Float]()
        output.reserveCapacity(data.count / 110 * 256)
        for blockOffset in stride(from: 0, to: data.count, by: 110) {
            let d = Float(float16Value(data, at: blockOffset + 108))
            var iscale = 0
            var mask: UInt8 = 1
            for half in 0 ..< 2 {
                var shift = 0
                for _ in 0 ..< 4 {
                    let scale1 =
                        d
                        * Float(
                            Int(readQ3KScale(data, scalesOffset: blockOffset + 96, group: iscale))
                                - 32)
                    for l in 0 ..< 16 {
                        let q =
                            Int((data[blockOffset + 32 + half * 32 + l] >> shift) & 0x03)
                            - ((data[blockOffset + half * 16 + l] & mask) != 0 ? 0 : 4)
                        output.append(scale1 * Float(q))
                    }
                    iscale += 1
                    let scale2 =
                        d
                        * Float(
                            Int(readQ3KScale(data, scalesOffset: blockOffset + 96, group: iscale))
                                - 32)
                    for l in 0 ..< 16 {
                        let q =
                            Int((data[blockOffset + 32 + half * 32 + 16 + l] >> shift) & 0x03)
                            - ((data[blockOffset + half * 16 + 16 + l] & mask) != 0 ? 0 : 4)
                        output.append(scale2 * Float(q))
                    }
                    iscale += 1
                    shift += 2
                    mask <<= 1
                }
            }
        }
        return output
    }

    private func referenceQ8KDequantized(_ data: Data) -> [Float] {
        var output = [Float]()
        output.reserveCapacity(data.count / 292 * 256)
        for blockOffset in stride(from: 0, to: data.count, by: 292) {
            let d = float32Value(data, at: blockOffset)
            for index in 0 ..< 256 {
                output.append(d * Float(Int8(bitPattern: data[blockOffset + 4 + index])))
            }
        }
        return output
    }

    private func dequantizedValues(_ arrays: GGUFAffineQuantizedArrays) throws -> [Float] {
        let actual = dequantized(
            arrays.weight,
            scales: arrays.scales,
            biases: arrays.biases,
            groupSize: arrays.groupSize,
            bits: arrays.bits,
            dtype: .float32)
        eval(actual)
        return actual.asArray(Float.self)
    }

    private func writeScaleMin(group: Int, scale: UInt8, minimum: UInt8, into bytes: inout [UInt8])
    {
        if group < 4 {
            bytes[group] = (bytes[group] & 0xc0) | (scale & 63)
            bytes[group + 4] = (bytes[group + 4] & 0xc0) | (minimum & 63)
        } else {
            bytes[group + 4] = (scale & 0x0f) | ((minimum & 0x0f) << 4)
            bytes[group - 4] = (bytes[group - 4] & 0x3f) | ((scale >> 4) << 6)
            bytes[group] = (bytes[group] & 0x3f) | ((minimum >> 4) << 6)
        }
    }

    private func writeScaleMin(
        group: Int,
        scale: UInt8,
        minimum: UInt8,
        into bytes: UnsafeMutableBufferPointer<UInt8>,
        at scalesOffset: Int
    ) {
        if group < 4 {
            bytes[scalesOffset + group] = (bytes[scalesOffset + group] & 0xc0) | (scale & 63)
            bytes[scalesOffset + group + 4] =
                (bytes[scalesOffset + group + 4] & 0xc0) | (minimum & 63)
        } else {
            bytes[scalesOffset + group + 4] = (scale & 0x0f) | ((minimum & 0x0f) << 4)
            bytes[scalesOffset + group - 4] =
                (bytes[scalesOffset + group - 4] & 0x3f) | ((scale >> 4) << 6)
            bytes[scalesOffset + group] =
                (bytes[scalesOffset + group] & 0x3f) | ((minimum >> 4) << 6)
        }
    }

    private func writeQ3KScale(
        _ scale: UInt8,
        group: Int,
        into bytes: UnsafeMutableBufferPointer<UInt8>,
        at scalesOffset: Int
    ) {
        if group < 8 {
            bytes[scalesOffset + group] = (bytes[scalesOffset + group] & 0xf0) | (scale & 0x0f)
        } else {
            bytes[scalesOffset + group - 8] =
                (bytes[scalesOffset + group - 8] & 0x0f) | ((scale & 0x0f) << 4)
        }
        bytes[scalesOffset + 8 + group % 4] |= ((scale >> 4) & 0x03) << UInt8((group / 4) * 2)
    }

    private func readQ3KScale(_ data: Data, scalesOffset: Int, group: Int) -> UInt8 {
        let low: UInt8
        if group < 8 {
            low = data[scalesOffset + group] & 0x0f
        } else {
            low = data[scalesOffset + group - 8] >> 4
        }
        let high = (data[scalesOffset + 8 + group % 4] >> UInt8((group / 4) * 2)) & 0x03
        return low | (high << 4)
    }

    private func float16Values(_ data: Data) -> [Float] {
        stride(from: 0, to: data.count, by: 2).map { offset in
            let low = UInt16(data[offset])
            let high = UInt16(data[offset + 1]) << 8
            return Float(Float16(bitPattern: low | high))
        }
    }

    private func float16Value(_ data: Data, at offset: Int) -> Float16 {
        let low = UInt16(data[offset])
        let high = UInt16(data[offset + 1]) << 8
        return Float16(bitPattern: low | high)
    }

    private func float32Value(_ data: Data, at offset: Int) -> Float {
        let b0 = UInt32(data[offset])
        let b1 = UInt32(data[offset + 1]) << 8
        let b2 = UInt32(data[offset + 2]) << 16
        let b3 = UInt32(data[offset + 3]) << 24
        return Float(bitPattern: b0 | b1 | b2 | b3)
    }

    private func readScaleMin(_ data: Data, scalesOffset: Int, group: Int) -> (UInt8, UInt8) {
        if group < 4 {
            return (
                data[scalesOffset + group] & 63,
                data[scalesOffset + group + 4] & 63
            )
        }
        return (
            (data[scalesOffset + group + 4] & 0x0f)
                | ((data[scalesOffset + group - 4] >> 6) << 4),
            (data[scalesOffset + group + 4] >> 4)
                | ((data[scalesOffset + group] >> 6) << 4)
        )
    }
}

extension Data {
    fileprivate mutating func appendLittleEndian(_ value: UInt16) {
        append(UInt8(truncatingIfNeeded: value))
        append(UInt8(truncatingIfNeeded: value >> 8))
    }
}

private func writeFloat16(
    _ value: Float16,
    to bytes: UnsafeMutableBufferPointer<UInt8>,
    at offset: Int
) {
    let bitPattern = value.bitPattern
    bytes[offset] = UInt8(truncatingIfNeeded: bitPattern)
    bytes[offset + 1] = UInt8(truncatingIfNeeded: bitPattern >> 8)
}

private func writeFloat32(
    _ value: Float,
    to bytes: UnsafeMutableBufferPointer<UInt8>,
    at offset: Int
) {
    let bitPattern = value.bitPattern
    bytes[offset] = UInt8(truncatingIfNeeded: bitPattern)
    bytes[offset + 1] = UInt8(truncatingIfNeeded: bitPattern >> 8)
    bytes[offset + 2] = UInt8(truncatingIfNeeded: bitPattern >> 16)
    bytes[offset + 3] = UInt8(truncatingIfNeeded: bitPattern >> 24)
}
