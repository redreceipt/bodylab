// Apple's Vision person-instance masks (macOS 14+, on the Neural Engine), for app.py: built on first use into
// models/person_mask. Reads frames from stdin (uint32 width, uint32 height, then BGRA bytes); writes per frame a uint32
// count of people, then one 8-bit mask at the frame's size per person.
import CoreGraphics
import CoreVideo
import Foundation
import Vision

let input = FileHandle.standardInput, output = FileHandle.standardOutput

func read(_ n: Int) -> Data? {
    var d = Data()
    while d.count < n {
        let chunk = input.readData(ofLength: n - d.count)
        if chunk.isEmpty { return nil }
        d.append(chunk)
    }
    return d
}

func bytes(_ buf: CVPixelBuffer, _ w: Int, _ h: Int) -> Data {
    CVPixelBufferLockBaseAddress(buf, .readOnly)
    defer { CVPixelBufferUnlockBaseAddress(buf, .readOnly) }
    let bw = CVPixelBufferGetWidth(buf), bh = CVPixelBufferGetHeight(buf), row = CVPixelBufferGetBytesPerRow(buf)
    let base = CVPixelBufferGetBaseAddress(buf)!
    let float = CVPixelBufferGetPixelFormatType(buf) == kCVPixelFormatType_OneComponent32Float
    var out = Data(count: w * h)
    out.withUnsafeMutableBytes { (o: UnsafeMutableRawBufferPointer) in
        for y in 0..<h {
            let src = base + row * min(y * bh / h, bh - 1)
            for x in 0..<w {
                let sx = min(x * bw / w, bw - 1)  // nearest, if Vision's size differs from the frame's
                o[y * w + x] = float ? UInt8(max(0, min(1, src.load(fromByteOffset: sx * 4, as: Float32.self))) * 255)
                                     : src.load(fromByteOffset: sx, as: UInt8.self)
            }
        }
    }
    return out
}

while let head = read(8) {
    let w = Int(head.withUnsafeBytes { $0.load(fromByteOffset: 0, as: UInt32.self) })
    let h = Int(head.withUnsafeBytes { $0.load(fromByteOffset: 4, as: UInt32.self) })
    guard let px = read(w * h * 4) else { break }
    var masks: [Data] = []
    if let cg = CGImage(width: w, height: h, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: w * 4,
                        space: CGColorSpaceCreateDeviceRGB(),
                        bitmapInfo: CGBitmapInfo(rawValue: CGBitmapInfo.byteOrder32Little.rawValue | CGImageAlphaInfo.noneSkipFirst.rawValue),
                        provider: CGDataProvider(data: px as CFData)!, decode: nil, shouldInterpolate: false, intent: .defaultIntent) {
        let handler = VNImageRequestHandler(cgImage: cg)
        let req = VNGeneratePersonInstanceMaskRequest()
        if (try? handler.perform([req])) != nil, let r = req.results?.first {
            for i in r.allInstances {
                if let buf = try? r.generateScaledMaskForImage(forInstances: [i], from: handler) { masks.append(bytes(buf, w, h)) }
            }
        }
    }
    var n = UInt32(masks.count)
    output.write(Data(bytes: &n, count: 4))
    masks.forEach { output.write($0) }
}
