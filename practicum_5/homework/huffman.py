import heapq
import numpy as np


class HuffmanCoding:
    def __init__(self) -> None:
        self.codes = {}
        self.root = None

    def encode(self, sequence: list) -> str:
        freq = {}
        for symbol in sequence:
            freq[symbol] = freq.get(symbol, 0) + 1
        
        heap = []
        for symbol, count in freq.items():
            heapq.heappush(heap, (count, symbol))
        
        while len(heap) > 1:
            count1, symbol1 = heapq.heappop(heap)
            count2, symbol2 = heapq.heappop(heap)
            new_count = count1 + count2
            new_symbol = (symbol1, symbol2)
            heapq.heappush(heap, (new_count, new_symbol))
        
        def generate_codes(node, code=''):
            if isinstance(node, tuple):
                left, right = node
                generate_codes(left, code + '0')
                generate_codes(right, code + '1')
            else:
                self.codes[node] = code or '0'
        
        if heap:
            _, root = heap[0]
            self.root = root
            generate_codes(root)
        
        return ''.join(self.codes[symbol] for symbol in sequence)
        
    def decode(self, encoded_sequence: str) -> list:
        decoded = []
        current_node = self.root
        for bit in encoded_sequence:
            if bit == '0':
                current_node = current_node[0]
            else:
                current_node = current_node[1]
            
            if not isinstance(current_node, tuple):
                decoded.append(current_node)
                current_node = self.root
        
        return decoded


class LossyCompression:
    def __init__(self) -> None:
        self.levels = 16
        self.bins = None
        self.huffman = HuffmanCoding()

    def compress(self, time_series: np.ndarray) -> str:
        min_val = np.min(time_series)
        max_val = np.max(time_series)
        self.bins = np.linspace(min_val, max_val, self.levels + 1)
        quantized = np.digitize(time_series, self.bins) - 1
        
        return self.huffman.encode(quantized.tolist())

    def decompress(self, bits: str) -> np.ndarray:
        symbols = self.huffman.decode(bits)
        
        centers = (self.bins[:-1] + self.bins[1:]) / 2
        return np.array([centers[s] for s in symbols])


if __name__ == "__main__":
    ts = np.loadtxt("pershin/spbu-fundamentals-of-algorithms/ts_homework_practicum_5.txt")

    compressor = LossyCompression()
    bits = compressor.compress(ts)
    decompressed_ts = compressor.decompress(bits)

    compression_ratio = (len(ts) * 32 * 8) / len(bits)
    print(f"Compression ratio: {compression_ratio:.2f}")

    compression_loss = np.sqrt(np.mean((ts - decompressed_ts)**2))
    print(f"Compression loss (RMSE): {compression_loss}")
