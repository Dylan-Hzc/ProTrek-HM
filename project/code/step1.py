import numpy as np
import time
import sys
from numba import jit

@jit(nopython=True, cache=True)
def _numba_hash_only(seq_bytes, k, ascii_map, shifts):
    n = len(seq_bytes)
    if n < k:
        return np.empty(0, dtype=np.int64)

    codes = np.empty(n - k + 1, dtype=np.int64)
    count = 0

    for i in range(n - k + 1):
        code = 0
        valid = True

        for j in range(k):
            char_code = ascii_map[seq_bytes[i + j]]
            if char_code == 255:
                valid = False
                break
            code |= (np.int64(char_code) << shifts[j])
        
        if valid:
            codes[count] = code
            count += 1
            
    return codes[:count]

class KmerIndexer:
    """
    Step 1 :
    - Uses sliding_window_view for vectorized k-mer generation.
    - Bulk index construction using argsort to avoid loop overhead.
    """

    def __init__(self, k=5):
        self.k = k
        self.index = {}
        self.proteins = None

        aas = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        self.aa_map = {aa: i for i, aa in enumerate(aas)}

        self.bits_per_aa = 5
        self.shifts = np.arange(self.k - 1, -1, -1, dtype=np.int64) * self.bits_per_aa

        self._ascii_map = np.full(128, 255, dtype=np.uint8)
        for aa, code in self.aa_map.items():
            if ord(aa) < 128:
                self._ascii_map[ord(aa)] = np.uint8(code)

    def build_index(self, fasta_df):
        self.proteins = fasta_df

        all_codes = []
        all_pids = []

        ascii_map = self._ascii_map
        k = self.k
        shifts = self.shifts

        for row in fasta_df.itertuples():
            p_idx = row.Index
            seq = row.sequence
            
            seq_bytes = np.frombuffer(seq.encode("ascii"), dtype=np.uint8)
            valid_codes = _numba_hash_only(seq_bytes, k, ascii_map, shifts)
            
            if valid_codes.size > 0:
                all_codes.append(valid_codes)
                all_pids.append(np.full(valid_codes.size, p_idx, dtype=np.int32))

        if not all_codes:
            return

        flat_codes = np.concatenate(all_codes)
        flat_pids = np.concatenate(all_pids)

        sort_idx = np.argsort(flat_codes)
        sorted_codes = flat_codes[sort_idx]
        sorted_pids = flat_pids[sort_idx]
        
        mask = np.empty(len(sorted_codes), dtype=bool)
        mask[0] = True
        mask[1:] = sorted_codes[1:] != sorted_codes[:-1]
        
        unique_codes = sorted_codes[mask]
        split_indices = np.nonzero(mask)[0][1:]

        grouped_pids = np.split(sorted_pids, split_indices)

        self.index = dict(zip(unique_codes, grouped_pids))

    def search(self, query_seq):
        if len(query_seq) < self.k: return []

        seq_bytes = np.frombuffer(query_seq.encode("ascii"), dtype=np.uint8)
        valid_codes = _numba_hash_only(seq_bytes, self.k, self._ascii_map, self.shifts)
        
        if valid_codes.size == 0: return []

        found_arrays = []
        for c in valid_codes:
            hits = self.index.get(c)
            if hits is not None:
                found_arrays.append(hits)
                
        if not found_arrays:
            return []
            
        p_all = np.concatenate(found_arrays)
        counts = np.bincount(p_all, minlength=len(self.proteins))

        nonzero = np.nonzero(counts)[0]
        if nonzero.size == 0: return []

        order = np.argsort(counts[nonzero])[::-1]
        return nonzero[order].tolist()

    def evaluate(self, query_df, top_n=10):
        hits_at_1 = 0
        hits_at_5 = 0
        hits_at_10 = 0
        total_time = 0.0
        total_candidates = 0
        runtime_peak_mb = 0.0

        for i, row in enumerate(query_df.itertuples()):
            q_seq = row.sequence
            true_pid = row.origin_protein_id

            t0 = time.time()
            candidate_indices = self.search(q_seq)
            total_time += time.time() - t0

            top_indices = candidate_indices[:top_n]
            total_candidates += len(set(candidate_indices))
            predicted_pids = [self.proteins.at[idx, "protein_id"] for idx in top_indices]

            if predicted_pids:
                hits_at_1 += int(predicted_pids[0] == true_pid)
                hits_at_5 += int(true_pid in predicted_pids[:5])
                hits_at_10 += int(true_pid in predicted_pids)

        n = max(1, len(query_df))

        return {
            "recall@1": round(hits_at_1 / n, 4),
            "recall@5": round(hits_at_5 / n, 4),
            "recall@10": round(hits_at_10 / n, 4),
            "avg_time_ms": round((total_time / n) * 1000, 4),
            "avg_candidates": round(total_candidates / n, 2),
            "static_index_MB": round(self._get_index_size_mb(), 4),
            "runtime_peak_MB": round(runtime_peak_mb, 4),
            "index_size_keys": len(self.index),
            "k": self.k,
        }

    def _get_index_size_mb(self):
        size_bytes = sys.getsizeof(self.index)
        for k, arr in self.index.items():
            size_bytes += sys.getsizeof(k)
            size_bytes += arr.nbytes
        return size_bytes / (1024 * 1024)