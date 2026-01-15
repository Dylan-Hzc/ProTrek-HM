import numpy as np
import sys
from numba import jit
from .step3 import UngappedIndexer

@jit(nopython=True, cache=True)
def _numba_smith_waterman(s1, s2, match, mismatch, gap):
    n = len(s1)
    m = len(s2)

    H = np.zeros((n + 1, m + 1), dtype=np.int32)
    max_score = 0
    
    for i in range(1, n + 1):
        char1 = s1[i-1]
        for j in range(1, m + 1):
            char2 = s2[j-1]

            if char1 == char2:
                diag = H[i-1, j-1] + match
            else:
                diag = H[i-1, j-1] + mismatch

            up = H[i-1, j] + gap
            left = H[i, j-1] + gap

            score = diag
            if up > score: score = up
            if left > score: score = left
            if score < 0: score = 0
            
            H[i, j] = score
            
            if score > max_score:
                max_score = score
                
    return max_score

@jit(nopython=True, cache=True)
def _numba_ungapped_process(qs, ts, q_enc, t_enc, k, match, mismatch, drop_x):
    sort_idx = np.argsort(qs)
    qs_sorted = qs[sort_idx]
    ts_sorted = ts[sort_idx]

    q_len = len(q_enc)
    t_len = len(t_enc)
    
    best_total_score = -999999
    last_q_end = -k

    n_seeds = len(qs_sorted)

    for i in range(n_seeds):
        q_start = qs_sorted[i]
        t_start = ts_sorted[i]

        if q_start <= last_q_end:
            continue
        
        current_score = 0
        valid_seed = True
        for j in range(k):
            if q_start + j >= q_len or t_start + j >= t_len:
                valid_seed = False
                break
            if q_enc[q_start + j] == t_enc[t_start + j]:
                current_score += match
            else:
                current_score += mismatch
        
        if not valid_seed: continue

        best_l = 0
        curr_l = 0
        off = 1
        while q_start - off >= 0 and t_start - off >= 0:
            if q_enc[q_start - off] == t_enc[t_start - off]:
                curr_l += match
            else:
                curr_l += mismatch
            
            if curr_l > best_l: best_l = curr_l
            if curr_l < best_l - drop_x: break
            off += 1

        best_r = 0
        curr_r = 0
        r = 0
        while q_start + k + r < q_len and t_start + k + r < t_len:
            if q_enc[q_start + k + r] == t_enc[t_start + k + r]:
                curr_r += match
            else:
                curr_r += mismatch
            
            if curr_r > best_r: best_r = curr_r
            if curr_r < best_r - drop_x: break
            r += 1

        total = current_score + best_l + best_r
        
        if total > best_total_score:
            best_total_score = total

        last_q_end = q_start + r

    return best_total_score

class GappedIndexer(UngappedIndexer):
    """
    Step 5: Gapped Extension using Smith-Waterman.
    Optimized with Numba and Dict Lookups.
    """

    def __init__(self, k=5, dropoff_x=20, top_n_candidates=100, sw_top_n=50):
        super().__init__(k, dropoff_x, top_n_candidates)
        self.sw_top_n = sw_top_n
        self.gap_score = -5
        self._seq_cache = None

    def _ensure_seq_cache(self):
        if self._seq_cache is None and hasattr(self, 'proteins'):
            self._seq_cache = self.proteins["sequence"].to_dict()

    def search(self, query_seq):
        self._ensure_seq_cache()
        
        if len(query_seq) < self.k: return []

        q_enc = self._encode_sequence_to_uint8(query_seq)
        L = len(q_enc)
        bpa = self.bits_per_aa

        p_list, t_list, q_list = [], [], []
        code = 0
        for i in range(self.k): 
            code = (code << bpa) | int(q_enc[i])

        arr = self.index.get(code)
        if arr is not None: 
            p_list.append(arr[:, 0])
            t_list.append(arr[:, 1])
            q_list.append(np.zeros(len(arr), dtype=np.int32))

        for i in range(1, L - self.k + 1):
            v = int(q_enc[i + self.k - 1])
            code = ((code & self.mask) << bpa) | v
            arr = self.index.get(code)
            if arr is not None:
                p_list.append(arr[:, 0])
                t_list.append(arr[:, 1])
                q_list.append(np.full(len(arr), i, dtype=np.int32))

        if not p_list: return []
        
        p_all = np.concatenate(p_list)
        t_all = np.concatenate(t_list)
        q_all = np.concatenate(q_list)
        diags = q_all - t_all

        bias = np.int64(2_000_000_000)
        keys = (p_all.astype(np.int64) << 32) | (diags.astype(np.int64) + bias)
        uk, cnt = np.unique(keys, return_counts=True)
        
        order = np.argsort(cnt)[::-1]
        top_keys = order[:self.top_n_candidates]
        
        top_pids = (uk[top_keys] >> 32).astype(np.int32)
        top_diags = (uk[top_keys] & 0xFFFFFFFF).astype(np.int32) - bias

        results = []

        global_mask = np.isin(p_all, top_pids)
        p_cand_all = p_all[global_mask]
        t_cand_all = t_all[global_mask]
        q_cand_all = q_all[global_mask]
        d_cand_all = diags[global_mask]
        
        for pid, target_diag in zip(top_pids, top_diags):

            mask_p = (p_cand_all == pid) & (d_cand_all == target_diag)
            qs = q_cand_all[mask_p]
            ts = t_cand_all[mask_p]
            
            if len(qs) == 0: continue

            t_seq_str = self._seq_cache[pid]
            t_enc = self._encode_sequence_to_uint8(t_seq_str)

            best_s = _numba_ungapped_process(
                qs, ts,
                q_enc, t_enc,
                self.k, self.match_score, self.mismatch_score, self.dropoff_x
            )

            if best_s > -9999:
                results.append((pid, best_s))

        results.sort(key=lambda x: x[1], reverse=True)
        sw_candidates = results[:self.sw_top_n]
        
        final_results = []
        
        for pid, _ in sw_candidates:
            t_seq_str = self._seq_cache[pid]
            s1 = np.ascontiguousarray(q_enc, dtype=np.int32)
            s2 = np.ascontiguousarray(self._encode_sequence_to_uint8(t_seq_str), dtype=np.int32)

            sw_score = _numba_smith_waterman(
                s1, s2,
                self.match_score, self.mismatch_score, self.gap_score
            )
            final_results.append((pid, sw_score))
            
        final_results.sort(key=lambda x: x[1], reverse=True)
        
        return [p for p, _ in final_results]

    def _get_index_size_mb(self):
        size_bytes = 0
        if hasattr(self, 'index') and self.index:
            size_bytes += sys.getsizeof(self.index)
            for k, arr in self.index.items():
                size_bytes += sys.getsizeof(k)
                size_bytes += arr.nbytes 

        # Cache Size
        if self._seq_cache is not None:
            size_bytes += sys.getsizeof(self._seq_cache)
            for pid, seq in self._seq_cache.items():
                size_bytes += sys.getsizeof(pid)
                size_bytes += sys.getsizeof(seq)

        return size_bytes / (1024 * 1024)