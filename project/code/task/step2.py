import time
import numpy as np
from numba import jit
from .step1 import KmerIndexer

@jit(nopython=True, cache=True)
def _numba_encode_and_hash(seq_bytes, k, ascii_map, shifts):
    n = len(seq_bytes)
    if n < k:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int32)

    codes = np.empty(n - k + 1, dtype=np.int64)
    pos = np.empty(n - k + 1, dtype=np.int32)
    
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
            pos[count] = i
            count += 1
            
    return codes[:count], pos[:count]

@jit(nopython=True, cache=True)
def _numba_aggregate_diagonals(p_all, t_all, q_all):
    diags = q_all - t_all
    bias = 2_000_000_000

    keys = (p_all.astype(np.int64) << 32) | (diags.astype(np.int64) + bias)

    keys.sort()
    
    num_hits = len(keys)
    if num_hits == 0:
        return np.empty(0, dtype=np.int32)

    res_pids = np.empty(num_hits, dtype=np.int32)
    res_scores = np.empty(num_hits, dtype=np.int32)
    res_count = 0

    curr_key = keys[0]
    curr_p = curr_key >> 32
    
    diag_count = 1
    p_max_score = 0
    
    for i in range(1, num_hits):
        k = keys[i]
        
        if k == curr_key:
            diag_count += 1
        else:
            if diag_count > p_max_score:
                p_max_score = diag_count
            
            p = k >> 32
            
            if p != curr_p:
                res_pids[res_count] = curr_p
                res_scores[res_count] = p_max_score
                res_count += 1

                curr_p = p
                p_max_score = 0

            curr_key = k
            diag_count = 1

    if diag_count > p_max_score:
        p_max_score = diag_count
    res_pids[res_count] = curr_p
    res_scores[res_count] = p_max_score
    res_count += 1

    final_p = res_pids[:res_count]
    final_s = res_scores[:res_count]

    sort_idx = np.argsort(final_s, kind='mergesort') 
    
    out_p = np.empty(res_count, dtype=np.int32)
    for i in range(res_count):
        idx = sort_idx[res_count - 1 - i]
        out_p[i] = final_p[idx]
        
    return out_p

class DiagonalIndexer(KmerIndexer):
    """
    Step 2 :
    - Logic is identical to NumPy version (Diagonal Voting).
    - Uses JIT compiled functions for heavy loops and aggregation.
    """

    def __init__(self, k=5):
        super().__init__(k)
        self.index = {}
        self.shifts = self.shifts.astype(np.int64)

    def build_index(self, fasta_df):
        self.proteins = fasta_df

        all_codes = []
        all_pids = []
        all_tpos = []

        ascii_map = self._ascii_map
        k = self.k
        shifts = self.shifts

        for row in fasta_df.itertuples():
            p_idx = row.Index
            seq = row.sequence

            seq_bytes = np.frombuffer(seq.encode("ascii"), dtype=np.uint8)

            codes, t_indices = _numba_encode_and_hash(seq_bytes, k, ascii_map, shifts)
            
            if codes.size > 0:
                all_codes.append(codes)
                all_pids.append(np.full(codes.size, p_idx, dtype=np.int32))
                all_tpos.append(t_indices)

        if not all_codes:
            return

        flat_codes = np.concatenate(all_codes)
        flat_pids = np.concatenate(all_pids)
        flat_tpos = np.concatenate(all_tpos)

        sort_idx = np.argsort(flat_codes)
        sorted_codes = flat_codes[sort_idx]
        sorted_pids = flat_pids[sort_idx]
        sorted_tpos = flat_tpos[sort_idx]

        sorted_hits = np.column_stack((sorted_pids, sorted_tpos))

        mask = np.empty(len(sorted_codes), dtype=bool)
        mask[0] = True
        mask[1:] = sorted_codes[1:] != sorted_codes[:-1]

        unique_codes = sorted_codes[mask]
        split_indices = np.nonzero(mask)[0][1:]

        grouped_hits = np.split(sorted_hits, split_indices)

        self.index = dict(zip(unique_codes, grouped_hits))

    def search(self, query_seq):
        if len(query_seq) < self.k: return []

        seq_bytes = np.frombuffer(query_seq.encode("ascii"), dtype=np.uint8)
        valid_codes, valid_qpos = _numba_encode_and_hash(
            seq_bytes, self.k, self._ascii_map, self.shifts
        )

        if valid_codes.size == 0: return []

        found_p = []
        found_t = []
        found_q = []

        for i in range(len(valid_codes)):
            c = valid_codes[i]
            hits = self.index.get(c)
            if hits is not None:
                found_p.append(hits[:, 0])
                found_t.append(hits[:, 1])
                q_pos = valid_qpos[i]
                found_q.append(np.full(hits.shape[0], q_pos, dtype=np.int32))

        if not found_p:
            return []

        p_all = np.concatenate(found_p)
        t_all = np.concatenate(found_t)
        q_all = np.concatenate(found_q)

        sorted_pids = _numba_aggregate_diagonals(p_all, t_all, q_all)

        return sorted_pids.tolist()
    
    def evaluate(self, query_df, top_n=10):
            hits_at_1 = 0
            hits_at_5 = 0
            hits_at_10 = 0
            mrr_sum = 0.0
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

                    if true_pid in predicted_pids:
                        rank = predicted_pids.index(true_pid) + 1
                        mrr_sum += 1.0 / rank

            n = max(1, len(query_df))

            return {
                "mrr": round(mrr_sum / n, 4),
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