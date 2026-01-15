import numpy as np
import time
from numba import jit, int64
from .step5 import GappedIndexer, _numba_smith_waterman

@jit(nopython=True, cache=True)
def _numba_compute_spaced_hashes(seq_enc, care_positions, bits_per_aa, weight, span):
    L = len(seq_enc)
    if L < span:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int32)
    
    num_kmers = L - span + 1
    codes = np.zeros(num_kmers, dtype=np.int64)

    shifts = np.empty(len(care_positions), dtype=np.int64)
    for i in range(len(care_positions)):
        shifts[i] = (weight - 1 - i) * bits_per_aa

    for i in range(num_kmers):
        code = 0
        for j in range(len(care_positions)):
            pos_offset = care_positions[j]
            val = seq_enc[i + pos_offset]
            code |= (int64(val) << shifts[j])
        
        codes[i] = code
        
    indices = np.arange(num_kmers, dtype=np.int32)
    return codes, indices

@jit(nopython=True, cache=True)
def _numba_aggregate_diagonals(p_all, t_all, q_all):
    diags = q_all - t_all
    bias = 2_000_000_000
    keys = (p_all.astype(np.int64) << 32) | (diags.astype(np.int64) + bias)

    keys.sort()
    
    if len(keys) == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int32)

    unique_count = 1
    for i in range(1, len(keys)):
        if keys[i] != keys[i-1]:
            unique_count += 1
            
    uk = np.empty(unique_count, dtype=np.int64)
    cnt = np.empty(unique_count, dtype=np.int32)

    curr_idx = 0
    uk[0] = keys[0]
    current_cnt = 1
    
    for i in range(1, len(keys)):
        if keys[i] == keys[i-1]:
            current_cnt += 1
        else:
            cnt[curr_idx] = current_cnt
            curr_idx += 1
            uk[curr_idx] = keys[i]
            current_cnt = 1
            
    cnt[curr_idx] = current_cnt
    return uk, cnt

@jit(nopython=True, cache=True)
def _numba_spaced_extend_process(qs, ts, q_enc, t_enc, match, mismatch, drop_x, span):
    sort_idx = np.argsort(qs)
    qs_sorted = qs[sort_idx]
    ts_sorted = ts[sort_idx]

    q_len = len(q_enc)
    t_len = len(t_enc)
    
    best_total_score = -999999
    last_q_end = -span 

    n_seeds = len(qs_sorted)

    for i in range(n_seeds):
        q_start = qs_sorted[i]
        t_start = ts_sorted[i]

        if q_start <= last_q_end:
            continue

        # Right Extension
        best_r = 0
        curr_r = 0
        r = 0
        while q_start + r < q_len and t_start + r < t_len:
            if q_enc[q_start + r] == t_enc[t_start + r]:
                curr_r += match
            else:
                curr_r += mismatch
            
            if curr_r > best_r: best_r = curr_r
            if curr_r < best_r - drop_x: break
            r += 1

        # Left Extension
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
            
        total = best_l + best_r
        
        if total > best_total_score:
            best_total_score = total
            
        last_q_end = q_start + r

    return best_total_score

class SpacedSeedIndexer(GappedIndexer):
    """
    Step 6: Spaced Seeds Implementation.
    Uses a binary pattern (e.g., "11011") to allow mismatches at specific positions.
    """

    def __init__(self, seed_pattern="11011", dropoff_x=20, top_n_candidates=100, sw_top_n=50):
        self.weight = seed_pattern.count('1')
        self.span = len(seed_pattern)
        self.seed_pattern = seed_pattern
        self.care_positions = np.array([i for i, c in enumerate(seed_pattern) if c == '1'], dtype=np.int64)
        
        super().__init__(k=self.weight, dropoff_x=dropoff_x, top_n_candidates=top_n_candidates, sw_top_n=sw_top_n)

    def _compute_spaced_hashes(self, seq_enc):
        return _numba_compute_spaced_hashes(
            seq_enc, 
            self.care_positions, 
            self.bits_per_aa, 
            self.weight, 
            self.span
        )

    def build_index(self, df):
        self.proteins = df.reset_index(drop=True)
        self.index = {}

        for idx, row in self.proteins.iterrows():
            seq_enc = self._encode_sequence_to_uint8(row['sequence'])
            codes, positions = self._compute_spaced_hashes(seq_enc)
            
            for code, pos in zip(codes, positions):
                code = int(code)
                if code not in self.index:
                    self.index[code] = []
                self.index[code].append([idx, pos])

        for k in self.index:
            self.index[k] = np.array(self.index[k], dtype=np.int32)
        self._ensure_seq_cache()

    def search(self, query_seq):
        self._ensure_seq_cache()
        if len(query_seq) < self.span: return []
        
        q_enc = self._encode_sequence_to_uint8(query_seq)
        q_codes, q_positions = self._compute_spaced_hashes(q_enc)
        
        p_list, t_list, q_list = [], [], []

        for q_code, q_pos in zip(q_codes, q_positions):
            q_code = int(q_code)
            arr = self.index.get(q_code)
            if arr is not None:
                p_list.append(arr[:, 0])
                t_list.append(arr[:, 1])
                q_list.append(np.full(len(arr), q_pos, dtype=np.int32))
                
        if not p_list: return []
        
        p_all = np.concatenate(p_list)
        t_all = np.concatenate(t_list)
        q_all = np.concatenate(q_list)

        uk, cnt = _numba_aggregate_diagonals(p_all, t_all, q_all)

        bias = 2_000_000_000
        order = np.argsort(cnt)[::-1]
        top_keys = order[:self.top_n_candidates]

        top_composite = uk[top_keys]
        top_pids = (top_composite >> 32).astype(np.int32)
        top_diags = (top_composite & 0xFFFFFFFF).astype(np.int32) - bias
        
        results = []
        global_mask = np.isin(p_all, top_pids)
        p_cand_all = p_all[global_mask]
        t_cand_all = t_all[global_mask]
        q_cand_all = q_all[global_mask]
        d_cand_all = (q_cand_all - t_cand_all)
        
        for pid, target_diag in zip(top_pids, top_diags):
            mask_p = (p_cand_all == pid) & (d_cand_all == target_diag)
            qs = q_cand_all[mask_p]
            ts = t_cand_all[mask_p]
            
            if len(qs) == 0: continue
            
            t_seq_str = self._seq_cache[pid]
            t_enc = self._encode_sequence_to_uint8(t_seq_str)

            best_s = _numba_spaced_extend_process(
                qs, ts,
                q_enc, t_enc,
                self.match_score, self.mismatch_score, self.dropoff_x, self.span
            )
            
            if best_s > -9999:
                results.append((pid, best_s))

        results.sort(key=lambda x: x[1], reverse=True)
        sw_candidates = results[:self.sw_top_n]
        
        final_results = []
        s1 = np.ascontiguousarray(q_enc, dtype=np.int32)
        
        for pid, _ in sw_candidates:
            target_seq = self._seq_cache[pid]

            s2 = np.ascontiguousarray(self._encode_sequence_to_uint8(target_seq), dtype=np.int32)

            sw_score = _numba_smith_waterman(
                s1, s2,
                self.match_score, self.mismatch_score, self.gap_score
            )
            final_results.append((pid, sw_score))
            
        final_results.sort(key=lambda x: x[1], reverse=True)
        return [p for p, _ in final_results]

    def evaluate(self, query_df, top_n=10):
            r1 = r5 = r10 = 0
            p5 = p10 = 0.0
            rr = 0.0
            
            t_search = 0.0
            t_sw = 0.0
            
            total_cands = 0
            runtime_peak_mb = 0.0
            
            n = len(query_df)
            if n == 0: n = 1

            for row in query_df.itertuples():
                q_seq = row.sequence
                true_id = row.origin_protein_id

                start_time = time.time()
                candidates = self.search(q_seq)
                t_search += (time.time() - start_time)
                
                if not candidates:
                    continue
                    
                total_cands += len(candidates)

                preds = candidates[:top_n]
                
                if not preds: continue

                if preds[0] == true_id: r1 += 1
                if true_id in preds[:5]: r5 += 1
                if true_id in preds[:10]: r10 += 1

                if true_id in preds:
                    rank = preds.index(true_id) + 1
                    rr += 1.0 / rank

                hits_5 = preds[:5].count(true_id)
                p5 += hits_5 / 5.0

                hits_10 = preds[:10].count(true_id)
                p10 += hits_10 / 10.0

            return {
                "mrr": round(rr / n, 4),
                "recall@1": round(r1 / n, 4),
                "recall@5": round(r5 / n, 4),
                "recall@10": round(r10 / n, 4),
                "precision@5": round(p5 / n, 4),
                "precision@10": round(p10 / n, 4),
                
                "avg_time_ms": round((t_search + t_sw) / n * 1000, 4),
                "avg_search_time_ms": round(t_search / n * 1000, 4),
                "avg_SW_time_ms": round(t_sw / n * 1000, 4), 
                
                "avg_candidates": int(total_cands / n),
                "static_index_MB": round(self._get_index_size_mb(), 4),
                "runtime_peak_MB": round(runtime_peak_mb, 4),
                
                "index_size_keys": len(self.index),
                
                "k": f"Spaced(w={self.weight})",
                "seed_pattern": self.seed_pattern,
            }