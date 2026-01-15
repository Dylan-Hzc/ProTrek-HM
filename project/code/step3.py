import time
import numpy as np
from .step2 import DiagonalIndexer

class UngappedIndexer(DiagonalIndexer):
    """
    Step 3 Baseline (Inherits from Step 2):
    - Reuses Step 2's efficient index build (stores p_idx, t_pos).
    - Adds X-Drop Ungapped Extension logic.
    - specialized evaluate with Precision/MRR.
    """

    def __init__(self, k=5, dropoff_x=20, top_n_candidates=100):
        super().__init__(k)

        self.dropoff_x = dropoff_x
        self.top_n_candidates = top_n_candidates

        self.match_score = 5
        self.mismatch_score = -4

        self.mask = (1 << (self.bits_per_aa * (self.k - 1))) - 1
        
        # [New] Internal timer for extension
        self.total_extend_time = 0.0

    def _encode_sequence_to_uint8(self, seq):
            b = np.frombuffer(seq.encode("ascii"), dtype=np.uint8)
            return self._ascii_map[b]

    def _extend(self, q_enc, t_enc, q_start, t_start):
        score = 0
        k = self.k
        q_len = len(q_enc)
        t_len = len(t_enc)

        for i in range(k):
            if q_start + i >= q_len or t_start + i >= t_len: break
            score += self.match_score if q_enc[q_start+i] == t_enc[t_start+i] else self.mismatch_score

        # 2. Extend Left
        best_l = 0; cur = 0; off = 1
        while q_start - off >= 0 and t_start - off >= 0:
            match = q_enc[q_start - off] == t_enc[t_start - off]
            cur += self.match_score if match else self.mismatch_score
            if cur > best_l: best_l = cur
            # X-drop check
            if cur < best_l - self.dropoff_x: break
            off += 1

        # 3. Extend Right
        best_r = 0; cur = 0; r = 0
        # Start checking *after* the k-mer
        while q_start + k + r < q_len and t_start + k + r < t_len:
            match = q_enc[q_start + k + r] == t_enc[t_start + k + r]
            cur += self.match_score if match else self.mismatch_score
            if cur > best_r: best_r = cur
            # X-drop check
            if cur < best_r - self.dropoff_x: break
            r += 1

        return score + best_l + best_r, r

    def search(self, query_seq):

        if len(query_seq) < self.k: return []

        q_enc = self._encode_sequence_to_uint8(query_seq)
        L = len(q_enc)
        if L < self.k: return []

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
        
        mask = np.isin(p_all, top_pids)
        p_cand = p_all[mask]
        t_cand = t_all[mask]
        q_cand = q_all[mask]
        d_cand = diags[mask]

        for pid, target_diag in zip(top_pids, top_diags):
            
            mask_p = (p_cand == pid) & (d_cand == target_diag)
            qs = q_cand[mask_p]
            ts = t_cand[mask_p]
            
            if len(qs) == 0: continue

            ord_q = np.argsort(qs)
            qs = qs[ord_q]
            ts = ts[ord_q]

            t_seq_str = self.proteins.at[pid, "sequence"]
            t_enc = self._encode_sequence_to_uint8(t_seq_str)
            
            best_s = -1e9
            last_q = -self.k
            
            for q_start, t_start in zip(qs, ts):
                q_start = int(q_start)
                t_start = int(t_start)

                if q_start <= last_q: continue

                t_ext_start = time.time()
                
                s, r = self._extend(q_enc, t_enc, q_start, t_start)

                self.total_extend_time += (time.time() - t_ext_start)
                
                if s > best_s: best_s = s
                last_q = q_start + r
            
            if best_s > -99999:
                results.append((pid, best_s))

        results.sort(key=lambda x: x[1], reverse=True)
        return [p for p, _ in results]

    def evaluate(self, query_df, top_n=10):
        r1 = r5 = r10 = 0
        p5 = p10 = 0.0
        mrr = 0.0
        t_tot = 0.0
        n_cands = 0
        
        runtime_peak_mb = 0.0

        self.total_extend_time = 0.0
        
        for i, row in enumerate(query_df.itertuples()):
            t0 = time.time()
            cands = self.search(row.sequence)
            t_tot += (time.time() - t0)
            
            if not cands: continue
            n_cands += len(cands)

            preds = [self.proteins.at[p, 'protein_id'] for p in cands[:top_n]]
            true_id = row.origin_protein_id
            
            if not preds: continue

            if preds[0] == true_id: r1 += 1
            if true_id in preds[:5]: r5 += 1
            if true_id in preds[:10]: r10 += 1

            if true_id in preds:
                mrr += 1.0 / (preds.index(true_id) + 1)

            hits_5 = preds[:5].count(true_id)
            p5 += hits_5 / 5.0

            hits_10 = preds[:10].count(true_id)
            p10 += hits_10 / 10.0
        
        n = max(1, len(query_df))
        return {
            "mrr": round(mrr/n, 4),
            "recall@1": round(r1/n, 4),
            "recall@5": round(r5/n, 4),
            "recall@10": round(r10/n, 4),
            "precision@5": round(p5/n, 4),
            "precision@10": round(p10/n, 4),
            "avg_time_ms": round(t_tot/n * 1000, 4),
            "avg_extend_time_ms": round(self.total_extend_time/n * 1000, 4),
            "avg_candidates": int(n_cands/n),
            "static_index_MB": round(self._get_index_size_mb(), 4),
            "runtime_peak_MB": round(runtime_peak_mb, 4),
            "k": self.k
        }