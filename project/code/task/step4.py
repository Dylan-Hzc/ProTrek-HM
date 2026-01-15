import time
import numpy as np
from numba import njit, prange

@njit(cache=True)
def _numba_extend_ungapped(q_enc, t_enc, q_start, t_start, k, match, mismatch, drop_x):

    q_len = q_enc.shape[0]
    t_len = t_enc.shape[0]
    score = 0

    for i in range(k):
        if q_start+i >= q_len or t_start+i >= t_len: break
        score += match if q_enc[q_start+i] == t_enc[t_start+i] else mismatch

    l_best = 0; cur = 0; off = 1
    while q_start - off >= 0 and t_start - off >= 0:
        val = match if q_enc[q_start-off] == t_enc[t_start-off] else mismatch
        cur += val
        if cur > l_best: l_best = cur
        if cur < l_best - drop_x: break
        off += 1

    r_best = 0; cur = 0; r = 0
    while q_start + k + r < q_len and t_start + k + r < t_len:
        val = match if q_enc[q_start+k+r] == t_enc[t_start+k+r] else mismatch
        cur += val
        if cur > r_best: r_best = cur
        if cur < r_best - drop_x: break
        r += 1
        
    return score + l_best + r_best, r

@njit(cache=True)
def search_dense_lut_kmer(q_enc, k, bpa, mask, lut_lo, lut_hi, post_p, post_t, cap):

    L = q_enc.shape[0]
    est_hits = 0

    code = 0
    for i in range(k):
        code = (code << bpa) | q_enc[i]
    if code < lut_lo.shape[0]:
        lo = lut_lo[code]
        if lo != -1:
            cnt = lut_hi[code] - lo
            est_hits += min(cnt, cap)

    for i in range(1, L - k + 1):
        v = q_enc[i + k - 1]
        code = ((code & mask) << bpa) | v
        if code < lut_lo.shape[0]:
            lo = lut_lo[code]
            if lo != -1:
                cnt = lut_hi[code] - lo
                est_hits += min(cnt, cap)

    if est_hits == 0:
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)

    out_p = np.empty(est_hits, dtype=np.int32)
    out_t = np.empty(est_hits, dtype=np.int32)
    out_q = np.empty(est_hits, dtype=np.int32)
    
    idx = 0
    code = 0
    for i in range(k):
        code = (code << bpa) | q_enc[i]

    if code < lut_lo.shape[0]:
        lo = lut_lo[code]
        if lo != -1:
            cnt = min(lut_hi[code] - lo, cap)
            for j in range(cnt):
                out_p[idx] = post_p[lo+j]
                out_t[idx] = post_t[lo+j]
                out_q[idx] = 0
                idx += 1
                
    for i in range(1, L - k + 1):
        v = q_enc[i + k - 1]
        code = ((code & mask) << bpa) | v
        if code < lut_lo.shape[0]:
            lo = lut_lo[code]
            if lo != -1:
                cnt = min(lut_hi[code] - lo, cap)
                for j in range(cnt):
                    out_p[idx] = post_p[lo+j]
                    out_t[idx] = post_t[lo+j]
                    out_q[idx] = i
                    idx += 1
                    
    return out_p[:idx], out_t[:idx], out_q[:idx]

@njit(cache=True, parallel=True)
def parallel_ungapped_extension(
    q_enc, 
    top_pids, top_diags, 
    p_hits, t_hits, q_hits,
    all_seqs, offsets, lengths,
    k, match, mismatch, drop_x
):

    n_top = top_pids.shape[0]
    out_scores = np.full(n_top, -999999, dtype=np.int32)
    
    for i in prange(n_top):
        pid = top_pids[i]
        diag = top_diags[i]

        l = np.searchsorted(p_hits, pid, side='left')
        r = np.searchsorted(p_hits, pid, side='right')
        
        if l >= r: continue

        qs_cand = q_hits[l:r]
        ts_cand = t_hits[l:r]

        count = 0
        for j in range(r - l):
            d = qs_cand[j] - ts_cand[j]
            if d == diag:
                count += 1
        
        if count == 0: continue

        valid_qs = np.empty(count, dtype=np.int32)
        valid_ts = np.empty(count, dtype=np.int32)
        c = 0
        for j in range(r - l):
            if (qs_cand[j] - ts_cand[j]) == diag:
                valid_qs[c] = qs_cand[j]
                valid_ts[c] = ts_cand[j]
                c += 1

        sort_ord = np.argsort(valid_qs)
        valid_qs = valid_qs[sort_ord]
        valid_ts = valid_ts[sort_ord]

        start = offsets[pid]
        t_len = lengths[pid]
        t_seq = all_seqs[start : start + t_len]
        
        best_s = -999999
        last_q = -k

        for j in range(count):
            q_start = valid_qs[j]
            t_start = valid_ts[j]

            if q_start <= last_q: continue 
            
            s, r_len = _numba_extend_ungapped(q_enc, t_seq, q_start, t_start, k, match, mismatch, drop_x)
            if s > best_s: best_s = s
            last_q = q_start + r_len
            
        out_scores[i] = best_s
        
    return out_scores

class NumbaUngappedIndexer:
    """
    Step 4: Highly Optimized Ungapped Extension.
    - No deduplication in search (returns all HSPs).
    - Correct Precision@K calculation.
    - Flat memory structures for speed.
    """

    def __init__(self, k=5, dropoff_x=20, top_n_candidates=500, cap_hits=1000):
        self.k = k
        self.dropoff_x = dropoff_x
        self.top_n_candidates = top_n_candidates
        self.cap_hits = cap_hits

        self.lut_lo = None; self.lut_hi = None
        self.post_p = None; self.post_t = None

        self.all_sequences = None
        self.offsets = None
        self.lengths = None
        self.protein_ids = []

        self.bits_per_aa = 5
        self.mask = (1 << (self.bits_per_aa * (self.k - 1))) - 1
        self._ascii_map = np.full(128, 255, dtype=np.uint8)
        for i, aa in enumerate("ABCDEFGHIJKLMNOPQRSTUVWXYZ"):
            self._ascii_map[ord(aa)] = i
            
        self.match = 5
        self.mismatch = -4

    def build_index(self, fasta_df):
        # 1. Cache IDs
        self.protein_ids = fasta_df['protein_id'].tolist()
        
        total_len = fasta_df['sequence'].str.len().sum()
        self.all_sequences = np.zeros(total_len, dtype=np.uint8)
        self.offsets = np.zeros(len(fasta_df) + 1, dtype=np.int64)
        self.lengths = np.zeros(len(fasta_df), dtype=np.int32)
        
        all_codes = []; all_p = []; all_t = []
        
        curr = 0
        bpa = self.bits_per_aa
        mask = self.mask

        for row in fasta_df.itertuples():
            p_idx = row.Index
            seq = row.sequence
            L = len(seq)
            
            b = np.frombuffer(seq.encode('ascii'), dtype=np.uint8)
            enc = self._ascii_map[b]
            
            self.all_sequences[curr : curr+L] = enc
            self.offsets[p_idx] = curr
            self.lengths[p_idx] = L
            curr += L
            
            if L < self.k: continue

            c, t = self._generate_kmers(enc, self.k, bpa, mask)
            if c.size > 0:
                all_codes.append(c)
                all_t.append(t)
                all_p.append(np.full(c.size, p_idx, dtype=np.int32))
                
        if not all_codes: return
        
        codes_flat = np.concatenate(all_codes)
        t_flat = np.concatenate(all_t)
        p_flat = np.concatenate(all_p)

        order = np.argsort(codes_flat, kind='mergesort')
        sorted_codes = codes_flat[order]
        self.post_t = t_flat[order]
        self.post_p = p_flat[order]
        
        max_code = 1 << (self.k * bpa)
        self.lut_lo = np.full(max_code, -1, dtype=np.int32)
        self.lut_hi = np.full(max_code, -1, dtype=np.int32)
        self._fill_lut(sorted_codes, self.lut_lo, self.lut_hi)

    @staticmethod
    @njit(cache=True)
    def _generate_kmers(enc, k, bpa, mask):
        L = enc.shape[0]
        out_c = np.empty(L - k + 1, dtype=np.int32)
        out_t = np.empty(L - k + 1, dtype=np.int32)
        idx = 0
        
        code = 0
        for i in range(k):
            code = (code << bpa) | enc[i]
        out_c[idx] = code; out_t[idx] = 0; idx += 1
        
        for i in range(1, L - k + 1):
            v = enc[i + k - 1]
            code = ((code & mask) << bpa) | v
            out_c[idx] = code; out_t[idx] = i; idx += 1
        return out_c, out_t

    @staticmethod
    @njit(cache=True)
    def _fill_lut(codes, lo, hi):
        n = codes.shape[0]
        if n == 0: return
        curr = codes[0]
        start = 0
        for i in range(1, n):
            c = codes[i]
            if c != curr:
                if curr < lo.shape[0]:
                    lo[curr] = start
                    hi[curr] = i
                curr = c
                start = i
        if curr < lo.shape[0]:
            lo[curr] = start
            hi[curr] = n

    def search(self, query):
        if len(query) < self.k: return []
        b = np.frombuffer(query.encode('ascii'), dtype=np.uint8)
        q_enc = self._ascii_map[b]

        p_hits, t_hits, q_hits = search_dense_lut_kmer(
            q_enc, self.k, self.bits_per_aa, self.mask, 
            self.lut_lo, self.lut_hi, self.post_p, self.post_t, self.cap_hits
        )
        if p_hits.size == 0: return []

        diags = q_hits - t_hits
        bias = np.int64(2_000_000_000)

        keys = (p_hits.astype(np.int64) << 32) | (diags.astype(np.int64) + bias)
        uk, cnt = np.unique(keys, return_counts=True)

        k_top = min(self.top_n_candidates, uk.size)
        idx = np.argpartition(cnt, -k_top)[-k_top:]
        best_keys = uk[idx]
        
        top_pids = (best_keys >> 32).astype(np.int32)
        top_diags = (best_keys & 0xFFFFFFFF).astype(np.int32) - bias

        sort_p = np.argsort(p_hits, kind='mergesort')
        p_hits = p_hits[sort_p]
        t_hits = t_hits[sort_p]
        q_hits = q_hits[sort_p]
        
        scores = parallel_ungapped_extension(
            q_enc, top_pids, top_diags, 
            p_hits, t_hits, q_hits,
            self.all_sequences, self.offsets, self.lengths,
            self.k, self.match, self.mismatch, self.dropoff_x
        )

        valid = scores > -99999
        if not np.any(valid): return []
        
        final_pids = top_pids[valid]
        final_scores = scores[valid]
        order = np.argsort(final_scores)[::-1]

        return final_pids[order].tolist()

    def _get_index_size_mb(self):
        size_bytes = 0
        if self.lut_lo is not None: size_bytes += self.lut_lo.nbytes * 2
        if self.post_p is not None: size_bytes += self.post_p.nbytes * 2
        if self.all_sequences is not None: size_bytes += self.all_sequences.nbytes
        if self.offsets is not None: size_bytes += self.offsets.nbytes
        return size_bytes / (1024 * 1024)

    def evaluate(self, query_df, top_n=10):
        r1 = r5 = r10 = 0
        p5 = p10 = 0.0
        mrr = 0.0
        t_tot = 0.0
        n_cands = 0
        runtime_peak_mb = 0.0
        
        if not query_df.empty: self.search(query_df.iloc[0].sequence)
        
        n = 0
        queries = query_df['sequence'].tolist()
        true_pids = query_df['origin_protein_id'].tolist()
        
        for query, true_id in zip(queries, true_pids):
            n += 1
            t0 = time.time()
            cands = self.search(query)
            t_tot += (time.time() - t0)
            
            if not cands: continue
            n_cands += len(cands)

            preds = [self.protein_ids[p] for p in cands[:top_n]]
            
            if not preds: continue

            if preds[0] == true_id: r1 += 1
            if true_id in preds[:5]: r5 += 1
            if true_id in preds[:10]: r10 += 1

            if true_id in preds: mrr += 1.0/(preds.index(true_id)+1)

            hits_5 = preds[:5].count(true_id)
            p5 += hits_5 / 5.0

            hits_10 = preds[:10].count(true_id)
            p10 += hits_10 / 10.0
            
        return {
            "mrr": round(mrr/n, 4),
            "recall@1": round(r1/n, 4),
            "recall@5": round(r5/n, 4),
            "recall@10": round(r10/n, 4),
            "precision@5": round(p5/n, 4),
            "precision@10": round(p10/n, 4),
            "avg_time_ms": round(t_tot/n * 1000, 4),
            "avg_candidates": int(n_cands/n),
            "static_index_MB": round(self._get_index_size_mb(), 4),
            "runtime_peak_MB": round(runtime_peak_mb, 4),
            "k": self.k
        }