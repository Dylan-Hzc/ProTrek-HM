import pandas as pd
import os
import random
from Bio import SeqIO
import pandas as pd

def get_uniprot_gene_name(description: str):
    idx = description.find(" GN=")
    if idx == -1:
        return ""
    idx += 4
    end = description.find(" ", idx)
    if end == -1:
        return description[idx:]
    return description[idx : end]

class ProteinLoader:
    def __init__(self):
        self.fasta_df = None
        self.query_df = None
        self.amino_acids = list("ACDEFGHIKLMNPQRSTVWY")

    def load(self, fasta_path):
        data = []
        
        with open(fasta_path, encoding="utf-8") as handle:
            for record in SeqIO.parse(handle, "fasta"):
                parts = record.id.split("|")

                if len(parts) >= 2:
                    db_type = parts[0]
                    prot_id = parts[1]
                else:
                    db_type = 'tr'
                    prot_id = record.id

                sequence = str(record.seq)
                
                entry = {
                    "protein_id": prot_id,
                    "sequence": sequence,
                    "length": len(sequence),
                    "is_reviewed": db_type == 'sp',
                    # "gene_name": get_uniprot_gene_name(record.description),
                    # "description": record.description
                }
                data.append(entry)

        self.fasta_df = pd.DataFrame(data)
        self.fasta_df.reset_index(inplace=True)

    def _sample(self, length):
        valid_df = self.fasta_df[self.fasta_df['length'] >= length]
        if valid_df.empty:
            raise ValueError(f"No proteins found with length >= {length}")

        row = valid_df.sample(1).iloc[0]
        full_seq = row['sequence']

        start_pos = random.randint(0, len(full_seq) - length)
        sub_seq = full_seq[start_pos : start_pos + length]
        
        return sub_seq, row['protein_id'], start_pos

    def _mutation(self, seq, mutation_rate):
        if mutation_rate <= 0:
            return seq, 0
            
        seq_list = list(seq)
        n_mut = int(len(seq) * mutation_rate)
        indices = random.sample(range(len(seq)), n_mut)
        
        for i in indices:
            original_aa = seq_list[i]
            choices = [aa for aa in self.amino_acids if aa != original_aa]
            seq_list[i] = random.choice(choices)
            
        return "".join(seq_list), n_mut

    def _insert(self, seq, min_block_length, max_block_length, n_fragments=2):

            if max_block_length <= 0 or n_fragments <= 0: return seq, 0
            
            curr = seq
            count = 0
            for _ in range(n_fragments):
                block_len = random.randint(min_block_length, max_block_length)

                L = len(curr)
                if L > 20:
                    start = int(L * 0.25)
                    end = int(L * 0.75)
                    pos = random.randint(start, end)
                else:
                    pos = random.randint(0, L)
                
                block = "".join(random.choices(self.amino_acids, k=block_len))
                curr = curr[:pos] + block + curr[pos:]
                count += 1
            return curr, count

    def _delete(self, seq, min_block_length, max_block_length, n_fragments=2):

        if max_block_length <= 0 or n_fragments <= 0: return seq, 0
        
        curr = seq
        count = 0
        for _ in range(n_fragments):
            L = len(curr)
            if L <= max_block_length + 5: break
            
            block_len = random.randint(min_block_length, max_block_length)
            
            if L > 20:
                start = int(L * 0.25)
                end = int(L * 0.75) - block_len
                if end <= start: end = start + 1
                pos = random.randint(start, end)
            else:
                if L - block_len <= 0: break
                pos = random.randint(0, L - block_len)
            
            curr = curr[:pos] + curr[pos + block_len:]
            count += 1
        return curr, count

    def queries_generate(self, n_queries, length, mutation_rate=0, min_block_length=0, max_block_length=0):

        if self.fasta_df is None:
            raise ValueError("Data not loaded. Call load() first.")

        query_data = []

        for i in range(n_queries):
            seq, pid, start = self._sample(length)

            seq, n_subs = self._mutation(seq, mutation_rate)
            
            n_ins = 0
            n_del = 0
            if max_block_length > 0:
                random_value = random.random()
                if random_value < 0.4:
                    seq, n_ins = self._insert(seq, min_block_length, max_block_length, n_fragments = int(0.02 * length))
                elif random_value < 0.8:   
                    seq, n_del = self._delete(seq, min_block_length, max_block_length, n_fragments = int(0.02 * length))
                else:
                    seq, n_ins = self._insert(seq, min_block_length, max_block_length, n_fragments = int(0.01 * length))
                    seq, n_del = self._delete(seq, min_block_length, max_block_length, n_fragments = int(0.01 * length))

            query_data.append({
                "query_index": i,
                "sequence": seq,
                "origin_protein_id": pid,
                "origin_start": start,
                "origin_length": length,
                "n_subs": n_subs,
                "n_ins": n_ins,
                "n_del": n_del
            })

        self.query_df = pd.DataFrame(query_data)
        return self.query_df

    def queries_generate_from_other(self, query_df, length, mutation_rate=0, min_block_length=0, max_block_length=0):

        query_data = []

        for i, row in enumerate(query_df.itertuples()):
            seq = row.sequence
            pid = row.origin_protein_id
            start = row.origin_start

            seq, n_subs = self._mutation(seq, mutation_rate)
            
            n_ins = 0
            n_del = 0
            if max_block_length > 0:
                random_value = random.random()
                if random_value < 0.4:
                    seq, n_ins = self._insert(seq, min_block_length, max_block_length, n_fragments = int(0.1 * length))
                elif random_value < 0.8:   
                    seq, n_del = self._delete(seq, min_block_length, max_block_length, n_fragments = int(0.1 * length))
                else:
                    seq, n_ins = self._insert(seq, min_block_length, max_block_length, n_fragments = int(0.1 * length))
                    seq, n_del = self._delete(seq, min_block_length, max_block_length, n_fragments = int(0.1 * length))

            query_data.append({
                "query_index": i,
                "sequence": seq,
                "origin_protein_id": pid,
                "origin_start": start,
                "origin_length": length,
                "n_subs": n_subs,
                "n_ins": n_ins,
                "n_del": n_del
            })

        self.query_df = pd.DataFrame(query_data)
        return self.query_df

    def fasta_df_from_csv(self, folder_path, filename="fasta_df.csv"):
        csv_path = os.path.join(folder_path, filename)
        self.fasta_df = pd.read_csv(csv_path)
        return self.fasta_df

    def query_df_from_csv(self, folder_path, filename="30_0.0_0_0.csv"):
        csv_path = os.path.join(folder_path, filename)
        self.query_df = pd.read_csv(csv_path)
        return self.query_df
