# %%
# ! pip install ipykernel conda-forge pandas numpy biopython tqdm numba

# %%
import os
import random
import pandas as pd
from tqdm.auto import tqdm

# %% [markdown]
# ## Query Generate

# %%
from ProrteinLoader import ProteinLoader

seed = 1989
random.seed(seed)
lengths = [30, 60, 120, 240]
sub_rates = [0.0, 0.05, 0.10, 0.20, 0.30]
block_lengths = [(0, 0), (1, 3), (3, 5)]

fasta_path = r"..\data\uniprotkb_human_reviewed.fasta"
out_dir = r"..\result\query\reviewed"

os.makedirs(out_dir, exist_ok=True)
loader = ProteinLoader()
loader.load(fasta_path)
loader.fasta_df.to_csv(r"..\result\query\reviewed\fasta_df.csv", index=False)

for length in lengths:
    query_df_zero = loader.queries_generate(
        n_queries=int(len(loader.fasta_df) * 0.002),
        length=length,
        mutation_rate=0,
        min_block_length=0,
        max_block_length=0
    )
    for sub_rate in sub_rates:
        for min_block_length, max_block_length in block_lengths:
            query_df = loader.queries_generate_from_other(
                query_df=query_df_zero.copy(),
                length=length,
                mutation_rate=sub_rate,
                min_block_length=min_block_length,
                max_block_length=max_block_length
            )
            fname = f"{length}_{sub_rate}_{min_block_length}_{max_block_length}.csv"
            save_path = os.path.join(out_dir, fname)

            query_df.to_csv(save_path, index=False)

fasta_path = r"..\data\uniprotkb_human_unreviewed.fasta"
out_dir = r"..\result\query\unreviewed"

os.makedirs(out_dir, exist_ok=True)
loader = ProteinLoader()
loader.load(fasta_path)
loader.fasta_df.to_csv(r"..\result\query\unreviewed\fasta_df.csv", index=False)

for length in lengths:
    query_df_zero = loader.queries_generate(
        n_queries=int(len(loader.fasta_df) * 0.002),
        length=length,
        mutation_rate=0,
        min_block_length=0,
        max_block_length=0
    )
    for sub_rate in sub_rates:
        for min_block_length, max_block_length in block_lengths:
            query_df = loader.queries_generate_from_other(
                query_df=query_df_zero.copy(),
                length=length,
                mutation_rate=sub_rate,
                min_block_length=min_block_length,
                max_block_length=max_block_length
            )
            fname = f"{length}_{sub_rate}_{min_block_length}_{max_block_length}.csv"
            save_path = os.path.join(out_dir, fname)

            query_df.to_csv(save_path, index=False)


# %%
input_folder_path = [
    r"..\result\query\reviewed",
    r"..\result\query\unreviewed"
]

# %% [markdown]
# ## Step1

# %%
from step1 import KmerIndexer
from ProrteinLoader import ProteinLoader
from MemoryMonitor import MemoryMonitor

# %%
lengths = [30, 60, 120, 240]
ks = [3, 4, 5, 6, 7]
sub_rates = [0.0, 0.10, 0.30]
block_lengths = [(0, 0), (1, 3), (3, 5)]
top_n = 10

# %%
loader = ProteinLoader()
output_folder_path = [
    r"..\result\step1\reviewed",
    r"..\result\step1\unreviewed"
]
for folder_path in tqdm(input_folder_path, desc="Dataset", position=0):
    results = []
    fasta_df = loader.fasta_df_from_csv(folder_path)
    for k in tqdm(ks, desc="k-mer size", position=1, leave=True):
        kmer_indexer = KmerIndexer(k=k)
        kmer_indexer.build_index(fasta_df)
        for length in tqdm( lengths, desc="Sequence length", position=2, leave=False):
            for sub_rate in tqdm(sub_rates, desc="Mutation rate", position=3, leave=False):
                for min_block_length, max_block_length in block_lengths:
                    query_df = loader.query_df_from_csv(
                        folder_path,
                        filename=f"{length}_{sub_rate}_{min_block_length}_{max_block_length}.csv"
                    )

                    monitor = MemoryMonitor(interval=0.001)
                    monitor.start()
                    metrics = kmer_indexer.evaluate(
                        query_df,
                        top_n=top_n
                    )
                    monitor.stop(); monitor.join()
                    metrics["runtime_peak_MB"] = round(monitor.get_peak_delta_mb(), 4)

                    metrics.update({
                        "mutation_rate": sub_rate,
                        "length": length,
                        "num_queries": len(query_df),
                        "blocks": 0 if (min_block_length == 0 and max_block_length == 0) else f"({min_block_length}, {max_block_length})",
                        "dataset": "unreviewed" if "unreviewed" in folder_path else "reviewed",
                    })

                    results.append(metrics)

    results_df = pd.DataFrame(results)
    results_df.to_csv(
        os.path.join(
            output_folder_path[1] if "unreviewed" in folder_path else output_folder_path[0],
            f"step1.csv"
        ),
        index=False
    )


# %% [markdown]
# ## Step2

# %%
from step2 import DiagonalIndexer
from ProrteinLoader import ProteinLoader
from MemoryMonitor import MemoryMonitor

# %%
lengths = [30, 60, 120, 240]
ks = [4, 5, 6]
sub_rates = [0.0, 0.10, 0.30]
block_lengths = [(0, 0), (1, 3), (3, 5)]
top_n = 10

# %%
loader = ProteinLoader()
output_folder_path = [
    r"..\result\step2\reviewed",
    r"..\result\step2\unreviewed"
]
for folder_path in tqdm(input_folder_path, desc="Dataset", position=0):
    results = []
    fasta_df = loader.fasta_df_from_csv(folder_path)
    for k in tqdm(ks, desc="k-mer size", position=1, leave=True):
        kmer_indexer = DiagonalIndexer(k=k)
        kmer_indexer.build_index(fasta_df)
        for length in tqdm( lengths, desc="Sequence length", position=2, leave=False):
            for sub_rate in tqdm(sub_rates, desc="Mutation rate", position=3, leave=False):
                for min_block_length, max_block_length in block_lengths:
                    query_df = loader.query_df_from_csv(
                        folder_path,
                        filename=f"{length}_{sub_rate}_{min_block_length}_{max_block_length}.csv"
                    )

                    monitor = MemoryMonitor(interval=0.001)
                    monitor.start()
                    metrics = kmer_indexer.evaluate(
                        query_df,
                        top_n=top_n
                    )
                    monitor.stop(); monitor.join()
                    metrics["runtime_peak_MB"] = round(monitor.get_peak_delta_mb(), 4)

                    metrics.update({
                        "mutation_rate": sub_rate,
                        "length": length,
                        "num_queries": len(query_df),
                        "blocks": 0 if (min_block_length == 0 and max_block_length == 0) else f"({min_block_length}, {max_block_length})",
                        "dataset": "unreviewed" if "unreviewed" in folder_path else "reviewed",
                    })

                    results.append(metrics)

    results_df = pd.DataFrame(results)
    results_df.to_csv(
        os.path.join(
            output_folder_path[1] if "unreviewed" in folder_path else output_folder_path[0],
            f"step2.csv"
        ),
        index=False
    )


# %% [markdown]
# ## Step3

# %%
from step3 import UngappedIndexer
from ProrteinLoader import ProteinLoader
from MemoryMonitor import MemoryMonitor

# %%
lengths = [30, 60, 120, 240]
ks = [4, 5, 6]
sub_rates = [0.0, 0.10, 0.30]
block_lengths = [(0, 0), (1, 3), (3, 5)]
top_n = 10

# %%
loader = ProteinLoader()
output_folder_path = [
    r"..\result\step3\reviewed",
    r"..\result\step3\unreviewed"
]
for folder_path in tqdm(input_folder_path, desc="Dataset", position=0):
    results = []
    fasta_df = loader.fasta_df_from_csv(folder_path)
    for k in tqdm(ks, desc="k-mer size", position=1, leave=True):
        kmer_indexer = UngappedIndexer(k=k, dropoff_x=30, top_n_candidates=1000)
        kmer_indexer.build_index(fasta_df)
        for length in tqdm( lengths, desc="Sequence length", position=2, leave=False):
            for sub_rate in tqdm(sub_rates, desc="Mutation rate", position=3, leave=False):
                for min_block_length, max_block_length in block_lengths:
                    query_df = loader.query_df_from_csv(
                        folder_path,
                        filename=f"{length}_{sub_rate}_{min_block_length}_{max_block_length}.csv"
                    )

                    monitor = MemoryMonitor(interval=0.001)
                    monitor.start()
                    metrics = kmer_indexer.evaluate(query_df)
                    monitor.stop(); monitor.join()
                    metrics["runtime_peak_MB"] = round(monitor.get_peak_delta_mb(), 4)

                    metrics.update({
                        "mutation_rate": sub_rate,
                        "length": length,
                        "num_queries": len(query_df),
                        "blocks": 0 if (min_block_length == 0 and max_block_length == 0) else f"({min_block_length}, {max_block_length})",
                        "dataset": "unreviewed" if "unreviewed" in folder_path else "reviewed",
                    })

                    results.append(metrics)

    results_df = pd.DataFrame(results)
    results_df.to_csv(
        os.path.join(
            output_folder_path[1] if "unreviewed" in folder_path else output_folder_path[0],
            f"step3.csv"
        ),
        index=False
    )


# %% [markdown]
# ## Step 4

# %%
import gc
from step4 import NumbaUngappedIndexer
from ProrteinLoader import ProteinLoader
from MemoryMonitor import MemoryMonitor

# %%
lengths = [30, 60, 120, 240]
ks = [4, 5, 6]
sub_rates = [0.0, 0.10, 0.30]
block_lengths = [(0, 0), (1, 3), (3, 5)]
top_n_eval = 10

dropoff_xs = [30, 50]
top_n_cands = [1000, 5000]
cap_hits_list = [1000, 2000]
param_combinations = [
    (dx, tn, ch)
    for dx in dropoff_xs
    for tn in top_n_cands
    for ch in cap_hits_list
]

# %%
loader = ProteinLoader()
output_folder_path = [
    r"..\result\step4\reviewed",
    r"..\result\step4\unreviewed"
]
for dropoff_x, top_n_cand, cap_hit in tqdm(param_combinations, desc="Params Grid", position=0):
    for i, folder_path in enumerate(tqdm(input_folder_path, desc=f"Dataset ({top_n_cand},{cap_hit})", position=1, leave=False)):
        current_output_dir = output_folder_path[i]
        os.makedirs(current_output_dir, exist_ok=True)
        
        results = []

        fasta_df = loader.fasta_df_from_csv(folder_path)
        for k in tqdm(ks, desc="k-mer size", position=2, leave=False):
            kmer_indexer = NumbaUngappedIndexer(
                k=k, 
                dropoff_x=dropoff_x, 
                top_n_candidates=top_n_cand, 
                cap_hits=cap_hit
            )
            kmer_indexer.build_index(fasta_df)

            for length in tqdm(lengths, desc="Length", position=3, leave=False):
                for sub_rate in sub_rates:
                    for min_block, max_block in block_lengths:
                        filename = f"{length}_{sub_rate}_{min_block}_{max_block}.csv"
                        try:
                            query_df = loader.query_df_from_csv(folder_path, filename=filename)
                        except FileNotFoundError:
                            continue
                        gc.collect()
                        monitor = MemoryMonitor(interval=0.001)
                        monitor.start()
                        metrics = kmer_indexer.evaluate(query_df, top_n=top_n_eval)
                        monitor.stop()
                        monitor.join()
                        metrics["runtime_peak_MB"] = round(monitor.get_peak_delta_mb(), 4)

                        metrics.update({
                            "mutation_rate": sub_rate,
                            "length": length,
                            "num_queries": len(query_df),
                            "blocks": 0 if (min_block == 0 and max_block == 0) else f"({min_block}, {max_block})",
                            "dataset": "unreviewed" if "unreviewed" in folder_path else "reviewed",
                        })

                        results.append(metrics)

        results_df = pd.DataFrame(results)
        save_filename = f"step4_{dropoff_x}_{top_n_cand}_{cap_hit}.csv"
        save_path = os.path.join(current_output_dir, save_filename)
        
        results_df.to_csv(save_path, index=False)

# %% [markdown]
# ## Step5

# %%
from step5 import GappedIndexer
from ProrteinLoader import ProteinLoader
from MemoryMonitor import MemoryMonitor

# %%
lengths = [30, 60, 120, 240]
ks = [4, 5, 6]
sw_top_ns = [20, 50, 100]
sub_rates = [0.0, 0.10, 0.30]
block_lengths = [(0, 0), (1, 3), (3, 5)]
top_n = 10

# %%
loader = ProteinLoader()
output_folder_path = [
    r"..\result\step5\reviewed",
    r"..\result\step5\unreviewed"
]
for folder_path in tqdm(input_folder_path, desc="Dataset", position=0):
    results = []
    fasta_df = loader.fasta_df_from_csv(folder_path)
    for k in tqdm(ks, desc="k-mer size", position=1, leave=True):
        for sw_top_n in tqdm(sw_top_ns, desc="SW top N", position=2, leave=False):
            kmer_indexer = GappedIndexer(k=k, dropoff_x=30, top_n_candidates=1000, sw_top_n=sw_top_n)
            kmer_indexer.build_index(fasta_df)
            for length in tqdm( lengths, desc="Sequence length", position=3, leave=False):
                for sub_rate in tqdm(sub_rates, desc="Mutation rate", position=4, leave=False):
                    for min_block_length, max_block_length in block_lengths:
                        query_df = loader.query_df_from_csv(
                            folder_path,
                            filename=f"{length}_{sub_rate}_{min_block_length}_{max_block_length}.csv"
                        )

                        monitor = MemoryMonitor(interval=0.001)
                        monitor.start()
                        metrics = kmer_indexer.evaluate(query_df)
                        monitor.stop(); monitor.join()
                        metrics["runtime_peak_MB"] = round(monitor.get_peak_delta_mb(), 4)

                        metrics.update({
                            "mutation_rate": sub_rate,
                            "length": length,
                            "num_queries": len(query_df),
                            "blocks": 0 if (min_block_length == 0 and max_block_length == 0) else f"({min_block_length}, {max_block_length})",
                            "dataset": "unreviewed" if "unreviewed" in folder_path else "reviewed",
                            "sw_top_n": sw_top_n,
                        })

                        results.append(metrics)

    results_df = pd.DataFrame(results)
    results_df.to_csv(
        os.path.join(
            output_folder_path[1] if "unreviewed" in folder_path else output_folder_path[0],
            f"step5.csv"
        ),
        index=False
    )


# %% [markdown]
# ## Step6

# %%
from step6 import SpacedSeedIndexer
from ProrteinLoader import ProteinLoader
from MemoryMonitor import MemoryMonitor


# %%
lengths = [30, 60, 120, 240]
sw_top_ns = [20, 50, 100]
sub_rates = [0.0, 0.10, 0.30]
block_lengths = [(0, 0), (1, 3), (3, 5)]
top_n = 10
seed_patterns = [
    "11111", # Standard k=5. Fails easily with mutations.
    "1101011",     # (Span 7) Classic PatternHunter. Very robust.
    "11001011"
]

# %%
loader = ProteinLoader()
output_folder_path = [
    r"..\result\step6\reviewed",
    r"..\result\step6\unreviewed"
]

for folder_path in tqdm(input_folder_path, desc="Dataset", position=0):
    results = []
    fasta_df = loader.fasta_df_from_csv(folder_path)
    for seed_pattern in tqdm(seed_patterns, desc="seed_patterns", position=1, leave=True):
        for sw_top_n in tqdm(sw_top_ns, desc="SW top N", position=2, leave=False):
            kmer_indexer = SpacedSeedIndexer(seed_pattern=seed_pattern, dropoff_x=30, top_n_candidates=1000, sw_top_n=sw_top_n)
            kmer_indexer.build_index(fasta_df)
            for length in tqdm( lengths, desc="Sequence length", position=3, leave=False):
                for sub_rate in tqdm(sub_rates, desc="Mutation rate", position=4, leave=False):
                    for min_block_length, max_block_length in block_lengths:
                        query_df = loader.query_df_from_csv(
                            folder_path,
                            filename=f"{length}_{sub_rate}_{min_block_length}_{max_block_length}.csv"
                        )

                        monitor = MemoryMonitor(interval=0.001)
                        monitor.start()
                        metrics = kmer_indexer.evaluate(query_df)
                        monitor.stop(); monitor.join()
                        metrics["runtime_peak_MB"] = round(monitor.get_peak_delta_mb(), 4)

                        metrics.update({
                            "mutation_rate": sub_rate,
                            "length": length,
                            "num_queries": len(query_df),
                            "blocks": 0 if (min_block_length == 0 and max_block_length == 0) else f"({min_block_length}, {max_block_length})",
                            "dataset": "unreviewed" if "unreviewed" in folder_path else "reviewed",
                            "sw_top_n": sw_top_n,
                        })

                        results.append(metrics)

    results_df = pd.DataFrame(results)
    results_df.to_csv(
        os.path.join(
            output_folder_path[1] if "unreviewed" in folder_path else output_folder_path[0],
            f"step6.csv"
        ),
        index=False
    )



