# datasets/

This project benchmarks against a **SIFT1M subset** (the standard ANN
benchmark dataset), not synthetic random vectors.

## Getting the dataset

Download requires network access (this repo does not vendor the dataset).
From the project root, with network available:

```bash
mkdir -p datasets/sift
curl -o datasets/sift/sift.tar.gz ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz
tar -xzf datasets/sift/sift.tar.gz -C datasets/sift
```

This unpacks `.fvecs`/`.ivecs` files (`sift_base.fvecs`, `sift_query.fvecs`,
`sift_groundtruth.ivecs`). `benchmarks/load_sift.py` reads these directly --
no conversion step needed.

`benchmarks/run_benchmarks.py` falls back to a synthetic Gaussian dataset
if `datasets/sift/sift_base.fvecs` isn't present, so the benchmark suite
and its charts still run end-to-end. **Recall numbers from the synthetic
fallback are not comparable to published ANN benchmark numbers**.
