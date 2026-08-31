# Datasets

Training and evaluation use the **GOAT** dataset (Guitar On Audio and
Tablatures), which is not redistributed here — it is roughly 15 GB and has its
own licence. Obtain it from the authors' release accompanying:

> Loth et al., *GOAT: A Large Dataset of Paired Guitar Audio and Tablature*, 2025.

## Expected layout

`whisplab` reads the dataset from a `GOAT/` directory at the repository root:

```
GOAT/
├── metadata.csv
└── data/
    ├── item_0/
    │   ├── item_0.wav                 # direct-input audio (44.1 kHz)
    │   ├── item_0_amp_1.wav … _5.wav  # five re-amped renderings
    │   ├── item_0_gp.wav              # Guitar Pro render
    │   ├── item_0.mid                 # unaligned MIDI
    │   ├── item_0_fine_aligned.mid    # fine-aligned MIDI — the pitch/timing target
    │   ├── item_0.txt                 # DadaGP token encoding — the technique source
    │   ├── item_0.gp / .gp5           # Guitar Pro tablature
    └── item_1/ …
```

`metadata.csv` stores paths of the form `GOAT/item_0/item_0.wav`; the loader
rewrites the prefix to `GOAT/data/item_0/…`, so keep the `data/` level in
place. Point elsewhere by setting `Config.goat_root`.

## Filtering applied by the loader

`GOATDataset` drops recordings that:

- are outside the three supported tuning classes (`standard`, `dropd`,
  `downtuned`) — the `othertunings` class is excluded, and
- have no fine-aligned MIDI annotation.

Of the 172 recordings this leaves **146**: 139 in the training pool and 7 in
the test split. During training each of the 139 contributes its DI signal plus
five re-amped variants, giving **834 training samples**; validation and test
inference use DI audio only.
