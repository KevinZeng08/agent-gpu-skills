# NVIDIA GPU Spec Reference

Use this reference when choosing CUDA `-arch` targets, estimating kernel
rooflines, or checking whether a model/run is limited by memory capacity,
memory bandwidth, Tensor Core throughput, PCIe/NVLink bandwidth, or power.

## Scope and Provenance

- Source table: https://moonshot.feishu.cn/sheets/OaBLsKXazhMXjLt9dw5cC4Gvnzc?sheet=1357cc.
- AMD rows in the source table are intentionally omitted; this is a CUDA/NVIDIA
  reference.
- Public sources were used to correct or qualify obvious table issues. Prefer
  the corrected values below over the raw spreadsheet when they differ.
- Tensor throughput is dense unless the cell explicitly says sparse. Do not use
  sparse peaks for normal dense GEMM/attention rooflines.
- Bandwidth numbers are theoretical peak bandwidths from public specifications,
  not measured application throughput.

## Quick Lookup: NVIDIA Datacenter GPUs

| GPU | Arch | CUDA CC | SM | FP32 TFLOPS | FP16/BF16 TC TFLOPS | FP8 TC TFLOPS | FP4 TC TFLOPS | Memory | Mem BW | PCIe | NVLink / C2C | Power | Status / notes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| B200 | Blackwell | 10.0 | 148 | 75 | 2,250 | 4,500 | 9,000 | 180 GB HBM3e usable | 8,000 GB/s | PCIe 5.0 128 GB/s | 900 GB/s | 1,000 W class | TechPowerUp lists B200 SXM 192 GB as 148 SM / 18,944 CUDA cores; source table used 180 GB usable memory. |
| B300 | Blackwell Ultra | 10.3 | 160 | 75 | 2,250 | 4,500 | 13,500 | 288 GB HBM3e | 8,000 GB/s | PCIe 6.0 | 900 GB/s | 1,100 W class | NVIDIA Technical Blog states Blackwell Ultra GPU has 160 SM; source table `INT8=307 TOPS` is inconsistent and should not be used. |
| GB300 single Blackwell GPU | Blackwell Ultra | 10.3 | 160 | 80 | 2,500 | 5,000 | 15,000 | 288 GB HBM3e | 8,000 GB/s | PCIe 6.0 | 2 x 900 GB/s | up to 1,400 W module context | Values derived per Blackwell Ultra GPU from GB300 NVL platform specs. |
| GB200 single Blackwell GPU | Blackwell | 10.0 | 152 | 80 | 2,500 | 5,000 | 10,000 | 186 GB HBM3e in NVL72 context | 8,000 GB/s | PCIe 5.0 128 GB/s | 2 x 900 GB/s | up to 2,700 W superchip context | GB200 superchip contains two 152-SM Blackwell GPUs plus Grace CPU; avoid comparing module power to single-GPU TDP. |
| GH200 new | Hopper + Grace | 9.0 GPU | 132 GPU-side | 67 | 989.5 | 1,979 | N/A | 141 GB HBM3e | 4,900 GB/s | 4 x PCIe 5.0 equivalent | 450 GB/s NVLink-C2C | 1,000 W superchip context | Grace Hopper Superchip; power and interconnect are module/system context, not a standalone H200 PCIe card. |
| GH200 old | Hopper + Grace | 9.0 GPU | 132 GPU-side | 67 | 989.5 | 1,979 | N/A | 96 GB HBM3 | 4,000 GB/s | 4 x PCIe 5.0 equivalent | 450 GB/s NVLink-C2C | 1,000 W superchip context | Earlier GH200 memory configuration. |
| H100 SXM | Hopper | 9.0 | 132 | 67 | 989.5 | 1,979 | N/A | 80 GB HBM3 | 3,350 GB/s | PCIe via platform | 450 GB/s direction, 900 GB/s bidirectional | 700 W | Source table omitted FP32; corrected to 67 TFLOPS. |
| H800 SXM | Hopper | 9.0 | 132 | 67 | 989.5 | 1,979 | N/A | 80 GB HBM3 | 3,350 GB/s | PCIe via platform | reduced vs H100; source table 200 GB/s direction | 700 W | China-market H100 derivative; use exact vendor/system sheet for interconnect. |
| H200 SXM | Hopper | 9.0 | 132 | 67 | 989.5 | 1,979 | N/A | 141 GB HBM3e | 4,800 GB/s | PCIe via platform | 450 GB/s direction, 900 GB/s bidirectional | 700 W | Main change vs H100 SXM is capacity/bandwidth, not compute capability. |
| H20S | Hopper derivative | 9.0 class | 78 | 44 | 148 | 296 | N/A | 141 GB HBM3e | 4,800 GB/s | unknown | unknown | unknown | Source row has no link; treat as incomplete until an official datasheet is available. |
| H20 | Hopper derivative | 9.0 class | 78 | 44 | 148 | 296 | N/A | 96 GB HBM3 | 4,000 GB/s | unknown | 450 GB/s in source row | 400 W | China-market derivative; source link is non-official. Confirm on actual system. |
| A100 SXM 80GB | Ampere | 8.0 | 108 | 19.5 | 312 | N/A | N/A | 80 GB HBM2e | 2,039 GB/s | PCIe Gen4 via platform | 200 GB/s direction, 400 GB/s bidirectional in source row | 400 W | MIG up to 7 instances. |
| A800 SXM 80GB | Ampere | 8.0 | 108 | 19.5 | 312 | N/A | N/A | 80 GB HBM2e | 2,039 GB/s | PCIe Gen4 via platform | 200 GB/s direction in source row | 400 W | China-market A100 derivative; interconnect is reduced vs A100-class public marketing. |
| A800 PCIe 80GB | Ampere | 8.0 | 108 | 19.5 | 312 | N/A | N/A | 80 GB HBM2e | 1,935 GB/s | PCIe Gen4 x16 | 400 GB/s in source row | 300 W | PCIe card; lower TDP than SXM. |
| A30 | Ampere | 8.0 | 56 | 10.3 | 165 | N/A | N/A | 24 GB HBM2 | 933 GB/s | PCIe Gen4 | N/A | 165 W | Inference/HPC card; much lower memory capacity than A100/H100. |
| L40S | Ada | 8.9 | 142 | 91.6 | 362 | 733 | N/A | 48 GB GDDR6 ECC | 864 GB/s | PCIe Gen4 x16, 64 GB/s bidirectional | N/A | 350 W | Source table is consistent with NVIDIA L40S datasheet. |
| L40 | Ada | 8.9 | 142 | 90.5 | 181 | 362 | N/A | 48 GB GDDR6 ECC | 864 GB/s | PCIe Gen4 x16, 64 GB/s bidirectional | N/A | 300 W | Similar memory subsystem to L40S but lower AI Tensor throughput. |
| L4 | Ada | 8.9 | 58 | 30.3 | 121 | 242 | N/A | 24 GB GDDR6 ECC | 300 GB/s | PCIe Gen4 x16 | N/A | 72 W | Low-profile, low-power inference/video GPU. |

## Quick Lookup: NVIDIA Workstation and GeForce GPUs

| GPU | Arch | CUDA CC | SM | FP32 TFLOPS | FP16/BF16 TC TFLOPS | FP8 TC TFLOPS | FP4 TC TFLOPS | Memory | Mem BW | PCIe | NVLink | Power | Status / notes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| RTX PRO 6000 Blackwell | Blackwell | 12.0 family | 188 | about 126 | about 1,001 FP16 sparse class | about 2,002 sparse class | about 4,004 sparse class | 96 GB GDDR7 ECC | 1,792 GB/s | PCIe 5.0 x16 | no workstation NVLink | 600 W class depending edition | Source table memory bandwidth rounded to 1,800; corrected to 1,792 GB/s from public architecture/spec materials. |
| RTX PRO 5880 Ada | Ada | 8.9 | 108 | 69.3 | 277.1 | 554.2 | N/A | 48 GB GDDR6 ECC | 960 GB/s | PCIe Gen4 x16 | N/A | 285 W | China-market/sanctions-adjusted Ada workstation card. |
| RTX PRO 5000 Blackwell | Blackwell | 12.0 family | unknown | 65 | 258 | unknown | unknown | 48 GB or 72 GB GDDR7 ECC | 1,344 GB/s | PCIe 5.0 x16 | N/A | edition dependent | Source row lists `48/72`; keep both because both capacities appear in public materials. |
| GeForce RTX 5090 | Blackwell | 12.0 | 170 | about 104.8 | not comparable to datacenter TC table | not comparable to datacenter TC table | AI TOPS marketed, not datacenter TFLOPS | 32 GB GDDR7 | 1,792 GB/s | PCIe 5.0 x16 | N/A | 575 W | Use for CUDA dev/consumer inference only with consumer-card caveats: no ECC, no MIG, no datacenter support. |
| GeForce RTX 5090D | Blackwell | 12.0 class | unknown | unknown | unknown | unknown | unknown | unknown | unknown | PCIe 5.0 x16 likely | N/A | unknown | Source row has no link/data; leave incomplete. |
| GeForce RTX 4090 | Ada | 8.9 | 128 | about 82.6 | 330 | 660 | N/A | 24 GB GDDR6X | 1,008 GB/s | PCIe Gen4 x16 | N/A | 450 W | Source table is broadly consistent; Tensor peaks are not published in NVIDIA datacenter datasheet format. |
| GeForce RTX 4090D | Ada | 8.9 | 114 | about 73.5 | 293 | unknown | N/A | 24 GB GDDR6X | 1,008 GB/s | PCIe Gen4 x16 | N/A | 425 W | Source row only had partial Tensor value; completed with public 4090D specs. |
| GeForce RTX 3090 | Ampere | 8.6 | 82 | 35.6 | 142 | N/A | N/A | 24 GB GDDR6X | 936 GB/s | PCIe Gen4 x16 | 40 GB/s NVLink bridge | 350 W | Good for CUDA dev; no MIG/ECC datacenter behavior. |
| RTX 6000 Ada | Ada | 8.9 | 142 | 91.1 | not listed in source row | 728.5 | N/A | 48 GB GDDR6 ECC | 960 GB/s | PCIe Gen4 x16 | N/A | 300 W | Workstation version; do not confuse with RTX PRO 6000 Blackwell. |
| RTX 5880 Ada | Ada | 8.9 | 108 | 69.3 | 277.1 | 554.2 | N/A | 48 GB GDDR6 ECC | 960 GB/s | PCIe Gen4 x16 | N/A | 285 W | Same family as RTX PRO 5880 Ada row naming. |
| RTX 5000 Ada | Ada | 8.9 | 100 | 65.3 | 261.1 | unknown | N/A | 32 GB GDDR6 ECC | 576 GB/s | PCIe Gen4 x16 | N/A | 250 W | Source row had no link; values align with public RTX 5000 Ada specs. |

## Corrections Applied from the Spreadsheet

- `H100 SXM`: source table left pure CUDA/FP32 blank; use 67 TFLOPS for H100
  SXM-class FP32 peak.
- `B300`: source table `int8 TC = 307 TOPS` is inconsistent with the same row's
  FP8/FP4 Blackwell Ultra scale. Treat that cell as erroneous/unverified.
- `B200` and `GB200`: SM count filled as `148` from TechPowerUp-style GPU DB
  entries and GB200 configuration summaries (`18,944 CUDA cores / 128 = 148`).
- `B300` and `GB300`: SM count filled as `160` because NVIDIA's Blackwell
  Ultra technical blog says the Blackwell Ultra GPU has 160 SM. This also
  matches `20,480 CUDA cores / 128 = 160` from secondary reports. Some
  third-party DBs list a 144 GB B300 SKU at 148 SM; do not mix that SKU with the
  288 GB Blackwell Ultra row.
- `RTX PRO 6000 Blackwell`: source table rounded memory bandwidth to
  `1800 GB/s`; use `1792 GB/s`.
- `RTX 4090D`: source table was incomplete; completed core count/power class
  from public RTX 4090D information.
- `AMD MI*` rows were intentionally removed from this CUDA skill reference.
- Rows marked `rumored`, `incomplete`, or `non-official` should not be used for
  purchasing, export-control, or benchmark normalization without a direct
  vendor/system datasheet.

## CUDA Target Mapping

- Ampere datacenter A100/A800/A30: compile for `sm_80`.
- Ampere GA10x workstation/GeForce such as RTX 3090: compile for `sm_86`.
- Ada L40S/L40/L4/RTX 4090/RTX 6000 Ada/RTX 5000 Ada: compile for `sm_89`.
- Hopper H100/H800/H200/H20/GH200: compile for `sm_90`; use `sm_90a` only when
  intentionally using Hopper architecture-specific instructions.
- Blackwell datacenter B200/GB200: compile for `sm_100`. Blackwell Ultra
  B300/GB300 is `sm_103`. Use `sm_100a`, `sm_100f`, `sm_103a`, or `sm_103f`
  only when code requires Blackwell-specific/family features.
- Blackwell RTX/GeForce workstation cards use newer Blackwell RTX compute
  capability families such as `sm_120`; verify with `nvidia-smi --query-gpu=name,compute_cap`.

## Practical Use

- For memory-bound kernels, compare against `Mem BW` first. L40S and RTX 6000
  Ada can have high FP32 peaks but much less bandwidth than H100/H200/B200.
- For dense attention/GEMM rooflines, use dense Tensor Core values. Sparse or
  marketed AI TOPS are not interchangeable with dense FP16/BF16/FP8 TFLOPS.
- For multi-GPU training, separate host PCIe bandwidth, NVLink bandwidth, and
  NVSwitch topology. A card-level NVLink number does not imply all-to-all
  bandwidth without the matching platform.
- For exact runtime limits, query the actual machine:

```bash
nvidia-smi --query-gpu=name,compute_cap,memory.total,power.limit --format=csv
```

Then combine that with CUDA runtime properties:

```cpp
cudaDeviceProp prop{};
cudaGetDeviceProperties(&prop, device);
printf("%s sm_%d%d memBus=%d bits memClock=%d kHz\\n",
       prop.name, prop.major, prop.minor,
       prop.memoryBusWidth, prop.memoryClockRate);
```

## Public Sources

- NVIDIA CUDA GPU compute capability list:
  https://developer.nvidia.com/cuda-gpus
- CUDA Programming Guide, Compute Capabilities:
  https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capabilities
- NVIDIA Blackwell Tuning Guide:
  https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html
- NVIDIA Blackwell architecture resources:
  https://resources.nvidia.com/en-us-blackwell-architecture
- NVIDIA Technical Blog, Inside NVIDIA Blackwell Ultra:
  https://developer.nvidia.com/blog/inside-nvidia-blackwell-ultra-the-chip-powering-the-ai-factory-era/
- TechPowerUp B200 SXM 192 GB GPU database:
  https://www.techpowerup.com/gpu-specs/b200-sxm-192-gb.c4210
- TechPowerUp GPU database:
  https://www.techpowerup.com/gpu-specs/
- NVIDIA GB300 NVL72:
  https://www.nvidia.com/en-sg/data-center/gb300-nvl72/
- NVIDIA Grace Hopper Superchip:
  https://resources.nvidia.com/en-us-grace-cpu/grace-hopper-superchip
- NVIDIA H100:
  https://www.nvidia.com/en-us/data-center/h100/
- NVIDIA H200:
  https://www.nvidia.com/en-us/data-center/h200/
- NVIDIA A100:
  https://www.nvidia.com/en-us/data-center/a100/
- NVIDIA L40S:
  https://www.nvidia.com/en-us/data-center/l40s/
- NVIDIA L4:
  https://www.nvidia.com/en-us/data-center/l4/
- NVIDIA A30:
  https://www.nvidia.com/content/dam/en-zz/Solutions/data-center/products/a30-gpu/pdf/a30-datasheet.pdf
- NVIDIA RTX PRO 6000 Blackwell:
  https://www.nvidia.com/en-us/products/workstations/professional-desktop-gpus/rtx-pro-6000/
- NVIDIA RTX PRO 5000 Blackwell:
  https://www.nvidia.com/en-us/products/workstations/professional-desktop-gpus/rtx-pro-5000/
- NVIDIA GeForce RTX 5090:
  https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5090/
- NVIDIA GeForce RTX 4090:
  https://www.nvidia.com/en-us/geforce/graphics-cards/40-series/rtx-4090/
