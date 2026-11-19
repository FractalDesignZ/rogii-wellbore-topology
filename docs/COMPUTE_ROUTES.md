# Compute routes for candidate depth paths

The RTX 5070 evaluated 15,625 possible depth paths on the 416 development wells with sufficient evidence. The run scored all 619 wells and took 47.45 seconds from contract freeze through independent audit. The 25-path comparison was more accurate on the same observations. The field result measures both the cost of searching a wider set of paths and the accuracy of the resulting depth estimates.

## What moves between CPU and GPU

| Key or tensor axis | Meaning and units | Route in the field run |
| --- | --- | --- |
| Well, original station, channel | Survey row in acquisition order, measured depth (MD) in feet, gamma ray (GR) in the recorded curve units, visible depth and validity | CPU preserves row identity, units and observation role. The paired reference has its own depth axis. |
| Section, bin | An original 512-MD-ft interval containing eligible 32-MD-ft GR bins. Section identity survives a change of processing order. | CPU registers the reference and creates a common finite-bin mask across all three candidate grids. |
| Candidate, knot | A complete hypothesized trajectory and its depth displacements in feet. The 625- and 15,625-path grids have four and six adjustable knots. | CUDA evaluates tiled FP64 arrays of section, candidate and bin. Supported reference observations can be reused wherever candidate paths share the same local geometry. |
| Section, candidate | Loss comparing observed and candidate GR, signed disagreement and relative ordering of the alternatives | CUDA combines evidence across sections and returns the estimated displacements. CPU also evaluates the 25-path comparison and checks selected GPU outputs. |
| Well, original predicted station | One true vertical thickness (TVT) forecast in feet per native station | CPU restores native row order, seals predictions and independently replays squared errors against the original lateral rows. |

Candidate number denotes a physical hypothesis. Tile number denotes where a subset fits in device memory. The score uses the rank of a candidate across the full supported grid. An implementation that independently ranks each tile would change the method. The public harness describes the observation and target roles for the wider graph, tree, forest and neural comparison. [Nodal harness](NODAL_HARNESS.md)

## Single-well numerical benchmark

One well provided 170 GR bins in 11 sections. The benchmark increased the number of allowed depth levels at each of four knots. Each device returned the same eligible bins and section IDs as an independent FP64 CPU calculation. At 83,521 paths the maximum absolute discrepancy across 918,731 section–path entries was 1.33 × 10⁻¹⁵ in the normalized loss and 6.44 × 10⁻¹⁵ in signed disagreement. The benchmark used observed curves and a frozen geometric prediction. It produced no TVT accuracy score.

| Four-knot paths | CPU reference | RTX 5070 | GTX 1660 Ti | CUDA tile | Peak allocation above entry on either GPU |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 625 | 0.0130 s | 0.00746 s | 0.00739 s | 625 | 10.9 MiB |
| 83,521 | 1.837 s | 0.04047 s | 0.12542 s | 8,192 | 193.8 MiB |

Each GPU time includes host-to-device transfer, calculation of the two evidence matrices, device-to-host transfer and synchronization. CPU reference time also includes computing the posterior mean. The table gives individual warm observations. The first timed RTX call, with a 128-path tile, took 15.92 s. The GPU worker took 21.03 s in total across both devices and all tile sizes. These two timings matter when a caller starts a fresh process for a single small job. Numerical benchmark evidence root `52d54998560660f6bfc64840060e50e3d13472c8d974477379594116f9f2c2d3`.

## Field cost and accuracy

The field run used the RTX 5070, FP64 and 1,024-path tiles. Common support covered 63,577 GR bins across 4,109 sections, representing 2,046,530 native prediction rows on 416 wells. The other 203 wells retained the frozen geometry prediction. All 619 wells and 3,034,427 native station rows contribute to the following scores.

| Depth hypotheses and control | Pooled RMSE (ft) |
| --- | ---: |
| 25 paths, original support | 9.997405 |
| 25 paths, common support | 9.996844 |
| 625 paths, common support | 10.039549 |
| 15,625 paths, common support | 10.107876 |
| 15,625 paths, recurrence disabled | 10.110202 |
| 15,625 paths, observed GR rotated across eligible bins | 10.202485 |

Enlarging the grid from 25 to 15,625 paths under the specified priors and shared mask raised pooled RMSE by 0.111033 ft. All three whole-well folds showed an increase. The paired whole-well 95% interval for the increase was 0.056721 to 0.166756 ft. The six-knot model failed its predictive gate. Independent CPU checks of expected knot depths covered the four-knot readout on all 416 supported wells and the six-knot alternatives on three of them. The maximum difference was 3.20 × 10⁻¹⁴ ft.

Multiplying bins by six-knot paths gives **993,390,625 logical bin–path slots**. The corresponding section count is 64,203,125 section–path slots. Reuse of local reference values and padded batches changes the number of physical instructions. CUDA events summed over the field calculation span **16.07 s**. The inference loop, including host work and CPU parity checks, took **23.51 s**. The four-process freeze, forecast, score and audit took **47.45 s**. Peak PyTorch allocation was **72.52 MiB**, with **90 MiB** reserved by its allocator. These clocks overlap. The field run provides no CPU-only whole-cohort timing from which to calculate a field speedup.

Hardware reports 48 streaming multiprocessors on the RTX 5070 and 24 on the GTX 1660 Ti. Measuring achieved occupancy, memory traffic or tensor-core utilization would require profiling counters. The field forecast seal is `ee2ad030e0cff5004cc0d2195f85dc180dc3d46a82b9c29acefaea087cac57fa`. The verified execution-chain head is `5aabea722482188127aff77e022864f9999b1d0edb85602890989fc21f0871b7`.

## Proposed graphics routes

| Proposed use | Geometry and precision test before field scoring |
| --- | --- |
| A compute shader projects observed survey segments into a registered image or evaluates valid pixel distances along a proposed trace. Return the source station/segment, view ID, pixel coordinate and validity for each result. | Replay the CPU projection and inverse on synthetic axes, negative and large world coordinates, clipped segments, missing samples, and overlapping traces. Check pixel-center convention, image orientation, metric units and rounding at a threshold. A duplicate rendering must retain the original survey ancestry. [Registered image views](REGISTERED_IMAGE_VIEWS.md) |
| A ray query finds intersections of a directed candidate segment with an explicitly supported plane or triangular surface. Return signed distance, primitive ID and source provenance. | Compare against analytic CPU intersections for translated and rotated geometry, parallel and grazing rays, hits behind the origin, shared triangle edges, multiple hits, and gaps in surface support. Declare nearest-hit or all-hit selection, distance units, tolerance and tie handling before evaluation. Test whether lower precision changes hit identity or the decoded depth. |

For either proposed route, count supported queries and unresolved cases on identical well and station IDs. Include geometry construction, uploads, synchronization, readback and any acceleration-structure rebuild in wall time. A depth improvement requires its own paired field comparison following a numerical and identity check.
