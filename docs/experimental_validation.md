# Experimental validation and regression

Experiment 06 used the frozen G2-11 threshold `0.07826304331214909` and the Turkish INT8 model. It reported 3473 Test records, gate WER 10.0144%, CER 2.7444%, 17.94% retry and 1.6755× 3000-baseline speedup. These figures describe the original synchronized experiment, not measurements of this package and not guaranteed performance.

An offline exact check against all 3473 archived decoder feature rows was performed: all 3473 risk scores and retry decisions matched exactly. This does not validate model execution or transcript reproduction.

The development environment reports zero CUDA devices. CPU INT8 failed because CT2 does not support efficient INT8 here; CPU float32 loaded but inference failed with `No SGEMM backend on CPU`. Thus actual encoder execution, live transcription and the GPU/INT8 regression are **not verified in this environment**. The opt-in regression test provides exact stage-1/final text and decision checks, `1e-6` feature/risk tolerance, shape checks and word-error totals for archived records. `KORKUTWHISPER_REGRESSION_ALL=1` expands the sample to all records. These tests must run successfully on the actual compatible GPU before claims of reproduced inference are made.
