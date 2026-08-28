#!/bin/sh
# multi-type mesh sweep (P3a) — 15 buildable e-levels x 6 feeds = 90 cells,
# each scored at k = 2 / 3 / 4 fresh types.  5 shards x 4 torch threads.
#
# NOTHING here touches MASTER, the fleet, or data/store; the s1i ensemble is
# read-only.  See data/reports/mesh_multitype_20260818/PREREG_multitype_mesh_20260818.md
#
# The case table is computed ONCE (--cases-only) and the shards read it, so the
# C(37,4) enumeration is not repeated five times.
O=data/reports/mesh_multitype_20260818
C=$O/case_selection.csv
python mesh_multitype.py --out-dir $O --model s1i --tag _A --threads 4 --cases-from $C --e-targets 5.0,5.1,5.2 >$O/shA.out 2>&1 &
python mesh_multitype.py --out-dir $O --model s1i --tag _B --threads 4 --cases-from $C --e-targets 5.3,5.4,5.5 >$O/shB.out 2>&1 &
python mesh_multitype.py --out-dir $O --model s1i --tag _C --threads 4 --cases-from $C --e-targets 5.6,5.7,5.8 >$O/shC.out 2>&1 &
python mesh_multitype.py --out-dir $O --model s1i --tag _D --threads 4 --cases-from $C --e-targets 5.9,6.0,6.1 >$O/shD.out 2>&1 &
python mesh_multitype.py --out-dir $O --model s1i --tag _E --threads 4 --cases-from $C --e-targets 6.2,6.3,6.4 >$O/shE.out 2>&1 &
wait
echo "ALL SHARDS DONE"
