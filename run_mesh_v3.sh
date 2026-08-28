#!/bin/sh
# mesh v3 sweep — 15 buildable e-levels x 6 feeds = 90 model-only cells.
# 5 shards x 4 torch threads.  NOTHING here touches MASTER, the fleet, or
# data/store; the champion ensemble is read-only.  See
# data/reports/mesh_v3_20260817/PREREG_mesh_v3_20260817.md
O=data/reports/mesh_v3_20260817
python scoping_mesh.py --out-dir $O --model s1g --tag _A --threads 4 --e-targets 5.0,5.1,5.2 >$O/shA.out 2>&1 &
python scoping_mesh.py --out-dir $O --model s1g --tag _B --threads 4 --e-targets 5.3,5.4,5.5 >$O/shB.out 2>&1 &
python scoping_mesh.py --out-dir $O --model s1g --tag _C --threads 4 --e-targets 5.6,5.7,5.8 >$O/shC.out 2>&1 &
python scoping_mesh.py --out-dir $O --model s1g --tag _D --threads 4 --e-targets 5.9,6.0,6.1 >$O/shD.out 2>&1 &
python scoping_mesh.py --out-dir $O --model s1g --tag _E --threads 4 --e-targets 6.2,6.3,6.4 >$O/shE.out 2>&1 &
wait
echo "ALL SHARDS DONE"
