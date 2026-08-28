# verify_hires_198.ps1 -- post-restart check: are __traj/__axial keys landing?
$k='C:\Users\USER\lpopt_work\kit_frontier'
& "$k\venv\Scripts\python.exe" -c @"
import numpy as np
z=np.load(r'$k\data\store\maps.npz')
f=z.files
t=[k for k in f if k.endswith('__traj')]; a=[k for k in f if k.endswith('__axial')]
print('total_keys',len(f),'traj',len(t),'axial',len(a))
if t: print('traj shape',z[t[0]].shape)
if a: print('axial shape',z[a[0]].shape)
print('VERDICT:', 'HIGH-RES LANDING' if (t and a) else 'NOT YET (wait one wave)')
"@
