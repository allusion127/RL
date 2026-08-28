# Methodology Notes — KNGR 18-month Multicycle Analysis (N-411-FN-D301-011 Rev.01)

Short excerpts with page citations (PDF page = printed "Page N of 84" for pages 1-84;
Attachment/results pages 85-91 are printed as "Page 85 of 84" etc. — see README gap note).
Not a full transcription — see the source PDF for complete text.

## Computer codes (Sec. 3.2, p.8)

HP/Domain codes used, with DRF numbers:
- `cord1.3mod2` (CORD, bundle/cross-section processor) — DR-97-C149
- `rocs5.1mod0` (ROCS, 3-D nodal core simulator) — DR-97-C157
- `rocsedit1.2mod1` (ROCS edit/post-processor) — DR-97-C158
- `nconex1.2mod2` — DR-97-C153
- `centaur4.0mod1` — DR-97-C148

So yes: the core-follow / depletion engine is **ROCS** (3-D ROCS depletion calculations),
consistent with the task's assumption.

## Methodology overview (Sec. 5, p.10)

> "This multicycle analysis is to generate the physics data of KNGR core having 3983 MWth
> (core power) and 555 F moderator inlet temperature through 3-D ROCS depletions. 3-D ROCS
> depletions were performed for each cycle after completing loading pattern search. This
> analysis is based on the groundrules of YGN-3/4 multi-cycle analysis (Reference 3)."

Loading-pattern search procedure for a succeeding cycle (p.10):
1. Find a loading pattern with relative power density less than ~1.30.
2. Perform the depletion calculation.
3. Search for a loading pattern with Fxy < 1.55 if possible.

Design requirements for the searched core (p.10, repeated at Sec. 6.1 p.11):
1. Target Fxy: 1.55
2. Cycle length: 468 EFPD (= 365 x 1.5 x 0.9 x 0.95, where 0.9 = availability, 0.95 = load factor)
3. Most positive MTC(HFP) < 0 x 10^-4 /F

## 6.1 Core Loading Pattern Search (p.11)

> "The core loading pattern for Cycle 1 through 8 were searched to meet [the above]
> requirements... Table 6.1 lists the information of feed fuel assemblies and Figure 6.1
> shows the typical fuel assembly configurations loaded in Cycles 1 through 8 of KNGR.
> Figures 6.2 through 6.9 show the fuel assembly loading patterns from cycle 1 to cycle 8."

Sec. 6.2 (Feed assemblies / tablesets, p.11): number of feed assemblies is 80 for cy2 and 92
for cy3-cy8 (cy1 = full initial core, 241). Applying a load factor of 0.95 targets 468 EFPD;
actual EFPD obtained: cy1=454, cy2=366, cy3=463, cy4=473, cy5=458, cy6=463, cy7=466, cy8=464
(near-equilibrium, slightly below target — see Sec. 6.6 discussion).

## 6.3 ROCS Model and Input Preparation (p.11)

> "The ROCS model of cycle 1 had been used for calculations of following cycles. The major
> changes in ROCS model for the calculation of subsequent cycles will be a batch and
> composition assignment."

## 6.4 Radial Albedo Iteration (p.12)

> "The ROCS radial albedo boundary condition of PVNGS given as an initial guess can be
> updated through ROCS iteration jobs at BOC to match with their MC albedo file. However,
> these update of albedo set did not have much effect on reactivity and power distribution.
> And, the target MC albedo file was not come from KNGR, but come from PVNGS. Thus, at
> present, a ROCS albedo iteration job will be meaningless. Therefore, the radial albedo
> set (see Reference 1) of PVNGS will be also used for this KNGR multicycle analysis."

i.e. no independent albedo iteration was performed for KNGR — the PVNGS (Palo Verde) radial
albedo set was reused as-is, on the basis that KNGR shares PVNGS's core size/geometry
(consistent with the Sec. 4 Assumptions, p.9, which also borrows PVNGS assembly weighting
factors/shape annealing functions and YGN-3/4 axial boundary conditions).

## 6.5 ROCS Depletion Calculation (p.12-13)

Depletion run per cycle produces: Summary Edit 1&2 (Tables 6.2-6.9), pin peaking factors
(Tables 6.10-6.17, "raw"), axial power distributions (Tables 6.18-6.25), reactivity/CBC vs.
burnup (Tables 6.26-6.33), best-estimated pin peaking factors with a 1.0042 bias applied to
the raw Fr/Fxy/Fq values (Tables 6.34-6.41), IBW & MTC (Table 6.42), and CBC/CDROM file index
(Table 6.43). MOC (maximum-Fxy maneuver step) burnups per cycle are listed explicitly (p.12-13):
cy1=7028, cy2=11908, cy3=11975, cy4=10988, cy5=9995, cy6=11997, cy7=10999, cy8=11000 MWD/T.

## 6.6 Evaluation of Multicycle Loading Pattern Search (p.14)

> "The calculation of MTC is distributed from -1.16x10-4/F to -0.23x10-4/F at BOC and is
> distributed from -3.25x10-4/F to -2.13x10-4/F at EOC as shown in Table 6.42. These values
> of MTCs are satisfied with target value of 0.0x10-4/F."
>
> "The maximum pin peaking factors of Fxy in each cycle are 1.4950 in cy1, 1.5697 in cy2,
> 1.5569 in cy3, 1.5398 in cy4, 1.5516 in cy5, 1.5488 in cy6, 1.5422 in cy7 and 1.5372 in
> cy8. The calculated values of Fxy are lower than 1.55 except of cy2. This larger peaking
> factor can be by-passed through an appropriate assembly rotation in a KNGR final design
> stage."
>
> "The cycle length can be almost reached to a target cycle length (468 EFPD) near an
> equilibrium cycle. This small insufficiency to target cycle length will be also satisfied
> by an appropriate consideration for an enrichment selection of feed assembly in a KNGR
> final design stage. So, these loading patterns searched for KNGR multicycle analysis are
> judged to be satisfied considering the above design criteria and requirement."

Overall conclusion: the searched loading patterns for cy1-cy8 are judged acceptable against
the Fxy, cycle-length, and MTC criteria, with two flagged (not corrected) shortfalls carried
forward to final design: cy2 Fxy slightly exceeds 1.55 (1.5697), and cycle lengths run
slightly under the 468 EFPD target.

## 8. Results (p.84-88)

Results are pointed to Table 8.1 (loading pattern/ROCS results index per cycle), Table 8.2
(core depletion summary per cycle: assembly-type inventory, cycle length in MWD/MTU-EFPD-EFPH,
feed FA count, U mass, core avg. enrichment, Fq/Fxy/Fr max-raw, max rod burnup), and Table 8.3
(kinetics parameters — beta-effective / precursor data — by cycle, not extracted into the CSVs
per the task's scope).
