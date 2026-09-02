# docs — 문서 색인

`lpopt` 본문 문서 6편과 참고 자료의 진입점이다. 저장소 전체 개요는 루트
[`README.md`](../README.md)를 먼저 보라.

| 문서 | 한 줄 |
|---|---|
| [01_architecture.md](01_architecture.md) | 시스템 구조 — 패키지 모듈 전수, CLI 서브커맨드 표, TOML 덱 스키마, 데이터 레이아웃, 벤더 스냅샷 |
| [02_model_methodology.md](02_model_methodology.md) | 모델 방법론 — 집합체 표현, `cond_schema` 진화, PosValNet 구조·손실·물리 프라이어, 스플릿·정직 게이트, **`F_xy` 헤드 arm 1–3 · 서빙경로 결함과 보정 재적합**, 챔피언 계보(11세대) |
| [03_search_policy_autoeng.md](03_search_policy_autoeng.md) | 탐색 캠페인 루프, 프런티어 궤적, **`min_fxy` 목적 · 안전 실드 · 납품 판정 · 개입 웨이브**, 정책망 v1/v2 판정과 v2 서빙, 자동 엔지니어(autoeng), 설계공간 그물망, **외부 검토 격차표** |
| [04_timeline_and_results.md](04_timeline_and_results.md) | 개발 타임라인 — 라운드별 사건·판정·수치의 시계열 (2026-07-16 ~ 08-31, 사고·기각 장부 포함) |
| [05_report_index.md](05_report_index.md) | 리포트 색인 — `data/reports/` 사전등록·판정 문서 138건의 주제별 전수 목록 |
| [06_learned_loading_rules.md](06_learned_loading_rules.md) | **학습된 장전 규칙** — 결과 인자 사전, 규칙 24종(G/R/F군), **`F_xy` 시대에 확정된 규칙(M~S)**, DEAD 목록, 규칙 간 상충, 코드 구현 |

## 참고 자료

- [`reference/PWR_commercial_core_loading_pattern_engineering_rules_KO.md`](reference/PWR_commercial_core_loading_pattern_engineering_rules_KO.md)
  — 공개자료 기반 상용 PWR 장전 규칙 종합 보고서. 06 문서 §3 "채택/미채택" 판단의 원본 지식베이스.
- [`reference/EXCLUDED_LARGE_FILES.txt`](reference/EXCLUDED_LARGE_FILES.txt)
  — 용량 때문에 공개 저장소에서 제외된 리포트 아티팩트의 파일명·바이트 목록.
- [`dual_trunk_cyclen_isolation_DRAFT.md`](dual_trunk_cyclen_isolation_DRAFT.md)
  — 착수 보류된 설계 초안(dual-trunk cyclen 격리). **DRAFT — 판정된 결과가 아니다.**

**갱신 범위 (2026-08-31).** 여섯 문서 전부가 **2026-08-29 목적축 전환(`F_r` → `F_xy`)** 이후의
작업을 반영한다 — `F_xy` 데이터층·헤드 arm 1–3, 서빙 경로 피처화 결함과 보정 재적합,
11대 챔피언 `s1j` 승격, `min_fxy` 첫 캠페인과 **첫 납품 가능 노심**, 개입 웨이브 r1과
HGD569 축퇴 결함, 안전 실드(OOD/conformal), 그리고 외부 기술 검토(2026-08-29)에 대한 응답.

모든 문서는 **정직성 규약**을 따른다 — 수치는 저장소 내 실측 리포트에서 인용하고 측정 일자를
병기하며, 실패한 실험과 기각된 가설도 성공과 같은 비중으로 기록한다.
