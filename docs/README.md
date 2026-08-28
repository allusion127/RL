# docs — 문서 색인

`lpopt` 본문 문서 6편과 참고 자료의 진입점이다. 저장소 전체 개요는 루트
[`README.md`](../README.md)를 먼저 보라.

| 문서 | 한 줄 |
|---|---|
| [01_architecture.md](01_architecture.md) | 시스템 구조 — 패키지 모듈 전수, CLI 서브커맨드 표, TOML 덱 스키마, 데이터 레이아웃, 벤더 스냅샷 |
| [02_model_methodology.md](02_model_methodology.md) | 모델 방법론 — 집합체 표현, `cond_schema` 진화, PosValNet 구조·손실·물리 프라이어, 스플릿·정직 게이트, 챔피언 계보 |
| [03_search_policy_autoeng.md](03_search_policy_autoeng.md) | 탐색 캠페인 루프, 프런티어 궤적, 정책망 v1/v2 판정, 자동 엔지니어(autoeng), 설계공간 그물망 |
| [04_timeline_and_results.md](04_timeline_and_results.md) | 개발 타임라인 — 라운드별 사건·판정·수치의 시계열 (2026-07-16 ~ 08-20) |
| [05_report_index.md](05_report_index.md) | 리포트 색인 — `data/reports/` 사전등록·판정 문서 118건의 주제별 전수 목록 |
| [06_learned_loading_rules.md](06_learned_loading_rules.md) | **학습된 장전 규칙** — 결과 인자 사전, 규칙 24종(G/R/F군), DEAD 목록, 규칙 간 상충, 코드 구현 |

## 참고 자료

- [`reference/PWR_commercial_core_loading_pattern_engineering_rules_KO.md`](reference/PWR_commercial_core_loading_pattern_engineering_rules_KO.md)
  — 공개자료 기반 상용 PWR 장전 규칙 종합 보고서. 06 문서 §3 "채택/미채택" 판단의 원본 지식베이스.
- [`reference/EXCLUDED_LARGE_FILES.txt`](reference/EXCLUDED_LARGE_FILES.txt)
  — 용량 때문에 공개 저장소에서 제외된 리포트 아티팩트의 파일명·바이트 목록.
- [`dual_trunk_cyclen_isolation_DRAFT.md`](dual_trunk_cyclen_isolation_DRAFT.md)
  — 착수 보류된 설계 초안(dual-trunk cyclen 격리). **DRAFT — 판정된 결과가 아니다.**

모든 문서는 **정직성 규약**을 따른다 — 수치는 저장소 내 실측 리포트에서 인용하고 측정 일자를
병기하며, 실패한 실험과 기각된 가설도 성공과 같은 비중으로 기록한다.
