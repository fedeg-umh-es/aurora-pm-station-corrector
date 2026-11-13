# P1 decisions (recorded before any forecast-skill result exists)

These decisions were taken after P1 eligibility was computed and before any
Aurora/CAMS comparison. They use only coverage and metadata, never skill.

1. **Registered criteria are not amended.** Primary result: 9/37 stations
   (PILOT_ONLY). The registered sensitivity set (26/37) is reported as a
   robustness set only. Changing thresholds after seeing the September-2023 gap
   would be a post-hoc amendment.
2. **September 2023 is not filled.** The only alternative official source (EEA
   e-Reporting) is heterogeneous (validated data, different station identifiers).
   Mixing sources inside one series would confound the experiment. Open item for
   future work, not a blocker.
3. **Claim scope follows from (1).** With PILOT_ONLY, no regional claim is made.
   Conclusions are stated for the pilot stations only; the sensitivity set may
   support a clearly labelled, weaker regional statement.
4. **Pilot stations for Phase C** (rule in `scripts/p1_02_select_pilot_stations.py`):
   primary-eligible AND `coord_status == "ok"`. Output:
   `data/metadata/phase_c_pilot_stations.csv`. The two primary-eligible stations
   with duplicated catalogue coordinates stay in the eligibility table but are
   excluded from spatial extraction until their coordinates are verified.
5. **Time convention.** Working convention is UTC interval end (inferred, see
   `docs/gva_source_notes.md`). Phase C must include a sensitivity run with
   observations shifted by ±1 h; a conclusion that changes under that shift is
   not reportable. Confirmation from the GVA is required before submission.
6. **Tail counts.** `q90/q95_train_candidate_count` stay in the output but must
   not be cited as tail-sufficiency evidence. Tail sufficiency is assessed
   per rolling-origin fold from the train window.
7. **Data validation status is unknown** (no flag in the source). This is a
   stated limitation in any publication.
