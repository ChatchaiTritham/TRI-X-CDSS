# Triage–TiTrATE–XAI (TRI-X) — Concept & Methodology Specification

> **Canonical concept note** for the TRI-X work. Single source of truth that governs
> both the **manuscript** (`Manuscript/.../TRI-X*`) and the **GitHub repositories**
> (`GitHub/TRI-X`, `GitHub/TRI-X-CDSS`). When prose, claims, figures, or code drift,
> reconcile against *this* file.
>
> - **Scope:** decision-governance CDSS for *dizziness & vertigo* under clinical uncertainty.
> - **Data:** Synthetic only — **no real patient data, no single-expert bias.**
> - **Stance:** not "how accurately can we predict?" but **"how should the system behave when it is not sure?"**
> - **Last synced from:** `.docs/Triage-TiTrATE-XAI Framework.txt`

---

## 0. One-paragraph summary (EN)

TRI-X fuses three frames into one auditable framework to handle diagnostic
uncertainty systematically: **clinical Triage (ESI)** for urgency/acuity,
**TiTrATE** (Timing–Triggers–And–Targeted-Examination) for structured differential
reasoning, and **Explainable AI (XAI)** as a *design principle from the data layer
onward* (SHAP/LIME, NMF, counterfactual reasoning). Built on transparent **synthetic
data**, TRI-X formalizes clinical guidelines into checkable logic, treats
**uncertainty as a decision-control signal** rather than something to hide, and sorts
patients into **5 urgency/risk tiers** — from "go to ER now" (e.g. stroke
thrombolysis window) to "safe / non-urgent". It is positioned as a foundation for
**Clinical AI Governance** and for future retrospective/prospective studies with
multidisciplinary clinicians.

---

## 1. ปัญหาตั้งต้น (Problem framing)

อาการ **เวียนศีรษะและบ้านหมุน (dizziness & vertigo)** ถูกเลือกเป็น *กรณีที่มีความไม่แน่นอนสูง* เพราะ:

- ผู้ป่วยมัก **เล่าอาการไม่ครบ / คลาดเคลื่อน / อธิบายไม่ได้เลย**
- เป็น **differential diagnosis ที่กว้าง** — ตั้งแต่ไม่อันตราย จนถึงภาวะคุกคามชีวิต
- การวินิจฉัยขึ้นกับ **ความเชี่ยวชาญเฉพาะสาขา** (GP, ER, ENT, Neurology, Psychiatry …) ซึ่ง **ไม่มีกลุ่มใดครอบคลุมทุกสาเหตุ**

→ ความไม่แน่นอนจึงเป็น *ปัญหาเชิงโครงสร้าง* ไม่ใช่ noise ที่ควรลบทิ้ง

---

## 2. แนวคิดวิธีการใหม่ (Methodological novelty)

### 2.1 ขยายจาก TiTrATE → Triage–TiTrATE–XAI

| Layer | กรอบเดิม | TRI-X เพิ่มอะไร |
|---|---|---|
| Differential reasoning | **TiTrATE** (Timing, Triggers, Targeted Examination) | คงไว้เป็นแกน clinical reasoning |
| Urgency / acuity / life-risk | *(ขาด)* | **+ Triage (ESI)** → ความเร่งด่วน, ความเสี่ยงต่อชีวิต, การส่งต่อ |
| Explainability / auditability | *(ขาด)* | **+ XAI by-design** (ดู §2.3) |

หลักการร่วม:
> **Clinical guideline สามารถ formalize เป็น logic ที่ตรวจสอบได้** และ logic เหล่านี้เป็น **แกนกลางของ CDSS**

### 2.2 Uncertainty-first design — ความไม่แน่นอนเป็นแกนกลาง

แนวคิดที่สำคัญที่สุด: **ไม่ลบ/ไม่ซ่อนความไม่แน่นอน แต่ทำให้มันเป็น "decision control signal"**

แหล่งความไม่แน่นอน: ข้อมูลผู้ป่วยไม่ครบ/คลาดเคลื่อน · อาการไม่ชัด · ความต่างของความชำนาญแพทย์

แทนที่จะยุบเหลือ "คะแนนความมั่นใจ" ความไม่แน่นอนกำหนด *พฤติกรรมการตัดสินใจ*:

- ควร **ส่ง ER ทันที** (เช่น stroke window — สลายลิ่มเลือดให้ทันภายในกรอบเวลา)
- ควร **ส่งต่อผู้เชี่ยวชาญภายในกรอบเวลา** (Specialist OPD นัดหมาย)
- หรือ **ปลอดภัย / ไม่เร่งด่วน**

> เปลี่ยนบทบาท data science จาก **prediction → decision governance**

### 2.3 XAI-by-design — ใช้ XAI ตั้งแต่ระดับข้อมูล ไม่ใช่แค่ปลายทาง

| เทคนิค | บทบาทใน TRI-X |
|---|---|
| **SHAP / LIME** | อธิบาย reasoning ของ decision logic เชิงคุณลักษณะ |
| **NMF** (Non-negative Matrix Factorization) | สกัดโครงสร้างปัจจัยของอาการ / รูปแบบการนำเสนอโรค |
| **Counterfactual reasoning** | วิเคราะห์เงื่อนไขสมมติที่ทำให้การตัดสินใจ *เปลี่ยน* |

> **Implementation status (resolved 2026-06-27):** XAI ครบทุกตัวที่อ้าง — **SHAP / LIME / DiCE (counterfactual)** ระดับ output (`src/trix/empirical/explain.py`)
> และ **NMF** ระดับ data (`src/trix/empirical/nmf.py`, `factorize_symptoms`) สกัด symptom-pattern factors บน synthetic cohort, seed-42, deterministic
> (recon_err=90.748854, byte-stable), เขียนผล `results/nmf_factors.json` + สรุปใน `results/explainability.json`. ผลตรง clinical: F1→BPPV(positional 73%),
> F4→stroke/TIA(vascular risk), F0→neuritis/labyrinthitis. **integrity gap §5/R3 ปิดแล้ว** — คำอ้าง "SHAP/NMF/CF" ใน manuscript มี backing จริง.
>
> **Reconcile NMF r=20 vs k=6 (2026-06-27):** ESA/JIIS อ้าง NMF 2 ระดับ — (1) *design target* `r=20` บน 150-param/287-feature จาก **SynDX generator (คนละ paper → DMKD/KAIS)**, มี code ใน `GitHub/SynDX` แต่ ESA **ไม่ได้ใช้ประเมิน**; (2) *released* `k=6` บน 22-feature cohort (n=5000) = ที่รันจริง รายงานใน `tab:nmf_factors`. ESA มี `\begin{clinicalnote}[Scope]` ระบุชัดว่า 150/287/8400/r=20 เป็น design target ไม่ใช่ระบบที่ประเมิน + เพิ่ม bridge note ใน implementation.tex/methodology.tex ชี้ `tab:nmf_factors` แล้ว → **ไม่ขัดกัน, defensible**. headline 79.6%/96.6%/99.3% มาจาก TRI-X repo (22 feat, n=5000) ไม่ใช่ SynDX 8400.
>
> **Unbacked metrics ลบแล้ว (2026-06-27):** χ² "epidemiological fidelity p=0.087" และ "clinical expert review 92% realistic (n=100)" **ไม่มี code/data backing + ขัด stance no-expert-bias/no-real-patient** → ลบทั้ง 3 จุด (implementation.tex caption+itemize, figure1_architecture.tex node) แทนด้วยภาษา design-intent/provenance. McNemar table ใน supplementary.tex ยังอยู่ (มาจาก `results/*.json` จริง, ระบุ no physician test). recompile 88pp clean.

จุดเด่นเชิงจริยธรรม/governance:
- ใช้ **Synthetic Data** ที่ไม่มีข้อมูลผู้ป่วยจริง
- **ไม่มี bias** จาก expert รายใดรายหนึ่ง
- Logic + dataset **โปร่งใส ตรวจสอบ และอธิบายได้ทั้งหมด** (traceable end-to-end)

### 2.4 TRI-X System — การจัดกลุ่มเชิงพฤติกรรมการตัดสินใจ (5 tiers)

จาก synthetic dataset + logic ข้างต้น พัฒนาเป็น **TRI-X System** ที่แบ่งผู้ป่วยเป็น **5 กลุ่ม**
ตามความเร่งด่วน × ความเสี่ยง:

| Tier | นิยามเชิงพฤติกรรม (decision behavior) | Action |
|---|---|---|
| **1** | เร่งด่วนสูงสุด / คุกคามชีวิต — high-risk window | **ER ทันที** (เช่น stroke ≤ time window) |
| **2** | เร่งด่วน / เสี่ยงสูง | ส่ง ER / ประเมินด่วน |
| **3** | กลาง — ต้องติดตามโดยผู้เชี่ยวชาญ | นัด **Specialist OPD** ในกรอบเวลา |
| **4** | ต่ำ | ดูแลแบบผู้ป่วยนอก / นัดติดตาม |
| **5** | ปลอดภัย / ไม่เร่งด่วน | ไม่ต้องมา ER, คำแนะนำดูแลตนเอง |

> ระบบ **ไม่ทำนายโรคอัตโนมัติ** — แต่ **กำกับพฤติกรรมการตัดสินใจ** ของ CDSS ให้ปลอดภัยและตรวจสอบได้
> (จัดลำดับคิว · ส่งต่อ · นัด specialist · บริหารทรัพยากร · ลดความแออัด ER)

---

## 3. การประเมินและธรรมาภิบาล (Evaluation & Clinical AI Governance)

งานวิจัยลำดับถัดมาเน้น **Evaluation + Clinical AI Governance** โดย *จงใจหลีกเลี่ยง* การอ้างผลคลินิกจากข้อมูลจริงในระยะต้น
การประเมินจึงอธิบาย **"พฤติกรรมการตัดสินใจของระบบภายใต้ความไม่แน่นอน"** ผ่านตัวชี้วัดที่ **เกินกว่า accuracy**:

- **Calibration** — ความสอดคล้องของความมั่นใจกับความจริง
- **Coverage–risk trade-off** — ครอบคลุมเท่าไร แลกกับความเสี่ยงเท่าไร (abstention)
- **Harm-aware outcomes** — ผลลัพธ์ที่ถ่วงน้ำหนักอันตราย (เช่น false-discharge ของ high-risk)
- **XAI consistency** — คำอธิบายสอดคล้องกันและตรวจสอบได้

วางตำแหน่งเป็น **ฐานราก** สำหรับการศึกษา **retrospective** และ **prospective** ในผู้ป่วยจริงร่วมกับแพทย์หลายสาขาในอนาคต
เป้าหมาย: **Clinical AI ที่ปลอดภัย ตรวจสอบได้ และอยู่ภายใต้ธรรมาภิบาลทางคลินิก**

---

## 4. ประโยชน์ต่อระบบสุขภาพและสังคม (Impact)

1. **Patient Safety** — ลดวินิจฉัยผิดใน high-risk window (stroke), ลดคำแนะนำที่ "มั่นใจเกินจริง" เมื่อข้อมูลไม่ครบ, ลด **automation bias**
2. **ลดความเหลื่อมล้ำจากความเชี่ยวชาญ** — ให้ GP/ER ใช้ logic ระดับผู้เชี่ยวชาญร่วมกัน; รู้ว่า **"เมื่อไรควรส่งต่อ"** และ **"เมื่อไรควรงดตัดสินใจ"**
3. **Clinical AI Governance** — ไม่ใช้ข้อมูลผู้ป่วยจริง · logic ตรวจสอบย้อนหลังได้ · แยก decision behavior ออกจาก performance metric
4. **ต้นแบบ Data Science for Healthcare รุ่นใหม่** — ถ่ายทอดไปยังเวชฉุกเฉิน, โรคหายาก, ระบบปฐมภูมิ, บริบททรัพยากรจำกัด

---

## 5. การกำกับ Manuscript + Repos (Governance checklist)

ใช้ไฟล์นี้เป็นเกณฑ์ตรวจ — ทุก artifact ต้องสอดคล้องกับข้อความข้างต้น:

**Manuscript ต้อง:**
- [ ] กรอบเป็น **Triage–TiTrATE–XAI** (ไม่ใช่ TiTrATE เดี่ยว) ครบทั้ง 3 layer (§2.1)
- [ ] ระบุชัดว่า **uncertainty = decision-control signal** (§2.2), ไม่ใช่แค่ confidence score
- [ ] อธิบาย **XAI-by-design** + ระบุ SHAP/LIME, NMF, counterfactual (§2.3)
- [ ] ระบุ **synthetic data, no real patient, no single-expert bias** ทุกครั้งที่กล่าวถึงข้อมูล
- [ ] นำเสนอ **5-tier decision behavior** (§2.4) — *ไม่ใช่* การทำนายโรค
- [ ] Evaluation รายงาน **beyond-accuracy** (calibration / coverage-risk / harm-aware / XAI consistency) (§3)
- [ ] วาง retrospective/prospective เป็น **future work** เท่านั้น (ไม่อ้างผลคลินิกจริงตอนนี้)

**GitHub repos ต้อง:**
- [ ] README + CITATION สอดคล้องกับกรอบ §1–§4 (ไม่มี claim เกินจากนี้)
- [ ] data generator = **synthetic, seed-fixed, reproducible**; ไม่มีไฟล์ผู้ป่วยจริง
- [ ] โค้ดมีโมดูล triage logic + TiTrATE logic + XAI (SHAP/LIME/NMF/counterfactual) ตรงกับที่ manuscript อ้าง
- [ ] figure/results ที่อ้างใน manuscript **generate ได้จาก repo** (ไม่ hardcode ในสคริปต์รูป) และตัวเลข **ตรงกับ** manuscript
- [ ] 5-tier output ปรากฏในระบบจริง (ไม่ใช่แค่ binary)

> ⚠️ **Integrity note (จาก audit):** เคยพบ repo มีสคริปต์รูป/SHAP/ensemble ที่ *ไม่อยู่ใน method จริง* — ห้าม commit artifact ที่ขัดกับสเปกนี้ ก่อนแก้ให้ตรวจ §5 ทุกข้อ

---

## 6. Glossary

- **ESI** — Emergency Severity Index (5-level triage)
- **TiTrATE** — Timing, Triggers, And Targeted Examination (vertigo reasoning framework)
- **TRI-X** — Triage–TiTrATE–XAI (this framework / system)
- **NMF** — Non-negative Matrix Factorization
- **CDSS** — Clinical Decision Support System
- **Abstention / coverage** — การงดตัดสินใจเมื่อไม่แน่ใจ และสัดส่วนเคสที่ระบบยอมตัดสิน
