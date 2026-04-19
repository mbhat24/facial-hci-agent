# Research Foundations

## FACS — Facial Action Coding System
- Ekman, P., & Friesen, W. V. (1978). *Facial Action Coding System*.
- 30+ discrete Action Units (AUs). Basis of all modern emotion-from-face work.

## Emotion ↔ AU mapping
| Emotion    | Core AUs                       |
|-----------|--------------------------------|
| Happiness | AU6 + AU12                     |
| Sadness   | AU1 + AU15 + AU17              |
| Fear      | AU1 + AU2 + AU4 + AU5 + AU20   |
| Surprise  | AU1 + AU2 + AU5 + AU26         |
| Disgust   | AU9 + AU10 + AU15              |
| Anger     | AU4 + AU5 + AU7 + AU23         |
| Contempt  | AU14 + AU24                    |

## Key papers
- Sălăgean, Leba, Ionica (2025). *Real-Time Micro-Expression Recognition with
  Action Units and GPT-Based Reasoning.* Applied Sciences 15(12):6417.
  → Hybrid rule+LLM achieves 93.3% on CASME II.
- Yan et al. (2014). *CASME II: An Improved Spontaneous Micro-Expression
  Database.* PLOS ONE.
- Barrett, L. F. et al. (2019). *Emotional Expressions Reconsidered.*
  Psychological Science in the Public Interest.
- Baltrusaitis, T. et al. (2018). *OpenFace 2.0: Facial Behavior Analysis.*
- Cheong et al. (2023). *Py-Feat: Python Facial Expression Analysis Toolbox.*

## Cognitive state cues (HCI literature)
- **Attention**: head pose stability + forward gaze.
- **Cognitive load**: brow furrow (AU4) + reduced blink rate + pupil dilation.
- **Confusion**: AU4 + AU7 + head tilt.
- **Stress**: elevated blink rate + AU4 + jaw tension (AU17/AU23).
- **Boredom**: minimal AU activity + slow blinks + gaze aversion.
