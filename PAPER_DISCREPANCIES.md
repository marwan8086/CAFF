# Paper Discrepancies ? To Address Before Resubmission
This document tracks inconsistencies discovered between the paper text
and the implementation. Each item should be resolved before the camera-
ready submission.
---
## 1. Appendix C ? JSD Numerical Example (DISCREPANCY)
**Discovered:** April 28, 2026
**Severity:** Medium (factual error in the paper, fixable by changing numbers)
### What the paper says
The paper claims: For p = 0.872, q = 0.238, we obtain 2*JSD = 1.84 bits.
### What is mathematically true
The Jensen-Shannon Divergence (in bits) for two Bernoulli distributions
Bernoulli(p) and Bernoulli(q) is bounded:
- Maximum pure JSD = 1 bit (achieved as |p - q| -> 1)
- Maximum 2*JSD = 2 bits
For the specific values p = 0.872, q = 0.238:
    pure JSD = 0.319 bits
    2*JSD    = 0.639 bits
This is the correct value the implementation produces, and it is
nowhere near 1.84 bits.
### Why this matters
A reviewer running our code on the example will see 0.639 bits and
conclude the implementation is broken -- when in fact it is the paper
that is wrong.
### What value of (p, q) would produce 1.84 bits?
    p = 0.99, q = 0.01  ->  2*JSD = 1.8384 bits  (matches!)
    p = 0.98, q = 0.02  ->  2*JSD = 1.7171 bits
    p = 0.95, q = 0.05  ->  2*JSD = 1.4272 bits
So the paper almost certainly intended p = 0.99, q = 0.01 and
0.872 / 0.238 are typos.
### Resolution options for the paper
1. Change the example values in Appendix C to (0.99, 0.01) so that
   1.84 bits is correct.
2. Change the claimed result in Appendix C to ~ 0.64 bits and use
   the original (0.872, 0.238) values.
3. Re-derive what the original computation actually was -- maybe the
   paper meant a different divergence (e.g. KL, total variation, or
   JSD in nats instead of bits).
Recommendation: Option 1 is the cleanest. The qualitative claim of
the paper (CAFF achieves a much higher 2*JSD than baselines) is
preserved -- it just requires more extreme p, q to hit 1.84 bits.
### Code-side action taken
tests/test_evaluator.py::test_jsd_paper_appendix_c_calculation was
updated to use (p, q) = (0.99, 0.01).
---
## 2. (placeholder for future discrepancies)
When you find another discrepancy between paper and code, append it here
with the same structure: severity, what paper says, what is true,
recommended resolution.
