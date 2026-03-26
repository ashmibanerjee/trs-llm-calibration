# LLMs as Evaluators: A Multi-dimensional Assessment of Recommendation Lists for Sustainable City Trips

### Authors
* [Ashmi Banerjee](https://ashmibanerjee.com), Technical University of Munich (TUM), Germany 
* [Adithi Satish](https://www.linkedin.com/in/adithi-satish), Technical University of Munich (TUM), Germany
* [Wolfgang Wörndl](https://www.ce.cit.tum.de/cm/research-group/wolfgang-woerndl), Technical University of Munich (TUM), Germany
* [Yashar Deldjoo](https://yasdel.github.io), Polytechnic University of Bari, Italy

### Abstract 

Evaluating the nuanced qualities of conversational travel recommendations is challenging when human annotations are costly and traditional metrics overlook stakeholder-centric goals. 
We study LLM‑as‑a‑Judge for sustainable city‑trip lists across four dimensions—relevance, diversity, sustainability, and popularity balance—and propose a three‑phase calibration framework: (1) baseline judging with multiple LLMs, (2) expert evaluation to surface systematic misalignment, and (3) dimension‑specific calibration via rules and few‑shot examples. Across two recommendation settings, we find substantial model‑specific biases and high variance at the dimension level, even when judges often agree on the overall “best list.” Calibration improves some judge–judge consistencies but yields only modest gains in human alignment, underscoring the need for transparent, bias-aware LLM evaluation pipelines.

### Calibration Checklist

```
A. Relevance – Avoid default ties; prefer lists that reflect explicit constraints with city-level evidence. Focus on actual fit, not keyword overlap.

B. Sustainability – Reward concrete sustainability indicators: car-free cores, transit quality, smaller/less-commercialized cities, air-quality notes, seasonality to reduce overtourism; penalize vague “green” claims. Do not over-penalize popular cities in shoulder season: If the list argues seasonal de-concentration convincingly (e.g., October vs peak), do not auto-downgrade.

C. Popularity – Prefer lists mixing major and lesser-known spots. Reject all-popular lists unless explicitly requested in the query. Down-weight crowded ones unless mitigated. E.g., Prague, Porto, Budapest = Popular choices and NOT hidden gems.

D. Diversity – Prioritize broad regional coverage (unless query limits scope). Include thematic variety.

E. Tie-breaks – Compare constraint coverage → specificity → factual accuracy → fewer contextual violations.

F. Validation – Verify each entry’s type (city/country/landmark). Penalize factual errors (e.g., “Malta” and "Santorini" as cities).

G. Context & Seasonality – Penalize closed, unsafe, or seasonally unsuitable picks. E.g., Lviv during conflict, Keukenhof in October, Tallinn in November (limited daylight) should score lower on contextual fit, whereas a southern destination with favorable winter conditions should score higher).

H. Confidence – Only credit sustainability or cultural claims backed by verifiable evidence; mark “unsure” if unclear.


```
Exact calibration prompts can be found here: [calibration prompts](prompts/calibration/)
