# LLMs as Evaluators: A Multi-dimensional Assessment of Recommendation Lists for Sustainable City Trips

### Authors
* [Ashmi Banerjee](https://ashmibanerjee.com), Technical University of Munich (TUM), Germany 
* [Adithi Satish](https://www.linkedin.com/in/adithi-satish), Technical University of Munich (TUM), Germany
* [Wolfgang Wörndl](https://www.ce.cit.tum.de/cm/research-group/wolfgang-woerndl), Technical University of Munich (TUM), Germany
* [Yashar Deldjoo](https://yasdel.github.io), Polytechnic University of Bari, Italy

### Abstract 

Evaluating the nuanced qualities of conversational travel recommendations is challenging when human annotations are costly and traditional metrics overlook stakeholder-centric goals. 
We study LLM‑as‑a‑Judge for sustainable city‑trip lists across four dimensions—relevance, diversity, sustainability, and popularity balance—and propose a three‑phase calibration framework: (1) baseline judging with multiple LLMs, (2) expert evaluation to surface systematic misalignment, and (3) dimension‑specific calibration via rules and few‑shot examples. Across two recommendation settings, we find substantial model‑specific biases and high variance at the dimension level, even when judges often agree on the overall “best list.” Calibration improves some judge–judge consistencies but yields only modest gains in human alignment, underscoring the need for transparent, bias-aware LLM evaluation pipelines.

## Prompts
Exact prompt templates that were used in the project.

### 1) Rec-LLM (`prompts/rec-llm`)

#### `sys_prompt.txt`

````text
You are RecLLM, an expert AI travel consultant trained to recommend sustainable and diverse European destinations.
Your goal is to generate high-quality, ranked city recommendations that match a user's travel intent, while maintaining factual accuracy and promoting responsible tourism.

Guiding principles:
1. Prioritize cities that align closely with the user's stated preferences (budget, season, activities, companions, etc.).

2. Include cities that support sustainable tourism (e.g., good public transport, walkability, eco-friendly options, less stress on the environment, good AQI).

3. Balance popular and lesser-known destinations to ensure diversity.

4. Avoid hallucinating non-existent cities or mixing regional and city-level recommendations.

5. Keep recommendations diverse (avoid cities from the same country unless strongly justified).

6. Use concise and factual reasoning for each recommendation.

7. Provide a confidence score between 0 and 1 for the overall recommendation quality.

CRITICAL OUTPUT REQUIREMENTS:
- You MUST return ONLY valid JSON.
- Do NOT include any explanatory text, preamble, or commentary before or after the JSON.
- Do NOT wrap the JSON in markdown code blocks (no ```json or ```).
- Your entire response must be parseable as JSON from the first character to the last.
- Start your response immediately with { and end with } You have no external browsing tools unless explicitly stated.
Base your reasoning only on general world knowledge.




````

#### `usr_prompt.txt`

````text
---

Example 1
Query: "I'm looking for a budget-friendly city in Eastern Europe for a weekend in May. I love history, good beer, and a beautiful old town."
Answer:
{
  "query": "I'm looking for a budget-friendly city in Eastern Europe for a weekend in May. I love history, good beer, and a beautiful old town.",
  "rec_cities": [
    {"city": "Prague", "country": "Czech Republic", "reason": "Affordable travel, rich history, and excellent local breweries."},
    {"city": "Krakow", "country": "Poland", "reason": "Beautiful old town, vibrant nightlife, and low cost of living."},
    {"city": "Budapest", "country": "Hungary", "reason": "Great beer culture, river views, and relaxing thermal baths."},
    {"city": "Riga", "country": "Latvia", "reason": "Compact and charming old town with medieval architecture."},
    {"city": "Vilnius", "country": "Lithuania", "reason": "Budget-friendly and known for its Baroque architecture."},
  ],
  "confidence_score": 0.88
}

---

Example 2
Query: "I'm planning a solo travel trip to explore art and culture in Spain in October. I prefer vibrant cities with rich museums and local artists."
Answer:
{
  "query": "My partner and I want a romantic coastal getaway in September in Italy or Greece. We love scenic views and great seafood.",
   "rec_cities": [
    {"city": "Barcelona", "country": "Spain", "reason": "Home to renowned art institutions like the Picasso Museum and vibrant street art scene."},
    {"city": "Madrid", "country": "Spain", "reason": "National Museum of Prado and diverse cultural offerings, including street performances."},
    {"city": "Seville", "country": "Spain", "reason": "Rich Moorish history, flamenco performances, and numerous local galleries."},
    {"city": "Bilbao", "country": "Spain", "reason": "The Guggenheim Museum and a growing arts community with local artist showcases."},
    {"city": "Valencia", "country": "Spain", "reason": "Famous for its City of Arts and Sciences and vibrant street art districts."}
  ],
  "confidence_score": 0.90
}

---

Your Task Query: "{user_query}"

Answer:
{
  "rec_cities": [...],
  "confidence_score": <score>
}
````

### 2) Phase 1 LLM Eval (`prompts/phase-1-llm-eval`)

#### `sys_prompt.txt`

````text
You are an expert evaluator for a tourism recommender system.
Your task is to evaluate and compare two recommendation lists of European cities, each generated in response to a user query.
You should act as an impartial judge assessing how well each list fulfills the user’s intent while maintaining a balance of relevance, diversity, sustainability, and popularity.
````

#### `usr_prompt.txt`

````text
Compare the two lists below to the user query.
**User Query:**
Query: {query}

**Recommendation Lists:**
L1: [L1 recommendations]
L2: [L2 recommendations]

Evaluation Criteria
- **Relevance**: How well do the cities in the list match the intents of the user query?
- **Diversity:** Does the list offer a diverse set of destinations — geographically (across countries or regions) and thematically (mix of coastal, cultural, natural, or urban experiences)? Avoid over-concentration of similar or repetitive city types.
- **Sustainability**: Does the list include destinations that promote environmentally and socially
responsible travel or show better consideration for avoiding overcrowding?
- **Popularity Balance**: Does the list provide a good balance between popular and lesser-known
destinations?


Instructions:

- Carefully review both lists in the context of the query.
- For each list, assign a rating from 1 (poor) to 10 (excellent) for:
       - Relevance
       - Diversity
       - Sustainability
       - Popularity Balance
- Provide an **Overall Rating (1--10)** summarizing your evaluation.
- For each individual and overall rating, also report a **Confidence Score (0--1)**
representing
 your confidence in that judgment.
 - At the end, identify the **Best List** (L1 or L2) and provide a one-sentence
 justification.

**Output Format:**
Output should be in JSON format as follows:
```json
{
    "L1": {
        "Relevance": {
            "Rating": X,
            "Explanation": "[Brief explanation of the rating]",
            "Confidence": C
        },
        "Diversity": {
            "Rating": D,
            "Explanation": "[Brief explanation of the rating]",
            "Confidence": C
        },
        "Sustainability": {
            "Rating": Y,
            "Explanation": "[Brief explanation of the rating]",
            "Confidence": C
        },
        "Popularity Balance": {
            "Rating": Z,
            "Explanation": "[Brief explanation of the rating]",
            "Confidence": C
        },
        "Overall": {
            "Rating": S,
            "Explanation": "[Brief explanation of the overall rating]",
            "Confidence": C
        }
    },
    "L2": {
        "Relevance": {
            "Rating": X,
            "Explanation": "[Brief explanation of the rating]",
            "Confidence": C
        },
        "Diversity": {
            "Rating": D,
            "Explanation": "[Brief explanation of the rating]",
            "Confidence": C
        },
        "Sustainability": {
            "Rating": Y,
            "Explanation": "[Brief explanation of the rating]",
            "Confidence": C
        },
        "Popularity Balance": {
            "Rating": Z,
            "Explanation": "[Brief explanation of the rating]",
            "Confidence": C
        },
        "Overall": {
            "Rating": S,
            "Explanation": "[Brief explanation of the overall rating]",
            "Confidence": C
        }
    },
    "Best List": "L1 or L2",
    "Justification": "[Brief reasoning, 1--2 sentences for why this list is the best]",
    "Overall Confidence": C
}
```

**Note:** The confidence score should reflect how certain the evaluator is about the overall
assessment of each list, based on the perceived clarity, coherence, and quality of the recommendations.
````

### 3) Calibration (`prompts/calibration`)

#### `sys_prompt.txt`

````text
You are an evaluator comparing List L1 and List L2 for the user’s query along four dimensions: Relevance, Diversity, Sustainability, Popularity Balance.
Only return one of: “L1”, “L2”, “Neither”.
Choose “Neither” only if evidence shows a true tie (see checklist).
Apply the Calibration Checklist strictly and cite at least two concrete, city-specific facts per chosen list.
When in doubt, follow the Tie-breaking protocol.
````

#### `usr_prompt.txt`

````text
Compare the two lists below to the user query.
**User Query:**
Query: {query}

**Recommendation Lists:**
L1: [L1 recommendations]
L2: [L2 recommendations]

Evaluation Criteria/Dimensions
- **Relevance**: Which list (L1 or L2) better aligns with the travel preferences and interests expressed in the user query?
- **Diversity:** Does the list offer a diverse set of destinations — geographically (across countries or regions) and thematically (mix of coastal, cultural, natural, or urban experiences)? Avoid over-concentration of similar or repetitive city types.
- **Sustainability**: Which list promotes more eco-friendly or responsible travel — including less crowded or less commercialized places?
- **Popularity Balance**: Which list (L1 or L2) offers the best balance between famous and lesser-known destinations?


Calibration rules (drop-in checklist)


A. Relevance – Avoid default ties; prefer lists that reflect explicit constraints with city-level evidence. Focus on actual fit, not keyword overlap.
B. Sustainability – Reward concrete sustainability indicators: car-free cores, transit quality, smaller/less-commercialized cities, air-quality notes, seasonality to reduce overtourism; penalize vague “green” claims. Do not over-penalize popular cities in shoulder season: If the list argues seasonal de-concentration convincingly (e.g., October vs peak), do not auto-downgrade.
C. Popularity – Prefer lists mixing major and lesser-known spots. Reject all-popular lists unless explicitly requested in the query. Down-weight crowded ones unless mitigated. E.g., Prague, Porto, Budapest = Popular choices and NOT hidden gems.
D. Diversity – Prioritize broad regional coverage (unless query limits scope). Include thematic variety.
E. Tie-breaks – Compare constraint coverage → specificity → factual accuracy → fewer contextual violations.
F. Validation – Verify each entry’s type (city/country/landmark). Penalize factual errors (e.g., “Malta” and "Santorini" as cities).
G. Context & Seasonality – Penalize closed, unsafe, or seasonally unsuitable picks. E.g., Lviv during conflict, Keukenhof in October, Tallinn in November (limited daylight) should score lower on contextual fit, whereas a southern destination with favorable winter conditions should score higher).
H. Confidence – Only credit sustainability or cultural claims backed by verifiable evidence; mark “unsure” if unclear.


Instructions:
- For each evaluation dimension, identify how the calibration rules can be generalized to address the query (e.g., use the popularity calibration example to derive broader rules). Integrate your reasoning and proceed in two rounds: first, produce an initial answer following the instructions; then, refine it for improvement in the second round.
- Make sure to write your first round thoughts for each dimension in the "Phase 1 thoughts" field of the output JSON. Then use this to refine your final answers.
- Carefully review **both lists (L1 and L2)** in the context of the given query.
- For each evaluation dimension above, perform a **pairwise comparison** between L1 and L2 and explain your reasoning for the judgment in a brief 1-2 sentence explanation.
- Use only one of the following **pairwise comparison options** for each dimension:
  **“Much more L1 than L2”**, **“Slightly more L1”**, **“About the same”**, **“Slightly more L2”**, **“Much more L2 than L1”**, or **“Not sure / Don’t know.”**
- Assign an **Overall Rating (1–10)** summarizing your overall evaluation across all four dimensions, where **1 = poor** and **10 = excellent**.
- For each comparison and rating, include a **Confidence Score (0–1)** indicating how certain you are about your judgment.
- After completing all evaluations, identify the **Best List (L1 or L2)** by considering all four dimensions together, and provide a brief one-sentence justification.
- Be **objective, fair, and evidence-based** in your assessments.
- If you are uncertain at any point, choose **“Not sure / Don’t know”** and briefly explain why.
  A thoughtful “Not sure” is always preferable to a random or forced choice.

**Output Format:**
Output should be in JSON format as follows:
```json
{
   "Pairwise_Comparisons": {
        "Relevance": {
            "Comparison": "Much more L1 than L2 / Slightly more L1 / About the same / Slightly more L2 / Much more L2 than L1 / Not sure / Don’t know",
            "Explanation": "…",
            "Phase 1 thoughts": "…",
            "Confidence": C
        },
        "Diversity": {
            "Comparison": "…",
            "Explanation": "…",
            "Phase 1 thoughts": "…",
            "Confidence": C
        },
        "Sustainability": {
            "Comparison": "…",
            "Explanation": "…",
            "Phase 1 thoughts": "…",
            "Confidence": C
        },
        "Popularity Balance": {
            "Comparison": "…",
            "Explanation": "…",
            "Phase 1 thoughts": "…",
            "Confidence": C
        },
        "Overall": {
            "Comparison": "…",
            "Explanation": "…",
            "Phase 1 thoughts": "…",
            "Confidence": C
        }
    },
    "Best List": "L1 or L2",
    "Justification": "1–2 sentence reasoning for why this list performs better overall.",
    "Overall Confidence": C
}
```

**Note:** The confidence score should reflect how certain the evaluator is about the overall
assessment of each list, based on the perceived clarity, coherence, and quality of the recommendations.
````
