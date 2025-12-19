# Summarization Prompts Reference

This document details the system prompt and default user prompt used for article summarization. These prompts are designed for AI/technology news summarization targeting technical professionals.

---

## System Prompt

The system prompt establishes the AI's persona and enforces consistent output quality across all summarization requests.

### Prompt Text

```
You are an expert technical journalist specializing in AI and technology news. Your summaries are written for AI developers, researchers, and technology professionals who value precision, technical depth, and direct communication.

Core principles:
- Present information directly and factually in active voice
- Avoid meta-language like 'This article explains...', 'This is important because...', or 'The author discusses...'
- Include technical details, specifications, and industry implications
- Use clear, straightforward language without hype, exaggeration, or marketing speak
- Focus on what matters to technical practitioners: capabilities, limitations, pricing, availability

Style conventions:
- Use active voice and non-compound verbs (e.g., 'banned' not 'has banned')
- Spell out numbers and 'percent' (e.g., '8 billion', not '8B' or '%')
- Use smart quotes, not straight quotes
- Use 'U.S.' and 'U.K.' with periods; use 'AI' without periods
- Avoid the words 'content' and 'creator' when possible
```

### Strategy Explanation

| Element | Purpose |
|---------|---------|
| **"Expert technical journalist"** | Establishes authority and domain expertise. The AI adopts a professional editorial voice rather than a generic assistant tone. |
| **Target audience specification** | Explicitly naming "AI developers, researchers, and technology professionals" ensures the output assumes technical literacy and avoids over-explaining basic concepts. |
| **"Precision, technical depth, and direct communication"** | Sets quality expectations—summaries should be information-dense, not fluffy. |
| **Anti-meta-language rule** | Prevents filler phrases that waste reader time. "This article explains X" adds nothing; just state X directly. |
| **Technical details requirement** | Ensures summaries include concrete data (benchmarks, prices, dates) rather than vague claims. |
| **Anti-hype language** | Filters out marketing language common in tech press releases. Keeps summaries objective. |
| **Practitioner focus** | Prioritizes actionable information (what can I use? what does it cost? when?) over abstract analysis. |
| **Style conventions** | Enforces consistency across outputs. Active voice is more direct; spelled-out numbers are more readable; avoiding "content/creator" sidesteps overused jargon. |

---

## Default User Prompt

The user prompt provides specific structural requirements and content priorities for each summarization request.

### Prompt Text

```
Summarize the article below following these guidelines:

Structure:
1. First line: Create a headline in sentence case that:
   - Captures the core news or development
   - Uses strong, specific verbs
   - Avoids repeating exact phrases from the summary
2. Then a blank line
3. Then a focused summary of three to five sentences:
   - First sentence: State the core announcement, finding, or development
   - Following sentences: Include 2-3 of these elements as relevant:
     • Technical specifications (model sizes, performance metrics, capabilities)
     • Pricing, availability, and access details
     • Key limitations or constraints
     • Industry implications or competitive context
     • Concrete use cases or applications
   - Prioritize information that answers: What changed? What can it do? What does it cost? When is it available?
4. Then a blank line
5. Then add 'Source: [publication name]' followed by the URL

Style guidelines:
- Use active voice (e.g., 'Company released product' not 'Product was released by company')
- Use non-compound verbs (e.g., 'banned' instead of 'has banned')
- Avoid self-explanatory phrases like 'This article explains...', 'This is important because...', or 'The author discusses...'
- Present information directly without meta-commentary
- Avoid the words 'content' and 'creator'
- Spell out numbers (e.g., '8 billion' not '8B', '100 million' not '100M')
- Spell out 'percent' instead of using the '%' symbol
- Use 'U.S.' and 'U.K.' with periods; use 'AI' without periods
- Use smart quotes, not straight quotes

Additional guidelines:
- For product launches: Always include pricing and availability if mentioned
- For research papers: Include key metrics, dataset sizes, or performance improvements
- For company news: Focus on concrete actions, not just announcements or intentions
- Omit background information readers likely already know (e.g., 'OpenAI is an AI company')

Article:
{article_text}

URL: {url}
Publication: {source_name}
```

### Strategy Explanation

#### Structure Design

| Element | Purpose |
|---------|---------|
| **Headline first** | Gives readers immediate context. Sentence case is more readable than ALL CAPS or Title Case for news. |
| **"Strong, specific verbs"** | Prevents weak headlines like "Company makes announcement about product." Pushes toward "Company launches product" or "Company cuts prices." |
| **"Avoids repeating exact phrases"** | Prevents redundancy between headline and body. The headline should complement, not duplicate. |
| **3-5 sentence constraint** | Forces prioritization. Long summaries defeat the purpose; this constraint ensures density. |
| **First sentence = core development** | Inverted pyramid style—lead with the news. Readers who only read one sentence get the essential information. |
| **Element menu (specs, pricing, limitations, etc.)** | Provides a prioritized checklist of what technical readers care about. Not all apply to every article, so "2-3 as relevant" gives flexibility. |
| **"What changed? What can it do? What does it cost? When?"** | These four questions capture 90% of what practitioners need to know about any tech announcement. |
| **Source attribution** | Maintains journalistic standards and allows readers to verify or read the full article. |

#### Style Guidelines Rationale

| Guideline | Why It Matters |
|-----------|----------------|
| **Active voice** | More direct and engaging. "Google released Gemini" is stronger than "Gemini was released by Google." |
| **Non-compound verbs** | "Banned" is tighter than "has banned." Reduces word count without losing meaning. |
| **No meta-commentary** | "This article explains how..." wastes words. Just explain how. |
| **Spelled-out numbers** | "8 billion" reads more naturally than "8B" in prose. Prevents ambiguity (is "8B" bytes or billion?). |
| **Spelled-out "percent"** | Matches journalistic style guides. More readable in flowing text. |
| **Smart quotes** | Professional typography. Straight quotes look like code or unformatted text. |
| **Abbreviation rules** | Consistency. "U.S." with periods follows AP style; "AI" without periods is standard usage. |
| **Avoid "content/creator"** | Overused buzzwords that often obscure meaning. Forces more specific language. |

#### Content-Type-Specific Guidelines

| Content Type | Guidance | Rationale |
|--------------|----------|-----------|
| **Product launches** | Include pricing and availability | These are the most common questions readers have. Summaries without them feel incomplete. |
| **Research papers** | Include metrics, datasets, improvements | Technical readers want to assess significance. "Improves accuracy" means nothing without numbers. |
| **Company news** | Focus on actions, not intentions | "Company plans to..." is weaker than "Company will..." which is weaker than "Company did..." Prioritize concrete over speculative. |
| **Background omission** | Skip obvious context | Don't waste words explaining that OpenAI makes AI or that Google is a tech company. Assume reader knowledge. |

---

## Usage Notes

### Combining System and User Prompts

When calling the LLM API:
1. Pass the **system prompt** as the `system` parameter (or system message role)
2. Pass the **user prompt** (with article text substituted) as the `user` message

### Adapting for Other Domains

To adapt these prompts for non-tech summarization:

1. **System prompt**: Change the expertise domain ("expert technical journalist" → "expert [domain] analyst") and update the target audience
2. **User prompt element menu**: Replace technical elements (model sizes, benchmarks) with domain-relevant elements (financial metrics, clinical outcomes, etc.)
3. **Style guidelines**: Most are universal; domain-specific abbreviation rules may need updates

### Output Format

Expected output structure:
```
[Headline in sentence case]

[3-5 sentence summary paragraph]

Source: [Publication Name] [URL]
```

---

## Alternative Style: Axios-Style Bullet Points (Optional)

> **TODO**: Implement if your use case benefits from scannable, structured summaries with hierarchical information.

### Prompt Text

```
Create an Axios-style summary of the article following these guidelines:

Structure:
1. First line: Create a bold, catchy headline in sentence case
2. Then a blank line
3. Then a brief 1-2 sentence overview of what the article is about
4. Then a blank line
5. Then a section called 'The big picture:' with 1-2 sentences of context
6. Then a section called 'Key points:' with 4-6 bullet points that:
   - Start each bullet with '•' followed by a bold statement or statistic
   - Follow each bold statement with 1-2 explanatory sentences
   - Include surprising details, not just the obvious points
   - Mix essential facts with interesting implications
7. If applicable, a section called 'What's next:' with 1-2 bullets about future implications
8. Then a blank line
9. Then add 'Source: [publication name]' followed by the URL

{common_style_guidelines}

- Make bullet points conversational but insightful
- Ensure some bullets contain surprising or counterintuitive information
```

### Strategy Explanation

| Element | Purpose |
|---------|---------|
| **Axios format inspiration** | Axios pioneered a scannable news format optimized for busy readers. The structure lets readers extract value at multiple depths. |
| **"Bold statement + explanation" pattern** | Frontloads the key fact in each bullet. Readers scanning only bold text still get the essentials. |
| **"The big picture" section** | Provides context without burying the news. Answers "why should I care?" separately from "what happened?" |
| **"Key points" with 4-6 bullets** | More granular than paragraph summaries. Each bullet is self-contained and skippable. |
| **"What's next" section** | Forward-looking analysis separated from facts. Optional because not all stories have clear implications. |
| **"Surprising details" requirement** | Prevents bullet points from being obvious restatements. Pushes for non-obvious insights. |

### When to Use

- Newsletter formats where readers scan quickly
- Slack/Teams digests where bullet structure renders well
- Situations where readers have varying interest levels (some want headlines only, others want depth)

---

## Alternative Style: Newswire/AP Style (Optional)

> **TODO**: Implement if your use case requires formal, objective reporting style suitable for syndication or archival.

### Prompt Text

```
Create a traditional newswire-style article summary following these guidelines:

Structure:
1. First line: Create a concise, factual headline in title case (AP style)
2. Then a blank line
3. Then a dateline in all caps (e.g., 'SAN FRANCISCO —')
4. Then a first paragraph (lead) that covers the 5 Ws (who, what, when, where, why) in a single sentence
5. Then 3-5 additional paragraphs that:
   - Follow the inverted pyramid structure (most important to least important)
   - Include at least one direct quote if present in the source material
   - Provide context and background in later paragraphs
   - Maintain a formal, objective tone throughout
6. Then a blank line
7. Then add 'Source: [publication name]' followed by the URL

{common_style_guidelines}

- Use short paragraphs (1-2 sentences each)
- Focus on facts over analysis
- Avoid subjective language or speculation
```

### Strategy Explanation

| Element | Purpose |
|---------|---------|
| **AP style headline** | Title case is the wire service standard. Signals formal journalism rather than blog-style writing. |
| **Dateline** | Traditional journalism convention indicating story origin. Adds authenticity and context. |
| **5 Ws in the lead** | Classic journalism structure ensuring the first paragraph is self-sufficient. Editors can cut from the bottom. |
| **Inverted pyramid** | Most important information first, least important last. Allows flexible truncation without losing key facts. |
| **Direct quote requirement** | Adds credibility and human voice. Preserves original source attribution. |
| **Short paragraphs** | Wire service style uses 1-2 sentence paragraphs for readability and easy reformatting by publishers. |
| **"Facts over analysis"** | Newswire is meant to be objective and reusable. Analysis belongs in opinion pieces, not news summaries. |

### When to Use

- Formal documentation or archival purposes
- Contexts requiring objective, unbiased tone
- Syndication where multiple outlets may republish
- Legal or compliance contexts where subjective language is problematic

---

## Common Style Guidelines Reference

Both alternative styles share these guidelines with the default style:

```
Style guidelines:
- Use active voice (e.g., 'Company released product' not 'Product was released by company')
- Use non-compound verbs (e.g., 'banned' instead of 'has banned')
- Avoid self-explanatory phrases like 'This article explains...', 'This is important because...', or 'The author discusses...'
- Present information directly without meta-commentary
- Avoid the words 'content' and 'creator'
- Spell out numbers (e.g., '8 billion' not '8B', '100 million' not '100M')
- Spell out 'percent' instead of using the '%' symbol
- Use 'U.S.' and 'U.K.' with periods; use 'AI' without periods
- Use smart quotes, not straight quotes
```

---

## Source

These prompts are implemented in `summarization/text_processing.py`:
- `get_system_prompt()` — returns the system prompt
- `create_summary_prompt()` — generates the user prompt with article content
