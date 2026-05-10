# Rollout Plan

A draft plan for sharing `https://wdanfort.github.io/thinkingtype/` with the right communities and people. Not a checklist to execute mechanically. Pick the venues that feel right and skip the rest.

## Strategy in one paragraph

This is independent research with one practical finding (VLMs flag fewer borderline cases for human review than LLMs alone) and one curiosity finding (font choice matters). The page is short, has interactive elements, real citations, and honest limitations. The right move is to share it like a research post, not a product launch. Lead with the practical finding. Let venues that care about the curiosity angle find their way to it.

The page already calls out that the tested models are not the latest, so be ready for "have you re-run on the newest models?" as the #1 question, and for "show full prompts and judge specs" as #2.

## Pre-launch (do these once, not per-venue)

- [ ] Verify all five Related Work citations one more time before posting. The page has live links.
- [ ] Stage a one-image social card. Decision-flip chart (`figures/comparison_decision_flip.png`) is already set as the OG image, but consider designing a card with the headline and the 73/100/~0 numbers, since people share screenshots.
- [ ] Decide whether you want to be tagged in posts. If yes, make sure your handle on each platform is in the page footer.
- [ ] Set up Google Alerts for "thinkingtype" and your name so you catch picks-up you'd otherwise miss.
- [ ] Have a v0.1 follow-up draft ready: "things people asked about, things we'd change." Even a stub. The follow-up is what carries the conversation.
- [ ] Pin a tracking spreadsheet (or just a doc): venue, date, response, follow-up needed.

## Where to post

Ordered roughly by signal-to-effort. Pick maybe 4 to 6 of these for the first wave.

### Tier 1: do these

**1. Hacker News (a regular submission, not Show HN.)** Show HN is for things people can run; this is a write-up. Submit with the actual page title or a clear variant. HN rewards plain descriptive framing.
- Best windows: Tuesday to Thursday, 8 to 11 AM Eastern. Counter-intuitive option: Sunday late evening Pacific, lower competition.
- First comment from you: post your own context comment within the first 30 minutes. Say "author here, happy to answer," briefly flag that the tested models are not the current frontier, and link the limitations section.
- If it doesn't catch in the first 2 hours, do not repost. Move on. Try again in a few weeks with the v1 (newer models) re-run.
- [HN guidelines](https://news.ycombinator.com/showhn.html). [Best time to post analysis](https://blog.alcazarsec.com/tech/posts/best-time-to-post-on-hacker-news).

**2. X / Twitter thread.** This is still where AI research gets discussed in real time. Lead post: the headline, the decision-flip number, and one image (the OG card). Subsequent posts: one finding each, ending with a link and an invitation for harder questions.
- Tag people who might amplify (see "Who to send" below). Don't tag more than three people in a single post.
- A second-day quote-tweet of your own thread with one updated insight ("here's what people asked about; here's what I'd change") doubles your reach without feeling spammy.

**3. BlueSky.** Academic and AI-research migration to BlueSky has been real over the last year. [Scientific content performs about 3x better there than on X for the same author](https://opentools.ai/news/bluesky-takes-flight-why-scientists-prefer-it-over-x-for-sharing-research). Cross-post your thread there. Get added to an [AI researchers starter pack](https://blueskystarterpack.com/ai-researchers).
- A custom feed for VLM eval / multimodal robustness work might be worth subscribing to before posting, so you know who is active.

**4. LinkedIn.** Lower-effort copy of the X thread, written for a product / policy audience instead of an ML audience. Lead with the triage-decision finding. The OpenDyslexic / accessibility note plays especially well here.

**5. Personal network: 5 to 10 DMs.** Before any public post, send a draft to 5 to 10 people whose feedback you actually want. Ask: "is this overclaiming anything? what would you change? is the headline boring?" This catches the worst stuff and gives you a few people who'll amplify when you go live.

### Tier 2: targeted submissions

**6. r/MachineLearning** with the `[R]` flair. The community is strict about self-promotion outside the weekly self-promotion thread, but a [R]-flagged research post with methodology, limitations, and citations passes. Lead with the data table and the decision-asymmetry finding, not the headline. The mods reward technical substance and remove anything that reads like marketing. [r/MachineLearning marketing-side overview](https://www.reddit-radar-marketing.com/guides/r/machinelearning).

**7. r/typography and r/Design.** Different angle: framed as "AI is now reading our typography, and our typography is changing how it reads." Skip the methodology details. The OpenDyslexic finding is the hook here. Link the page; don't paste the whole thing.

**8. r/accessibility.** Same angle as r/typography but lead with the OpenDyslexic fairness flag explicitly, and be careful: this community is allergic to "ableism by accident" framings that are slightly off. Phrase it as a question or a flag, not a finding.

**9. Lobsters.** Smaller and more technical than HN. Works if you have a Lobsters invite. Tag with `ai` and `practices`.

### Tier 3: newsletters and mailing lists (longer lead time)

Send a short pitch (3 sentences and the link) to:

- **Import AI** ([Jack Clark](https://importai.substack.com/), Anthropic). He covers VLM and multimodal robustness work regularly. Goes out weekly.
- **The Batch** ([Andrew Ng / DeepLearning.AI](https://www.deeplearning.ai/the-batch/)). They have a research desk that picks up independent work occasionally.
- **AlphaSignal**. Has a ranking algorithm; a clean arxiv-style write-up surfaces better than a blog post, but they pick up substantive blog posts too.
- **AI Snake Oil** ([Sayash Kapoor and Arvind Narayanan](https://www.aisnakeoil.com/)) and **AI as Normal Technology** ([same authors](https://www.normaltech.ai/)). The decision-deflection finding fits their thesis about practical AI reliability. Specifically pitch the practical-impact angle, not the typography angle.
- **One Useful Thing** ([Ethan Mollick](https://www.oneusefulthing.org/)). Lighter touch; more about implications than methodology. The hook is "your font choice is now an AI input."
- **Last Week in AI** podcast / newsletter. Independent research with a clear finding is the kind of thing they read out.

### Tier 4: longer plays

- **arxiv.** Write up a 4-to-6 page note version (methodology, dimension list, prompts, judge specs, full results). Post to cs.CL or cs.HC. This is the version that makes the work citable; it also gives the v1 re-run somewhere to land.
- **A blog post / Substack of your own.** The github.io page is a research artifact. A Substack post that is more personal ("here is why I ran this, here is what I'd do next") drives different traffic and gets you a subscriber list for the v1.
- **Conference workshops.** ICLR, NeurIPS, and ACL all have multimodal-robustness / evaluation workshops. The v1 (re-run on newer models, full document tests) is what you'd submit. Mark the deadlines now.

## Who to send to specifically

Don't cold-DM everyone on this list. Pick 4 or 5 whose feedback would actually shape v1. Send a short, specific note. Mention what you'd want from them ("would love your take on framing" or "does this read as overclaiming?"). Do not mass-email.

### Authors of the related work cited on the page

- **Melanie Sclar, Yejin Choi, Yulia Tsvetkov, Alane Suhr.** Prompt-format sensitivity authors. Sclar in particular is the natural reviewer for the analog claim. Find current handles on each researcher's webpage.
- **Tianrui Guan, Fuxiao Liu** (HallusionBench). They'd find the angle interesting and might know the related VLM-eval lit better than the page covers.
- **Yifan Li and the POPE team.**
- **Xiangyu Qi** (visual adversarial / jailbreaking). The non-adversarial-typography angle is a counterpoint they'd care about.

For each: their personal page or Google Scholar profile has the current contact / handle. Do not guess at handles.

### Writers and curators who could amplify

- **Ethan Mollick** ([Wharton, One Useful Thing](https://www.oneusefulthing.org/)). Practical-AI hub. The "your fonts are now an AI input" framing fits his audience.
- **Simon Willison** ([simonwillison.net](https://simonwillison.net/)). Engineering-practical AI; will appreciate the methodology rigor and the reproducible eval harness.
- **Sayash Kapoor** and **Arvind Narayanan** (Princeton, AI Snake Oil). The triage-deflection finding aligns directly with their critique of AI-as-decision-maker hype.
- **Jack Clark** (Anthropic, Import AI). Weekly newsletter; tends to surface independent work.
- **Karen Hao** (independent AI journalist, formerly MIT Tech Review). The OpenDyslexic angle is journalism-shaped.
- **Casey Newton** (Platformer). Less likely to pick this up but the practical-triage framing matches his beat.
- **Dan Hendrycks** and the **CAIS** group. Robustness / evals angle.

### Accessibility and typography people (for the OpenDyslexic finding specifically)

- **Abelardo Gonzalez**, creator of OpenDyslexic. Worth flagging directly; you don't want him to find out from a screenshot. Frame it carefully: you're flagging it as a possible fairness issue, you want more evidence before making strong claims, and you'd value his perspective.
- **Adrian Roselli, Marcy Sutton, Eric Bailey, Hidde de Vries.** Accessibility veterans with active audiences.
- **Léonie Watson** (TetraLogical). Web accessibility leader; reads research carefully.
- Find current handles via personal sites; the BlueSky migration since 2024 has shuffled where each is most active.

### Trust & Safety and product people

The decision-deflection finding has direct T&S implications. Categories rather than names because the relevant people are usually inside companies and respond to topic-fit rather than cold outreach:

- T&S / content-moderation eval folks at Meta, Discord, Reddit, Roblox.
- Anthropic and OpenAI safety / evals teams (they have public-facing researchers who post on this kind of finding).
- Healthcare-AI evaluation folks (Stanford HAI, MIT Jameel, Hugging Face health group). The medical-triage example you use lands specifically with this audience.
- Fairness / fair-ML researchers at FAccT-adjacent labs.

## Timing

Suggested two-week cadence, but adapt:

- **Day -3 to -1.** Send the page to your 5 to 10 personal-network reviewers. Incorporate the worst-case feedback. Fix anything that overclaims.
- **Day 0 (Tuesday morning ET).** Post to HN. Post X thread. Cross-post to BlueSky. LinkedIn post. Send 3 to 4 targeted DMs to the people you most want to hear from (not a mass send).
- **Day 1 to 2.** Respond to comments on HN and X. Update the page if something specific gets flagged repeatedly (you can ship small edits same-day to GitHub Pages).
- **Day 3.** Post to r/MachineLearning with [R] flair (give HN a couple of days first so you don't compete with yourself).
- **Day 4 to 5.** Send pitches to newsletters (Import AI, The Batch, AI Snake Oil). Submit to Lobsters if you have an invite.
- **Day 7.** Post to r/typography and r/accessibility with the design / fairness framing. By this point you've heard the main objections and can pre-empt them.
- **Day 10 to 14.** Write a follow-up post ("what people asked about, what I'd change, what's next") and link from the original. Pin it on X / BlueSky.
- **Day 21 onward.** Start the v1 re-run on current frontier models. Plan the arxiv version. Note when conference workshop deadlines hit.

## Responding to feedback

You will get some of these. Have answers ready.

- **"Have you re-run on the latest models?"** No, and that is called out at the top of the findings and in Limitations. v1 is planned. (This will be the #1 question; consider having a single tweet-length response prepped.)
- **"Show me the prompts and the judge spec."** Point at the GitHub repo and the configs directory. If the repo doesn't make this easy yet, fix that before launching.
- **"How is this different from Sclar et al. / POPE / HallusionBench?"** The Related Work section answers this; you can quote it directly.
- **"Why temperature 0?"** Reproducibility. The t=0.3 sanity run is mentioned in Limitations.
- **"Is the OpenDyslexic finding a real fairness issue?"** Be careful here. The page says "possible fairness issue, want more evidence, worth flagging" and that is the right line. Don't escalate it. If pressed, point at the small sample.
- **"Did Claude / GPT-4 write this?"** "Code and analysis were built with Claude Code. Framing and writing are mine, with editor passes."

## What success looks like

- A handful of substantive critiques from people whose work you cited or who work in the area. (More valuable than upvotes.)
- One or two follow-up conversations that change what v1 looks like.
- Inbound from at least one T&S / product team who saw the decision-deflection finding and wants to talk about their pipeline.
- Newsletter mention or two.

Front-page HN or viral X thread would be nice but is mostly noise. The conversation you want is downstream of the people who actually do related work seeing it.

## What failure looks like (and is fine)

- Quiet launch. No big response. This is the most likely outcome. The page is still there; it gets cited later when someone runs into the question.
- The page gets picked up but only the catchy framing ("Comic Sans changes AI judgments!") propagates. If this happens, post a clarifying follow-up immediately. The current framing of the page is built to resist this, but framings drift in retelling.
