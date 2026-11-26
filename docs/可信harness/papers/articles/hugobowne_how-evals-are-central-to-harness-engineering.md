# How Evals Are Central To Harness Engineering

- 来源：https://hugobowne.substack.com/p/how-evals-are-central-to-harness
- 抓取日期：2026-10-25（curl 抓 HTML 后转纯文本，未读原始排版）
- 性质：博客/技术文章，不是论文；观点归作者，未经本仓库核实

---

How Evals Are Central To Harness Engineering
## Vanishing Gradients
SubscribeSign in
## How Evals Are Central To Harness Engineering
## You can’t build an agent harness without evals
Antaripa Saha, Hamel Husain, and Hugo Bowne-Anderson
Sep 01, 2026
40
5
9
Share
Today’s post comes from Antaripa Saha, a machine learning practitioner who worked directly with production evals at Quotient AI.
She makes the case that evals belong at the center of harness engineering. A harness gives an agent tools, memory, permissions, context, and an environment in which to act. Evals tell us whether changes to any of those things actually made the system better.
👉 This essay draws on Antaripa’s experience at Quotient AI and on Hamel Husain and Shreya Shankar’s AI Evals course, which helped her organize that practical experience into a clearer workflow from traces to failure modes, domain-specific criteria, and validated judges. Hamel and Shreya are teaching their next cohort of the year starting September 6. Vanishing Gradients readers get 25% off using this link. 👈
Over to Antaripa!
Vanishing Gradients is a reader-supported publication. To receive new posts and support my work, consider becoming a free or paid subscriber.
Subscribe
## Agent = Model + Harness.
That has become the cleanest way to describe modern agent systems.
The model brings capability. The harness turns that capability into work by giving the model tools, state, memory, execution environments, constraints, and feedback.
But I think the part that gets least attention and under-discussed is feedback.
Most harness discussions spend more time talking about the parts of the harness that let agents act: tools, filesystems, terminals, sandboxes, and so on. These make the agent more capable, but they do not tell us whether the capability is helping or just making it more complicated.
That is the central argument of this post: AI evals should be treated as a first-class harness component.
I’ll start by mapping what harnesses are, the different flavors of harnesses people mean today, then show why the current harness conversation needs a stronger measurement lens, and finally walk through how to build eval-centric harnesses for real AI systems.
TLDR: Harness engineering is how we build systems around models to make them useful. Evals are the feedback mechanism that lets a harness learn, self-correct, and improve.
By an eval, I mean a way of testing some aspect of model or agent behavior against what we expect.
A harness eval is narrower. It is meant to tell us whether the system around the model is doing its job: whether the harness is helping the agent complete the task correctly and reliably. A model can be capable while the overall agent still fails because of how the surrounding system is designed.
## What Exactly Is a Harness?
A harness is everything around the raw model that shapes how the agent behaves.
In practice, a harness includes:
-
Tools and execution environment (bash, filesystem, APIs, sandboxes)
-
Memory and state management (short-term context, long-term skills, conversation history)
-
Constraints and guardrails (rules about what the agent is allowed to do, when to ask for help, how to validate outputs)
-
Orchestration logic (planning loops, sub-agents, retry mechanisms)
-
Feedback and observability (logs, traces, validation steps, human approval gates)
Harnesses matter more now because models have become capable enough to do real work, but not reliable enough to do that work alone. Once agents moved beyond chat and started touching codebases, browsers, files, and business workflows, the hard problem shifted from “Can the model answer?” to “Can we design the environment around the model so it can act safely and reliably?”
That brings us to harness engineering.
Harness engineering is the practice of designing that surrounding system. It asks: what should the model be allowed to do, what context should it receive, what tools should it have, how should it recover from mistakes, and how should humans understand what happened?
💡 The phrase “harness engineering” was first coined by Vivek Trivedy in late 2025 (ref tweet and blog), and got popularized in early 2026 and widely discussed as teams realized this harness layer was where most of the real work (and leverage) lives.
## Different Flavors of Harnesses
Not every harness looks the same. People adapt the idea based on what they are building.
-
Test Harness: The older software meaning. A controlled setup for running and testing code with test data, mocks, stubs, drivers, scripts, and reporting. This is where the intuition comes from: isolate the system enough that you can check whether it behaves as expected.
-
General Agent Harness: This is the complete runtime environment that lets a model operate as an agent. It includes memory, tool orchestration, permissions, guardrails, execution environments, and observability. Most people picture this as the broad operating system for any agent.
-
Coding and workflow harnesses: This is where harnesses become specific to the job. A coding harness might include repo access, terminals, tests, logs, CI, and review flows; a support or research harness might include policies, retrieval, escalation paths, source constraints, and human handoff. The key idea is that a good harness is shaped by the workflow, not just by the model.
-
Evaluation harness: The measurement and feedback layer of the harness. It includes traces, eval datasets, scoring logic, failure taxonomies, custom metrics, LLM judges, human labels, and regression checks. This is where evals become central: they turn messy agent behavior into evidence the harness can use to improve.
The flavor you choose depends on what you are building, but strong evals are what make any of them actually reliable in production.
## Why Evals Are the Feedback Mechanism of the Harness
A harness without strong evals can make an agent do more things. It can give the model tools, memory, permissions, state, and an environment to act in.
But doing more is not the same as getting better.
The hard part of harness engineering is not only deciding what the agent can access. It is knowing whether those design choices actually improved the system.
Did the new tool help, or did it create another way to fail? Did the new memory rule reduce repeated mistakes, or did it introduce stale context? Did the stricter guardrail make the agent safer, or did it make the product less useful?
This is where evals become the feedback mechanism of the harness.
Vivek Trivedy (from LangChain) put it best in this blog Better Harness:
In classical machine learning, training data gives the model a learning signal; in agent development, evals give the harness a learning signal. They tell us whether a change to prompts, tools, memory, routing, permissions, or orchestration actually made the agent better.
Evals give us the feedback signal that lets us move from “the agent failed” to “we know why it failed and what to change next.” It helps locate where the system broke down. If the agent retrieved the wrong document, the harness may need better retrieval. If it chose the wrong tool, the harness may need clearer tool descriptions or better routing. If it produced a plausible but unsupported answer, the harness may need grounding checks.
The core pattern is simple:
💡 agent behavior → traces → error analysis → evals → harness changes → regression checks → better agent behavior
## What should a Good Harness Eval Measure?
A useful way to think about harness evals is in two layers. First, measure end-to-end task success: did the agent actually accomplish the job? Then, once you know which workflows fail, add step-level diagnostics: where did the failure happen, and which part of the harness should be inspected?
The categories below are not hard and fast rules, they are more like to make failures easier to locate:
-
Agent Outcome. This is the most honest signal of whether the agent accomplished its job. Did the code pass tests. Did the booking actually happen. Did the deploy succeed. Outcome signals keep the harness grounded in the real-world result.
-
Action Correctness. Outcome matters most, but for agents the outcome often depends on whether the right action actually happened. A support agent can say “your refund has been processed,” but if the refund tool was never called and the account balance never changed, the agent did not complete the task.
The goal is not to force one rigid path. It is to verify the actions or environment changes that the workflow actually depends on. If the task requires issuing a refund, booking a ticket, sending an email, or opening a PR, the harness should check that it happened instead of trusting the final message.
Hamel makes a version of this point in Revenge of the Data Scientist and even Anthropic makes a similar distinction in their agent evals guide separating transcript/trajectory from outcome and defines outcome as the final environment state.
-
Trajectory and Recovery. Most evals only check the end state. The agent either reached the goal or it didn’t. A good harness eval should tell us whether the agent noticed it was off track and corrected.
A tool returned an error and the agent retried. A search came back empty and the agent broadened the query. A sub-step failed silently and the agent noticed and re-planned. Or none of that happened, and the agent confidently delivered a final answer on top of a broken intermediate step.
-
Regression signals. These are evals we keep around to make sure the agent does not get worse at things it used to do well. Vivek Trivedy’s Better Harness post leans on this idea: every fixed bug should become part of the eval suite, so the harness cannot silently break old capabilities while improving new ones. This is how evals become a safety net for harness iteration.
-
Criteria freshness. The metrics that mattered three months ago aren’t necessarily the ones that matter now. The agent has gotten better, the use cases have shifted, and looking at the outputs has changed your sense of what good even means.
Shreya Shankar’s paper on criteria drift makes this explicit: people don’t know what they want until they see the agent’s behaviour and outputs, and the act of grading agent behavior changes the grader’s sense of what good looks like. A good harness eval process should expect the definition of quality to evolve.
The best teams do not use evals only to decide whether the system is good. They use them to locate where the harness is weak.
## What a Harness Eval Isn’t
The word eval gets used loosely, so it is worth separating harness evals from the things people often confuse with them.
-
Generic benchmark scores (like MMLU, HumanEval, Terminal-Bench): These can be useful for comparing models or agent setups, but they do not automatically tell whether a specific agent is doing the specific job well. A production harness eval has to be grounded in the workflow, tools, constraints, and outcomes the agent actually operates within.
-
Off-the-shelf quality metrics (like helpfulness, coherence, faithfulness, hallucination rate): These may be useful starting labels, but they are usually too generic to drive harness engineering. These metrics ship with every eval platform, which is exactly why they tell us nothing specific about our product.
-
Latency, cost, token usage: These are observability signals, not evals. Worth tracking, but they don’t tell us if the agent did the right thing.
-
Unverified LLM judges: A judge can be part of an eval, but it is not an eval by itself. Until it has been checked against human labels and shown to catch the failures that matter, it is another model output.
All of these measure something real. But a harness eval should answer a more specific question: Given this agent, in this environment, doing this job, did the system behave the way we wanted and recovering when it isn’t?
That usually means evaluating the outcome, the required actions, the recovery behavior, and the criteria used to grade it. Everything else is supporting instrumentation.
## How to Approach Building an Eval-Centric Harness
The default approach is to build the agent first, ship it, and bring evals in later to measure how it is doing.
An eval-centric approach works in the opposite direction: start with the behaviors you need to measure, then let those measurements shape the harness.
The categories mentioned above are what a good harness eventually needs to understand. The practical question is how a team gets there from nothing.
Start with one workflow, not the whole agent. Pick one job the agent should perform reliably: fixing a failing test, answering a refund-policy question, booking a meeting, researching a topic, or analyzing a dataset. A harness eval becomes useful only when it is grounded in a specific job.
Then read traces from that workflow before writing metrics. Look at what the agent saw, what it retrieved, what tools it called, what came back, where it recovered, and where it drifted. This is the least glamorous part of eval work, and probably the most important one: you have to LOOK AT THE DATA!! The final answer is not enough. For agents, the path often matters as much as the outcome.
The point is not to manually inspect traces forever. The point is to look long enough to understand the shape of failure before you turn it into a metric.
Once you have seen enough traces, label them. Ideally, by hand at first, and with someone who understands the domain.
Start by naming the failure modes: retrieval failure, tool-selection failure, tool-execution failure, citation failure, policy failure, escalation failure, completion failure. LLM-judge failure, etc. These names do not need to be perfect, they need to point to a harness lever. A retrieval failure points to retrieval. A tool-use failure points to tool design. A judge failure means the eval itself needs work.
This is also where Shreya Shankar’s criteria drift paper becomes important: you often do not know what you want from the agent until you have watched it fail in concrete ways.
Once patterns emerge from the labels, the temptation is to automate. That instinct is fine. Nobody wants to hand-grade traces forever, and LLM-as-a-judge is often the only practical way to scale evals beyond a small pile of hand-labeled examples. The danger is an unverified judge.
The judge will always give you a score. The question is whether that score reflects your product’s standards or just the judge’s own preferences.
The LLM-judge discussion between Hugo and Shreya puts this point well: judges can scale evaluation, but only after humans have done the work of defining what good looks like.
So the first job of an LLM judge is to earn the right to be trusted. Treat it like a classifier. Compare it against human labels, inspect disagreements, and check whether it catches the failure modes you actually care about. If the judge is wrong, the harness will optimize toward the wrong behavior.
That is the practical core of an eval-centric harness: start from behavior, find the failure mode, and let the eval point to the next harness change.
If I had to compress the whole approach into a protocol, it would be this:
-
Pick one workflow
-
Define end-to-end task success
-
Read real traces
-
Label failures with someone who understands the domain
-
Build local evals for the common failure modes
-
Validate any LLM judges against human labels
-
Turn fixed failures into regression cases
The details will vary by product, but this order matters: behavior first, metrics later.
## A Note on Where This Thinking Came From
The reason I care about eval-centric harnesses is that I have felt the gap between “we have evals” and “our evals actually help us build better systems.”
I saw this up close while working at Quotient AI, where production evals were not an abstract idea. They were part of the daily work of understanding agent behavior, finding failure modes, improving criteria, and making systems better over time.
A lot of my thinking here was shaped by reading and learning from people working seriously on this problem like Hamel, Vivek, Dex, Shreya. Would highly recommend going through their blogs and posts on X.
I also took Hamel’s AI Evals course, which helped me put more structure around ideas I had already encountered in practice. I was not new to evals. My background is in machine learning, and I had already worked around production systems. What the course helped clarify was the flow: how to move from messy traces to failure modes, from failure modes to domain-specific criteria, and from criteria to validated judges.
The part I found especially valuable was the discussion with peers. Seeing engineers, PMs, data scientists, and founders apply the same ideas inside very different pipelines made the concepts feel less abstract. It reinforced the point this post is trying to make: a useful eval is one that fits the domain, captures the behavior you actually care about, and creates a feedback loop for improving the system.
If this post resonated with you and feel like your evals are too generic, too disconnected from production behavior, or too dependent on unverified judges, Hamel’s course is the best next step I know of for learning the practice properly. You can check it out here: Hamel’s AI Evals course.
## Conclusion
Harness engineering is becoming a high-leverage skill for anyone shipping AI systems. But the best harnesses are not just bigger scaffolds around the model. They are systems that can notice failure, preserve what works, and improve deliberately.
Start small. Grab a few traces, build one solid custom eval, analyze the failures, and make one targeted change to the harness. You will feel the difference quickly.
The models will keep getting smarter. But the people who master eval-centric harnesses are the ones who will ship agents others can actually trust.
Happy harnessing!
👉 This essay draws on Antaripa’s experience at Quotient AI and on Hamel Husain and Shreya Shankar’s AI Evals course, which helped her organize that practical experience into a clearer workflow from traces to failure modes, domain-specific criteria, and validated judges. Hamel and Shreya are teaching their next cohort of the year starting September 6. Vanishing Gradients readers get 25% off using this link. 👈
## How You Can Support Vanishing Gradients
Vanishing Gradients is a podcast, workshop series, blog, and newsletter focused on what you can build with AI right now. More than 70 episodes with expert practitioners from Google DeepMind, Netflix, Stanford, and elsewhere. Hundreds of hours of free, hands-on workshops. All independent, all free.
If you want to help keep it going:
-
Become a paid subscriber, from $8 per month
-
Share this post with a builder who’d find it useful
-
Subscribe to the Vanishing Gradients YouTube channel
-
Join another Vanishing Gradients workshop
Thanks for reading Vanishing Gradients! This post is public so feel free to share it.
Share
40
5
9
Share
## Discussion about this post
CommentsRestacks
Rafael Ibayev
Oct 16
Agree that feedback is the under-discussed part. Most harness talk is about letting the agent do more - tools, terminals, sandboxes - and much less about knowing whether it did the right thing. One distinction I find useful: evals tell you whether a harness change helped across many runs, but you still need checks inside a single run, comparing what the agent claims against what actually changed. Both seem necessary, and they catch different failures.
Reply
Share
Yonis D.
Sep 24
Tools make the agent do more. Evals tell you whether that helped. Read traces before you invent metrics, then name failures that map to a harness lever: retrieval, tool selection, completion, judge. An unverified LLM judge is just another model output with a score attached.
Reply
Share
3 more comments...
TopLatestDiscussions
No posts
## Ready for more?
Subscribe
© 2026 Hugo Bowne-Anderson · Privacy ∙ Terms ∙ Collection notice
Start your SubstackGet the app
Substack is the home for great culture
