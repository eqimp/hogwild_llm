from copy import deepcopy
from functools import lru_cache
from typing import Sequence, Tuple, Dict

import torch
import transformers
import shared_cache


def get_math_input_prompts(tokenizer: transformers.PreTrainedTokenizer,
                           workers: Sequence[str] = ("Alice", "Bob"),
                           include_examples: bool = True):
    """Create a namespace with formatting, prompt components, etc - this one is tuned for Qwen models"""
    _w = workers

    class Formatting:
        workers = list(_w)
        step_separator = SEP = '\n\n'
        history_header = "### Past steps".strip()
        work_in_progress_others = "### Work in progress (others)".strip()
        work_in_progress_self = "### Work in progress (own)".strip()
        begin_of_reasoning = "<think>"
        workers_collective = f'{", ".join(workers[:-1])} and {workers[1]}'
        pivot_message = "<...>".strip()
        s1_collab_message = "Quick check: am I doing redundant work? (yes/no): "
        finisher_s1_suffix = "\n\nWait, given the limited time, I have to give an answer right now. Considering all my previous attempts, I have to conclude that the final answer is \\boxed{"

        task_prefix = "**Problem:** "
        get_step_prefix = "**{0} [{1}]:** ".format
        forbidden_token_ix = [tokenizer.vocab[x] for x in ["#", tokenizer.bos_token, tokenizer.eos_token, '</think>']
                              if x in tokenizer.vocab]
        end_of_step_chars = ['.', '?', '!', '。', '۔', '؟', '।', '॥', '…', '‽', '།', '᠃', '։', '჻', '¶', '❧']  # before SEP
        final_answer_example = "\\boxed{number here}"

        @staticmethod
        def is_final_answer_given(generated_step: str) -> bool:
            if "\\boxed{" in generated_step:
                if "}" in generated_step[generated_step.index("\\boxed{"):]:
                    return True
            return False

        current_step_header = SEP + work_in_progress_others + SEP
        current_worker_header = pivot_message + SEP + work_in_progress_self + SEP

        # few-shot-ish examples
        make_example_incomplete = lambda question, answer: (
                "<example>\n\n" + tokenizer.apply_chat_template(
            [dict(role='user', content=question)],
            tokenize=False, add_generation_prompt=True
        ) + answer + "\n\n</example>")
        make_example_step = lambda step: f"<example>\n{step}\n\n</example>"

        rules = f"""
I will collaborate on this problem with another assistant. We will write our thoughts simultaneously and collaborate without redundant work. We can collaborate by doing different parts of the problem, double-checking each other's results, trying different approaches, or any other means.

There are {len(workers)} assistants, including myself. We will refer to each other as {workers_collective}.

We will solve the problem together, writing our thoughts in parallel. We will be able to see each other's past and current thoughts as we write them. We will see each other's previous steps as {get_step_prefix('AssistantName', 'step')}<...> .

In the '{history_header}' section, the automated system will gather the thoughts of {workers_collective} as we write them.

After the '{work_in_progress_others}' section, I will see the other assistants' unfinished steps. They will write those steps concurrently with me. I will take into account what they are doing. If another assistant gives me suggestions, I will address them.

I will always see *other* assistants' incomplete thoughts first, and then, after {work_in_progress_self}, my own step. Other assistants will continue writing their thoughts in their thoughts, while I will continue writing my own.

Since we both write our thoughts in parallel, I will initially see only partial (unfinished) thoughts that will be continued in parallel with mine. Others' thoughts will appear at the end of their unfinished step, near {pivot_message}. Other assistants may write new thoughts while I am writing mine.

I will use these partial thoughts to decide how best to collaborate without doing the same work twice. I will periodically check what other assistants are doing and I should adjust my actions based on what they are doing so we collaborate efficiently.

If what I am currently doing is the same that another assistant is doing or has already done, I will stop (e.g. {_w[0]} may say 'Wait, I was doing the same as {_w[1]} ...') and change to a different task right away.

When we are done with the problem, any one of us ({workers_collective}) can return the **final** answer as  {final_answer_example} , after which, we will no longer be able to update it. 
""".strip()

        example_first_step = make_example_incomplete(
            question=f"""
Solve two problems. Ann has 2 apples, Mark has 5 apples. How many apples do they have? Also, solve the equation x + y = 4, if y = 5. Write both answers in coma-separated format.
            """.strip(),
            answer=f"""
{begin_of_reasoning}{SEP}{history_header}{SEP}{work_in_progress_others}

{get_step_prefix(_w[0], 1)}I will solve the first problem. Ann has 2 apples, Mark has Wait, {_w[1]}, I am doing that already, please switch to the equation. If Mark has 5 apples, they have 2 +{pivot_message}

{work_in_progress_self}

{get_step_prefix(_w[1], 1)}Ann has 2 apples, Mark has 5 apples. How many... Wait, {_w[0]} is already solving that problem. I will instead solve x + y = 4, if y = 5: substituting y with 5 gives x + 5 = 4. The answer to the equation is
        """.strip())

        example_medium_calc = make_example_incomplete(
            question=f"""
Calculate S(x) = x + x^2 + x^3 + x^4 + x^5 for x = 1..10.
        """.strip(),
            answer=f"""
{begin_of_reasoning}{SEP}{history_header}

{get_step_prefix(_w[0], 1)}Here's how we should collaborate: I'll handle the odd numbers from 1 to 10, and you ({_w[1]}) can handle the even numbers. That way, we can split the work evenly. Let me start with x=1, 3, 5, 7, 9.

{get_step_prefix(_w[1], 1)}Here's how we should collaborate: Wait, {_w[0]} is already suggesting so I will listen. Let me wait for {_w[0]} to finish . . . . . . . Okay, {_w[0]} wants me to do odd numbers. I will do x=1, 3, 5, 7, 10.

{get_step_prefix(_w[0], 2)}For x=1: S(1) = 1 + 1 + 1 + 1 + 1 = 5. That's straightforward. I will do x=2 next. Wait, we agreed that {_w[1]} does even numbers and {_w[1]} is already doing x=2. I'll do x=3 instead.

{get_step_prefix(_w[1], 2)}Calculating S(2): 2 + 4 + 8 + 16 + 32. Let me add them step by step. 2+4=6, 6+8=14, 14+16=30, 30+32=62. So S(2)=62.

{get_step_prefix(_w[1], 3)}Calculating S(4): Let's compute each term: 4, 16, 64, 256, 1024. Adding them up: 4+16=20; 20+64=84; 84+256=340; 340+1024=1364. So S(4)=1364.

{get_step_prefix(_w[0], 3)}Next, x=3. Let's compute each term step by step to avoid mistakes. 3 + 9 (3²) is 12, plus 27 (3³) gives 39, plus 81 (3⁴) is 120, then add 243 (3⁵) to get 363. Wait, let me check again: 3 + 9 =12; 12+27=39; 39+81=120; 120+243=363. Yes, that's correct. Let me look to {_w[1]}'s progress to see how our computations are going. Looks like he computed x=2 and is currently computing x=4 in the incomplete step. Since we agreed that {_w[1]} handles even numbers, this checks out. We should continue as planned. 

{work_in_progress_self}

{get_step_prefix(_w[1], 4)}Calculating S(6): 6 + 36 + 216 + 1296 + 7776. Let me add step by step. 6+36=42; 42+216=258; 258+1296=1554; 1554+7776=9330. Wait, let me check again. 6^1=6, 6^2=36, 6^ {pivot_message}

{work_in_progress_self}

{get_step_prefix(_w[0], 4)}Next, x=5
    """.strip())

        example_debate_geom = make_example_incomplete(
            question=f"""
Three vertices of a cube are $P=(7,12,10)$ , $Q=(8,8,1)$ , and $R=(11,3,9)$ . What is the surface area of the cube?
        """.strip(),
            answer=f"""
{begin_of_reasoning}{SEP}{history_header}

{get_step_prefix(_w[1], 1)}Hey! Let me suggest splitting the problem into two parts. {_w[0]}, you can work on the first logarithmic equation to find possible relationships between x and y, while I can work on the second equation involving K. Once you have some relationships, I can substitute them into my part to solve for K. Let me know if you need help.

{get_step_prefix(_w[0], 1)}Here's how we should collaborate: Let me start by trying to find vectors between the points to see if I can find edge lengths or face diagonals. Since it's a cube, all edges are equal and the angles between edges are 90 degrees. Maybe we can compute the distances between P, Q, R and see which pairs are edges, face diagonals, or space diagonals. Let me compute the distances PQ, QR, and RP first.

{get_step_prefix(_w[0], 2)}Let's compute the distances between each pair of points to see the possible edge lengths.

{get_step_prefix(_w[0], 3)}Let me compute PQ first. The distance between P(7,12,10) and Q(8,8,1):

{get_step_prefix(_w[1], 2)}Alternatively, since three vertices are given, perhaps they form a triangle, and we can use coordinates to find edge lengths. Let me start by calculating the distance between P=(7,12,10) and Q=(8,8,1). The distance formula in 3D is sqrt[(x2-x1)^2 + (y2-y1)^2 + (z2-z1)^2]. Wait, {_w[0]} has a suggestion.

{get_step_prefix(_w[0], 4)}{_w[1]}, we are both doing PQ. Let me proceed.

{get_step_prefix(_w[1], 3)}Okay, I'll do QR then. QR is between Q=(8,8,1) and R=(11,3,9).

{work_in_progress_others}

{get_step_prefix(_w[1], 4)}The differences are 3 in x, -5 in y, 8 in z. So distance squared is 3{pivot_message}

{work_in_progress_self}

{get_step_prefix(_w[0], 5)}I will compute PQ now: x difference is 8-7=1, y difference 8-12=-4, z
""".strip())

        example_full_equation = make_example_incomplete(
            question=f"""
For each ordered pair of real numbers $(x,y)$ satisfying \[\log_2(2x+y) = \log_4(x^2+xy+7y^2)\] there is a real number $K$ such that \[\log_3(3x+y) = \log_9(3x^2+4xy+Ky^2).\] Find the product of all possible values of $K$ .
        """.strip(),
            answer=f"""
{begin_of_reasoning}{SEP}{history_header}

{get_step_prefix(_w[0], 1)}Hey! Here's how we should collaborate: Let's first tackle the first equation to find relationships between x and y, and then use those to find K in the second equation. I'll work on the first equation, and {_w[1]} can handle the second equation once we have expressions for x and y.

{get_step_prefix(_w[1], 1)}Hey! Let me suggest splitting the problem into two parts. {_w[0]}, you can work on the first logarithmic equation to find possible relationships between x and y, while I can work on the second equation involving K. Once you have some relationships, I can substitute them into my part to solve for K. Let me know if you need help.

{get_step_prefix(_w[0], 2)}Okay, starting with the first equation: log₂(2x + y) = log₄(x² + xy + 7y²). Hmm, since the logs have different bases, maybe I can convert them to the same base. Remember that log₄(z) is equal to (log₂z)/(log₂4) which is (log₂z)/2. So the equation becomes log₂(2x + y) = (1/2) log₂(x² + xy +7y²). Let me write that down:

{get_step_prefix(_w[1], 2)}Alright, I'll focus on the second equation. Let me write down what's given: log₃(3x + y) = log₉(3x² +4xy + Ky²). I know that log base 9 can be expressed as log base 3 squared. So log₉(w) = (log₃w)/2. Therefore, the equation becomes log₃(3x + y) = (1/2) log₃(3x² +4xy + Ky²). Multiply both sides by 2 to eliminate the fraction: 2 log₃(3x + y) = log₃(3x² +4xy + Ky²). Using logarithm properties, that's log₃((3x + y)^2) = log₃(3x² +4xy + Ky²). So the arguments must be equal: (3x + y)^2 = 3x² +4xy + Ky². Let me expand the left side: 9x² +6xy + y² = 3x² +4xy + Ky². Subtract the right side from the left: (9x² -3x²) + (6xy -4xy) + (y² - Ky²) =0 → 6x² +2xy + (1 - K)y² =0. So 6x² +2xy + (1 - K)y² =0. Hmm, so this equation must hold for the same x and y that satisfy the first equation. So I need to find K such that this holds given the constraints from the first equation. 

{get_step_prefix(_w[0], 3)}So, starting from log₂(2x + y) = (1/2) log₂(x² + xy +7y²). Let me let A = log₂(2x + y), so the equation becomes A = (1/2) log₂(...) which implies 2A = log₂(...) so 2^(2A) = x² + xy +7y². But since A is log₂(2x + y), 2^A = 2x + y. So 2^(2A) is (2x + y)^2. Therefore, (2x + y)^2 = x² + xy +7y². Let me expand the left side: 4x² +4xy + y² = x² + xy +7y². Subtract the right side from the left: 3x² +3xy -6y² =0 → simplifying: divide by 3: x² + xy -2y² =0. So x² +xy -2y² =0. Let me factor this quadratic in x: x² +xy -2y². Let me see: factors of -2y² that add to +y. Hmm, (x + 2y)(x - y) = x² +xy -2y². Yes! So (x + 2y)(x - y) =0. Therefore, either x = -2y or x = y. So possible solutions are x = -2y or x = y. 

{get_step_prefix(_w[0], 4)}So, the first equation gives x = -2y or x = y. Now, but we also need to ensure that the original logarithm arguments are valid. For log₂(2x + y) to be defined, 2x + y >0. Similarly, the argument of log₄ must be positive: x² +xy +7y² >0. Let me check for both cases:

{get_step_prefix(_w[0], 5)}Now, for validity: If x = -2y, then 2x + y = 2*(-2y) + y = -4y + y = -3y. But log₂(-3y) must be defined, so -3y >0 → y <0. Also, the other log's argument: x² +xy +7y². Plugging x=-2y: (4y²) + (-2y)(y) +7y² =4y² -2y² +7y² =9y². Since y is real, 9y² is always non-negative, but must be positive. Since y≠0 (since x=-2y, if y=0 then x=0, but then 2x+y=0 which is invalid). So y≠0 and y<0. 

{get_step_prefix(_w[1], 3)}Okay, so I have 6x² +2xy + (1 - K)y² =0. But from {_w[0]}'s first equation, there might be a relation between x and y that I can substitute here. Let me see what {_w[0]} found. She's working on the first equation. Let me check her steps. She arrived at x² +xy -2y² =0, which factors into (x + 2y)(x - y)=0, so x = -2y or x = y. So, these are the possible cases. Therefore, in each case, I can substitute x in terms of y into the equation I have (6x² +2xy + (1 - K)y² =0) and solve for K. Let me do that. Since there are two cases, maybe I should handle them separately. Since {_w[0]} is done with the first part, I can proceed with substitution. Let me start with the first case: Case 1: x = -2y. Substitute into my equation: 6*(-2y)^2 +2*(-2y)y + (1 - K)y² =0. Compute each term: 6*(4y²) =24y², 2*(-2y)y =-4y², so total: 24y² -4y² + (1 - K)y² =0 → (24 -4 +1 - K)y² = (21 - K)y² =0. Since y is real and in this case x=-2y, but y can't be zero (as earlier), so y≠0, so the coefficient must be zero: 21 - K =0 → K=21. 

{get_step_prefix(_w[0], 6)}For the case x = y: Substitute into 2x + y = 3x, which must be positive, so 3x >0 → x>0. Also, the other argument: x² +x*x +7x² = x² +x² +7x² =9x²>0, so x≠0. So x>0. So both cases are valid as long as the conditions hold. Now, moving on, since {_w[1]} is handling K, perhaps he can proceed. 

{get_step_prefix(_w[1], 4)}For Case 1 (x=-2y): So K=21. Now, moving to Case 2: x=y. Substitute x=y into my equation: 6x² +2x*x + (1 - K)x² =0 → 6x² +2x² + (1 - K)x² = (8 +1 - K)x² = (9 - K)x² =0. Again, x≠0 (since x=y and in this case x must be positive from my earlier validity check). So 9 - K =0 → K=9. Thus, possible K values are 21 and 9. Therefore, the product is 21*9=189. 

{work_in_progress_others}

{get_step_prefix(_w[0], 7)}Wait, {_w[1]}, let me confirm your calculation for the second case. When x=y, substituting into 6x² +2xy + (1 - K)y²: since x=y, that becomes 6x² +2x² + (1 - K)x² = (6+2+1 - K)x² = (9 - K)x². So yes, so 9 - K =0 ⇒ K=9. So the two K values are 21 and 9, so product is 189. Hmm, but let me double-check the first case. For x = -2y{pivot_message}

{work_in_progress_self}

{get_step_prefix(_w[1], 5)}Exactly, so 6+2=8 plus 1-K gives 9-K. Therefore, to have (9-K)x²=0 with x≠0, K must be 9. Therefore, the possible K values are 9 and 21. Their product is 189. Hey, {_w[0]}, what should I do next?
    """.strip())

        suggestions_on_collaborating = f"""
I will take into account what the other assistant is doing and change my actions accordingly. Here is how we can collaborate:

- **1. Strategizing:** we should think on how best to divide work between us (e.g. if {_w[0]} writes: {_w[1]}, please do this, then {_w[1]} should take this into account). If we disagree about what to do, we will default to {_w[0]}'s version.
- **2. Splitting:** we can split the problem into subtasks (simplify one equation or the other) and split the tasks between us. Prioritize subtasks that are not redundant (i.e. do not verify minor calculation done by another worker if there is another calculation that wasn't attempted yet).
- **3. Alternatives:** we can each try to solve a problem with different methods (e.g. calculate a mathematical expression by brute force vs mathematical derivations) and see which approach is faster.
- **4. Communicating:** we can look at each other's thoughts, ask each other questions (e.g. '{_w[0]}, which of these should I do first?'), give each other suggestions or corrections (e.g. 'Hey, {_w[1]}! You have a mistake in step 3 ...')
- **5. Announcing:** I can announce what I will do next (e.g. 'Let me try x=5 next' or 'I will double-check {_w[0]}'s result from step 5'). If another assistant says this, I will take it into consideration and do something else to avoid redundancy.
- **6. Reacting:** if I notice that another assistant is doing the same thing as I do, I should stop and think what else can I do to avoid redundancy. If I am ahead of the other assistant, I will instead ask them to change task problem.
- **7. Pivoting:** if I notice that what I am doing is no longer useful after change in circumstances, I will stop mid-sentence and pivot to another direction (e.g. '... computing p^4 | Wait, {_w[0]} is already on it, I should switch to adding up the results.')

We can also collaborate in any different way. We can invent new ways that would help us arrive at the correct solution faster.

To decide how best to collaborate, I will periodically, every few steps or more often, think what I am doing and if I am contributing or doing redundant work. If it is the latter, I will stop and choose something else to do to better contribute to solving the problem.
    """.strip()

        example_step_avoid_redundancy_1 = make_example_step(f"""
(previous steps omitted)

{get_step_prefix(_w[1], 7)}So the equation becomes: The right side is (a +30)(a +k) = a² +a k +30a +30k.

{get_step_prefix(_w[0], 5)}: Right. The right side is (a +30)(a +k) = a² +a k +30a +30k. So set equal to left side:

{work_in_progress_others}

{get_step_prefix(_w[0], 6)}Let me expand the left side: (a +2k)^2 = a² +4ak +4k². The right side is (a +30)(a +k) = a² + a k +30a +30k. Subtract right side from left side: (a² +4ak +4k²) - (a² +a k +30a +30k) =0 ⇒ 3ak +4k² -30a -30k =0. Let me factor terms: 3ak -30a +4k² -30k =0 ⇒ 3a(k -10) +k(4k -30)=0. Hmm, so {pivot_message}

{work_in_progress_self}

{get_step_prefix(_w[1], 8)}Wait, am I solving the same equation as {_w[0]}? Yes, it seems I am. {_w[0]} seems to have made more progress than me. Since I should not do the equation, what can I do instead? There is another case where a-d=-30 instead. Let me solve that part
""".strip())

        example_step_avoid_redundancy_2 = make_example_step(f"""
(previous steps omitted)

{work_in_progress_others}

{get_step_prefix(_w[0], 6)}Let's compute how many days each direction is used. Since 40 days divided by 4 directions gives 10 days per direction. So each direction (E,N,W,S) is used exactly 10 times. So for East, days 1,5,9,...,37 (step of 4). The same for the others. So we can compute the total East displacement as the sum over{pivot_message}

{work_in_progress_self}

{get_step_prefix(_w[1], 6)}Exactly. So for East, the days are 1,5,9,...,37 (total 10 days). Similarly for the others. Therefore, the total displacement in the East ... Wait, {_w[0]} is already doing that! I should compute N-S
    """.strip())

        example_step_avoid_redundancy_3 = make_example_step(f"""
(previous steps omitted)
{get_step_prefix(_w[0], 3)}Yes, expanding (a +2k)^2 gives a² +4ak +4k². Subtract a(a +k) which is a² +ak. So numerator is 3ak +4k². So:

{work_in_progress_others}

{get_step_prefix(_w[1], 3)}So equation: [k(3a +4k)] / (a +k) =30 ⇒ k(3a +4k) =30(a +k). Let me rearrange terms: 3a k +4k² =30a +30k. Let me bring all terms to left side: 3a k +4k² -30a -30k =0. Let me factor terms with a and terms{pivot_message}

{work_in_progress_self}

{get_step_prefix(_w[0], 4)}Wait, {_w[1]} has already started that equation before me and seems ahead of me, so I am doing redundant work. What can I do in the meantime?
    """.strip())

        combined_examples = f"""
# Examples

## 1. Basic example of collaborating within one step

{example_first_step}

## 2. Full example

{example_full_equation}

# How to collaborate

{suggestions_on_collaborating}

**Strategizing:**

{example_medium_calc}

**Communicating:**

{example_debate_geom}

**Detecting redundant work and changing task:**

{example_step_avoid_redundancy_1}

{example_step_avoid_redundancy_2}

{example_step_avoid_redundancy_3}
""".strip()

        final_system_prompt = f"""
# Collaborative Reasoning

{rules}

{combined_examples if include_examples else suggestions_on_collaborating}

# Solve the following problem

{workers_collective}, you will now solve the next problem together. Keep track of who does what work and communicate to avoid doing the same work twice.
        """.strip() + SEP

        @staticmethod
        def get_full_prompt(problem: str) -> str:
            return tokenizer.apply_chat_template(
                [dict(role='user', content=Formatting.final_system_prompt + Formatting.SEP + problem)],
                tokenize=False, add_generation_prompt=True)

        @lru_cache
        @staticmethod
        def get_stop_token_indices():
            _sep_token_index, = tokenizer.encode(Formatting.SEP, add_special_tokens=False)
            _sep_internal_str = {i: t for t, i in tokenizer.vocab.items()}[_sep_token_index]
            return {i for t, i in tokenizer.vocab.items() if _sep_internal_str in t}

        @staticmethod
        def is_end_of_step(worker_tokens) -> bool:
            if worker_tokens[-1] not in Formatting.get_stop_token_indices():
                return False
            step_string = tokenizer.decode(worker_tokens)
            step_string = step_string[:step_string.rindex(Formatting.SEP)].strip()
            return any(step_string.endswith(t) for t in Formatting.end_of_step_chars)

    return Formatting


def get_logits_processor(model: transformers.PreTrainedModel, forbidden_token_ix: Sequence[int]):
    """Create a transformers class that post-processes model logits for nucleus sampling, banned tokens, etc"""
    generation_config, model_kwargs = model._prepare_generation_config(model.generation_config)
    model._prepare_special_tokens(generation_config)
    device = next(model.parameters()).device
    return model._get_logits_processor(
        generation_config=generation_config,
        input_ids_seq_length=0,
        encoder_input_ids=None,
        prefix_allowed_tokens_fn=None,
        logits_processor=transformers.LogitsProcessorList([
            transformers.generation.logits_process.SuppressTokensLogitsProcessor(
                forbidden_token_ix, device=device)]),
        device=device,
        model_kwargs=model_kwargs
    )


def generate_reasoning_2agents(
    *,
    problem: str,
    model: transformers.PreTrainedModel,
    tokenizer: transformers.PreTrainedTokenizer,
    max_steps: int,
    save_on_steps: Sequence[int] = (),
    finisher_max_new_tokens: int = 8,
    insert_s1_collab_message_every_tokens: int = 256,
) -> Tuple[Dict[int, str], Dict[int, Sequence[int]]]:
    saved_states = dict()
    Formatting = get_math_input_prompts(tokenizer)
    tokenizer_kwargs = dict(return_tensors='pt', padding=True, padding_side='left', add_special_tokens=False)
    device = next(model.parameters()).device
    workers = Formatting.workers
    worker_prompts = [
        f"""{Formatting.get_step_prefix(workers[0], 1)}Hi, I'm {workers[0]}. Here's how we should do this:""".strip(),
        f"""{Formatting.get_step_prefix(workers[1], 1)}Hi, I'm {workers[1]}.""".strip()
    ]
    assert len(worker_prompts) == len(Formatting.workers)

    # make a sampler from model generation_config
    generation_config, model_kwargs = model._prepare_generation_config(model.generation_config)
    model._prepare_special_tokens(generation_config)
    logits_processor = get_logits_processor(Formatting.forbidden_token_ix)

    tokens_since_last_wait = 0
    cache_common, cache_current_step_header, cache_own_header, cache_w1, cache_w2 = (
        shared_cache.CacheBlock(config=model.config) for _ in range(5))
    cm = shared_cache.SharedCacheManager(cache_structure=[
        [cache_common, cache_current_step_header, cache_w2, cache_own_header, cache_w1],
        [cache_common, cache_current_step_header, cache_w1, cache_own_header, cache_w2],
    ])

    # pre-fill common cache parts
    with torch.inference_mode():
        model(**tokenizer(Formatting.get_full_prompt(problem), **tokenizer_kwargs).to(device),
              use_cache=True, past_key_values=cache_common)  # <-- write to common prompt

        model(**tokenizer(Formatting.current_step_header, **tokenizer_kwargs).to(device),
              use_cache=True, past_key_values=cache_current_step_header)  # <-- write to the separator after history

        model(**tokenizer(Formatting.current_worker_header, **tokenizer_kwargs).to(device),
              use_cache=True, past_key_values=cache_own_header)  # <-- write to separator between incomplete steps

    # generate interdependent reasoning chains in parallel
    current_step_index_by_worker = [1, 1]
    current_step_tokens_by_worker = tokenizer(worker_prompts, add_special_tokens=False)['input_ids']
    generation_stopped = False
    history = list()

    next_inputs = tokenizer(worker_prompts, **tokenizer_kwargs).to(device)
    for inference_step in range(max_steps + 1):
        if inference_step in save_on_steps or inference_step == max_steps:
            saved_states[inference_step] = dict(
                history=list(history), current_step_tokens_by_worker=deepcopy(current_step_tokens_by_worker)
            )
        if generation_stopped or (inference_step == max_steps):
            continue  # if the generation finished early, copy the generation state for the rest of the budget

        # run model with a shared cache (batched inference)
        with torch.inference_mode():
            logits = model(**cm.get_input_kwargs(**next_inputs)).logits[..., -1, :]
            logits = logits_processor(next_inputs['input_ids'], logits)
            new_tokens = torch.multinomial(logits.softmax(dim=-1), 1).flatten(
                ) if generation_config.do_sample else logits.argmax(-1)
            assert len(new_tokens) == len(Formatting.workers)

        # process generated tokens for printing; handle step change, update next_inputs
        next_input_tokens = new_tokens.unsqueeze(-1).tolist()
        for worker_index, (worker_name, worker_tokens, new_token) in enumerate(
                zip(Formatting.workers, current_step_tokens_by_worker, new_tokens.tolist())):
            worker_tokens.append(new_token)
            if Formatting.is_end_of_step(worker_tokens):
                if Formatting.is_final_answer_given(tokenizer.decode(worker_tokens)):
                    generation_stopped = True
                # worker just finished their step - add it to common history and start a new step
                current_step_index_by_worker[worker_index] += 1
                history.extend(worker_tokens)
                worker_tokens.clear()
                start_msg = Formatting.get_step_prefix(worker_name, current_step_index_by_worker[worker_index])
                if tokens_since_last_wait > insert_s1_collab_message_every_tokens:
                    start_msg += Formatting.s1_collab_message
                    tokens_since_last_wait = 0
                worker_tokens.extend(tokenizer.encode(start_msg, add_special_tokens=False))
                cache_common.append_from(cm.cache_structure[worker_index][-1])
                cm.cache_structure[worker_index][-1].clear()
                next_input_tokens[worker_index] = [new_token] + worker_tokens
            tokens_since_last_wait += len(next_input_tokens[worker_index])
        next_inputs = tokenizer.pad(dict(input_ids=next_input_tokens), padding_side='left', return_tensors='pt').to(device)

    # generate outputs (strings) with optional early finisher
    outputs = dict()
    for budget, saved_state in saved_states.items():
        output = tokenizer(Formatting.get_full_prompt(problem), **tokenizer_kwargs)['input_ids'].flatten().tolist()
        output.extend(saved_state["history"])
        output.extend(tokenizer.encode(Formatting.current_step_header, add_special_tokens=False))
        for worker_index, worker_tokens in enumerate(saved_state['current_step_tokens_by_worker']):
            output.extend(worker_tokens)
            output.extend(tokenizer.encode(Formatting.pivot_message + Formatting.SEP, add_special_tokens=False))

        response: str = tokenizer.decode(output)
        if finisher_max_new_tokens > 0:
            response: str = tokenizer.decode(model.generate(
                **tokenizer(response + Formatting.finisher_s1_suffix, **tokenizer_kwargs).to(device),
                max_new_tokens=finisher_max_new_tokens, use_cache=True, do_sample=False).flatten().tolist())
        outputs[budget] = response
    return outputs, saved_states
