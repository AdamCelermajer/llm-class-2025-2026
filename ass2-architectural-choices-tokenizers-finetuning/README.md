# Assignment 2 - architectural choices, decoding and fine tuning

## Architectural Choices

Huggingface is s software company that hosts a repository of models, both LLMs and others. These are models you can download, inspect, run and finetune.

In this part of the assignment you will explore some recent LLMs from the huggingface model library, and inspect their architectural choices.

We will be working with the following models:
1. `meta-llama/Llama-3.1-8B-Instruct`
2. `mistralai/Mistral-7B-Instruct-v0.3`
3. `Qwen/Qwen2.5-7B-Instruct`
4. `allenai/OLMo-2-1124-7B-Instruct`
5. `ibm-granite/granite-3.3-8b-instruct`
6. `deepseek-ai/DeepSeek-V3`
7. `HuggingFaceTB/SmolLM2-1.7B-Instruct`
8. `microsoft/Phi-4-mini-instruct`
9. `tiiuae/Falcon3-7B-Instruct`
10. `dicta-il/dictalm2.0-instruct`

### Extract architectural information

For each model, I want you to list:
1. Number of layers, width of layers
2. Number of attention heads
3. Sizes of everything: if it has MLP layers, what dimensions are they? if it has MoE layers, how many experts, and what are the dimensions involved? what are the dimensions of the different attention components? what are the sizes of the embeddings and the unembedding layers? etc.
4. How does it handle position encoding? does the method have a max position it supports? if so, what is it? does the method have hyper-parameters? if so, what are they and what values do they take?
5. What activations are used? is it the same activation everywhere, or do they mix different kinds?
6. What kind of normalizations are used, and on which components? is the model using pre-norm, post-norm, or something else?
7. Any other property of interest you may think of.

Some of these properties may be easier to extract than others. Part of the assignment is figuring out how to locate this information the most easily.
For each property you extract, mention (in your report) where you extracted it from and how.

In the report, include:

1. A short explanation of how you extracted the architectural information.
2. Any uncertainty, missing fields, or disagreements between sources.
3. A readable summary of the main architectural differences between the models.
4. The reflection and analysis requested below.

In addition to the report, submit `architecture.csv`. This file is for the structured model-by-model facts, not for long explanations. It should contain one row per model and include, when available, the following columns:

```text
model_id,hidden_size,num_layers,num_attention_heads,num_kv_heads,mlp_size,activation,norm_type,position_encoding,context_length,vocab_size,moe_details
```
Use `NA` for fields that do not apply or that you could not find.

### Organize, reflect and analyze

Inspect the extracted information and try to reach some general trends. I want to see some interesting insights about differences and commonalities in architectural choices.

Some kinds of questions to consider: are there choices that are consistent in all or most models? are there choices in which there is no consensus? are some specific models deviating from the norm? are there some ratios or rules of thumb that govern sizing choices (that is, "size x is often chosen to be k times the size of y", or "size x is often a power of 10")? etc

Based on these choices, if you were to build a new model, what settings would you choose?

## Tokenizers

One architectural choice we did not cover is the tokenization. We will cover it now.

Consider the tokenizers of the 10 models above. 

### Architectural choices

- How many tokens are there in the tokenizer?
- What strategy is the tokenizer using for indicating that a token is not an incomplete word? how are tokens supposed to merge back to words?
- Are there tokens that do not corresponds to words or parts of words? How many, what kind of things to they encode, and how are they represented?
- For each tokenizer, compute or estimate the average number of tokens per word for English and for Hebrew. Explain exactly how you computed or estimated this number, and what assumptions your method relies on.

Note that some of these answers may repeat across models. It is ok to respond with "the same as in model X" if that is indeed the case.

In the report, include:

1. A short explanation of how you inspected the tokenizers.
2. How you computed or estimated the average-tokens-per-word numbers.
3. How you counted words.
4. The tokenization-differences example requested below.
5. A discussion of the average-tokens-per-word numbers and what does it mean.


In addition to the report, submit `tokenizers.csv`. This file is for the structured tokenizer facts. It should contain one row per model and include, when available, the following columns:

```text
model_id,tokenizer_type,vocab_size,special_tokens,word_boundary_strategy,byte_fallback_or_byte_level,avg_tokens_per_english_word,avg_tokens_per_hebrew_word
```
Use `NA` for fields that do not apply or that you could not find. 

### Tokenization differences

Find an English text that is tokenized differently by at least three models. Provide the text, the different tokenizations by the three models, and a short description of what you think could be the cause of the difference.

How many of the remaining 7 tokenizers agree with each of the 3 tokenizers on this text?

## Constrained Decoding

### Get basic inference working

Load the `Qwen/Qwen2.5-7B-Instruct` model and the `mistralai/Mistral-7B-Instruct-v0.3` model, and see that you manage to query them with prompts containing English questions and obtain reasonable answers.

These are instruction-tuned models ("instruct" models). This means they are not only trained to continue text, but were further trained to respond to user instructions, usually in a chat-like format with user and assistant turns. When running them, make sure you use the tokenizer's chat template or the prompting format recommended by the model card; otherwise the model may behave worse than expected.

### Identify Hebrew-related tokens

For each of these two models, identify all the tokens in the tokenizer that are may participate in Hebrew text. This should include numbers and punctuation, but not words in other languages. (what was your strategy for identifying these tokens?)

Submit one JSON file per model with the Hebrew-allowed token set. Name the files:

1. `hebrew_allowed_tokens_qwen.json`
2. `hebrew_allowed_tokens_mistral.json`

Each file should use the following structure:

```json
{
  "model_id": "...",
  "allowed_token_ids": [1, 2, 3]
}
```

The full explanation of the token-selection strategy should be in the report.

### Constrained decoding

Modify the inference procedure to do simple constrained decoding of the following form: only allow the tokens from the set identified above to be generated.

Run the English queries from above (or others) using the constrained decoder. What answers do you get from each model? 

1. How did you implement the constrained decoding? describe briefly.
2. List 10 English language queries, and for each of them, for each model: (a) the model response with unconstrained decoding; (b) the model response with constrained decoding. 
3. A brief discussion of the results. Is the Hebrew text meaningful? does it relate to the question? does it differ between the models? are some languages favored over others?

Submit the decoding results in a file named `decoding_outputs.jsonl`. Each line should include: prompt, model, unconstrained output, constrained output, and the decoding parameters.

The report should contain the explanation and discussion: how you implemented constrained decoding, and what you conclude from the outputs. The JSON and JSONL files are for the raw token lists and generation outputs.

## Fine Tuning

In the previous section, you attempted to adapt a model to answer in Hebrew on English queries by intervening with the decoding procedure. Now we will do it with fine tuning.

You will be working with the `Qwen/Qwen2.5-1.5B-Instruct` model. This is a relatively small model, that should be easy to load and fine-tune also on a relatively weak GPU, like the ones available in Google Colab. It also has some basic knowledge of Hebrew, as you can see by prompting it in Hebrew and observing its answers.

Your job is to fine-tune the model so that it answers in Hebrew also on English queries. Ideally, the answers should be relevant to the query prompt (that is, it is easy to tune the model to always return a fixed Hebrew sentence like "לא יודע" but this is not a valid solution. The model should at least appear to be attempt to answer the prompt question/instruction, even if it will get it wrong).

This part of the assignment is open ended, that is on purpose. You are free and encouraged to explore various approaches, to find the one that works best. At the end it will boil down to "supervised fine tuning" (SFT): providing the model with input-output example pairs, and training it to follow them, with hope of generalization. So the task has two parts: creating the data, and training on the data you created. These parts may be interleaved, as data creation choices may be informed by experience with training on it.

### Data 

For obtaining the training data, you can either find relevant training data for this task, or create your own. Creating your own can be done from scratch, or based on some existing data which you will modify. You can also use queries to LLM to help you in creating the data, either in performing transformations (for example, translation) on existing data, or creating new data from scratch based on your instructions. Do whatever is the most effective for performing the task. 

Your training data must not include the evaluation inputs used in the "Did it work?" section below.

### Fine-tuning

For fine-tuning the model, you can either use full fine tuning, or a method for lightweight fine-tuning such as LoRA (we will learn about LoRA later in the course, but you can use it also without this knowledge). It is always a good idea to see that you can get basic fine-tuning to work on a very simple task (for example, overfit it to always answer a given prompt with a given answer) before attempting an open-ended one for the first time. This way you will know that you at least have the basic mechanics of training to work. Here are some additional technical tips on fine-tuning that you may find useful:

1. LoRA is recommended over full fine tuning. The main library for LoRA in the Hugging Face ecosystem is `peft`.
2. Start by checking that regular inference works before training.
3. Then try to overfit a tiny dataset of one or two examples. This is a useful sanity check that training is actually changing the model.
4. Use the model's chat template consistently for both training and inference.
5. Make sure the training examples match the actual task: English prompt in, Hebrew answer out.

### Did it work?

Finally, you will have to prove that your fine-tuning worked. This part is also left open-ended. Do something to convince us that the method worked, and the model is behaving as expected. The common way to do it is by (a) constructing an evaluation set and measuring some performance metrics on it; and (b) showing some anecdotal input-output pairs both before and after fine-tuning. Both of these parts are needed, but the composition of each part is left up to you. Describe you choices and try to convince us.

For evaluation, run both the original model and your fine-tuned model on:

1. 10 English inputs provided in the assignment.
2. 10 additional English inputs of your own choice.

Use these 10 English evaluation inputs:

1. Explain why the sky looks blue during the day.
2. Give two advantages and two disadvantages of public transportation.
3. Write a short email asking a professor for an extension on an assignment.
4. Describe how to make a simple omelette.
5. What is the difference between supervised and unsupervised learning?
6. Summarize the story of Cinderella in three sentences.
7. Suggest three ways to reduce smartphone distraction while studying.
8. Explain what happens when water boils.
9. Give a polite refusal to an invitation to a party.
10. Turn the idea "practice makes progress" into advice for a student.

The training data must not include any of these 20 evaluation inputs.

In the report, include the outputs on all 20 inputs and explain what changed after fine-tuning. You may also include quantitative summaries, such as the percent of answers that are in Hebrew and a manual relevance score.

Submit these results in a file named `eval_outputs.jsonl`. Each line should include: prompt, base-model output, fine-tuned-model output, and short notes if relevant.

The report should contain the explanation and conclusions: how you created the data, how you fine-tuned the model, what evaluation you ran, and whether you think the fine-tuning worked. `eval_outputs.jsonl` is for the raw before/after generations.

# What to submit?

A report in PDF format. The report should cover answers and discussions of all the sections and subsections mentioned above, and it should be easy to navigate and locate the individual sections and their answers. The report should also contain your names and ID numbers in a prominent font at the top. 

Large part of your grade will be based on the report quality and clarity: the report is the main window we have into your work. So you are encouraged to put some effort into it. 

In addition to the report, submit:

1. `architecture.csv`
2. `tokenizers.csv`
3. `hebrew_allowed_tokens_qwen.json`
4. `hebrew_allowed_tokens_mistral.json`
5. `decoding_outputs.jsonl`
6. `eval_outputs.jsonl`
7. The code used to identify tokens, perform constrained decoding, create or process the data, fine-tune the model, and run the evaluation.
8. The training data, if it fits in Moodle.

If the training data has more than 1000 examples, submit the first 1000 examples. This does not mean your data has to be of this size; it can also be either much smaller or much larger. If the full training data does not fit in Moodle, submit a representative sample plus the script or method used to generate it.

Do not submit full model weights. If you trained a LoRA adapter and it is small enough, you may submit it, but the report and required output files should be sufficient for grading.
