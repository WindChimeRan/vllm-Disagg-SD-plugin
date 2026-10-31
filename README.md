# vllm-Disagg-SD-plugin

## Project Goal

a plug-in for vllm. 
- Single-Node Multi-Card Disagg-SD: 1 card for speculative drafting, other cards for verification. (start with 1:1)
- Multi-Node Disagg-SD: local DGX-Spark/vLLM-metal for drafting, cloud B300 for verification. (start with DGX-Spark + 1xB300)
- compatible with the newest vllm

## Main Disagg Algorithm (batching, scheduling .etc)
- The most straightforward disagg algorithm, batch async like MineDraft https://arxiv.org/html/2603.18016v3
- My own algorithm, which requires retraining a dflash1 in a skip-gram way ....

## Main drafter

- DFlash1 only
- Others can be used as baseline comparison. But Disagg integrations are not planned.

## Main challenge
- input / output of drafter could be very big --> communication bound
- drafter's kv cache would be hard to manage ... H-Spec could simplify this (https://arxiv.org/pdf/2609.24197)

## Related work:

- https://arxiv.org/html/2603.18016v3
- https://arxiv.org/pdf/2511.11733
- https://arxiv.org/pdf/2407.11798
