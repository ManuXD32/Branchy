# Branchy: a language model split into parallel branches across your devices

Branchy is a research prototype. It asks whether a group of modest devices (a laptop, a phone, a small server) in
different places, connected by a VPN, can run a language model together better than the usual way of splitting one.

The usual way (llama.cpp RPC, Petals) puts **consecutive layers** on different machines. The work stays strictly
sequential and every token waits for one network round trip per machine.

Branchy trains a transformer whose middle part is a set of **parallel branches**. Each device computes one or more
branches at the same time, and one device fuses them with a single linear layer. Two training choices make this work
over a slow network:

- **One-token delay.** Remote branches are used one token late, so the coordinating device never waits for the
  network inside a token.
- **Branch dropout.** Branches are randomly dropped in training, so a device can be slow or missing.

Everything here was measured, including where the idea does not pay. The full write-up is the article:
[`branchy-article.pdf`](branchy-article.pdf)

## What we found

Two models of 300M parameters trained from scratch on TinyStories, one ordinary ("sequential") and one with 8
branches, on a server, a laptop (Ryzen 5 5500U) and a phone (Snapdragon 8 Gen 3) with 20 to 50 ms between them:

| Question | Result |
|---|---|
| What do branches cost in quality? | Held-out loss 1.255 against 1.143 for the sequential model after the same training. A blind judge prefers the branched model's story in 45% of comparisons. |
| Does the delay remove the network wait? | Yes. The coordinator waits 0.1 ms per token for round trips of 49 to 66 ms. Output identical to a single device (largest logit difference 8e-6). |
| Branch split against layer split, same devices | The branch split is 1.5 to 2.6 times faster. The layer split halves the speed of the coordinator alone. |
| Is it faster than one device when the model fits there? | Not on our unequal devices: the phone alone (14 to 24 tokens/s, depending on the session) beats every placement (best 16.2). On two equal simulated devices the branch split is 1.2 times faster than one of them. |
| And when the model does not fit in a device's RAM? | That is where a split pays. A device that must read weights from storage is 6 to 37 times slower (simulated: 8.1 → 0.9 tokens/s; real laptop 8.9 → 0.67; real phone 23.9 → 0.65), and 27 times slower with a pretrained 1.7B model on the laptop (2.30 → 0.08). Laptop and phone sharing the 300M model, each with too little RAM for it, are back at 7 tokens/s with the layer split and 9 to 11 with the branch split, over a real 46 ms VPN. |
| Can a device drop out? | One of seven remote branches missing costs +0.009 loss, three cost +0.044. |
| Can pretrained models be reshaped into branches? | Only for few, long branches. Layouts that shorten the token path a lot lose 0.6 to 0.9 in loss even after healing. A large branched model has to be trained in that shape. |

A projection, not a measurement: with eight comparable devices and 20% of the model on the coordinator, a token costs
about 30% of a full pass (3.3 times a single device that could hold the whole model), while a layer split stays at a
full pass plus seven round trips. Our two-device measurement matches the formula (predicted 1.7, measured 1.6 to 1.7).

## What is here

```
src/node/            worker daemon ("node"), control tools, placement benchmark, coordinator code
src/webui/           coordinator web UI: cluster status, side-by-side generation, benchmark
src/proxy/           the branched and sequential models and their trainers
src/quality/         quality benchmark, blind story judging, weight rounding check
src/lowmem/          simulated low-memory devices
src/pretrained/      reshaping pretrained models into branches (LFM3, Qwen)
scripts/             setup and benchmark scripts with your own worker names
```

## Quick start

```bash
pip install -r requirements.txt
BASE=<where the model files are published> bash scripts/get_models.sh   # two models, 1.25 GB each: MODELS.md
bash scripts/setup_coordinator.sh <address of this machine>             # secret, worker bundle, cluster file
# install a worker on each device with the one-line command it prints, list them in src/node/cluster.json, then:
bash scripts/start_webui.sh                                              # http://127.0.0.1:8780
```

Step by step: [`SETUP.md`](SETUP.md). Which script reproduces which result: [`REPRODUCE.md`](REPRODUCE.md).
Without any worker you can still run both models on one machine in the web UI and compare them.

## Limits you should know

- A research prototype: workers compute in PyTorch with 32-bit floats; nothing is optimised like llama.cpp.
- The models are small story models. The quality gap at larger scale and on broad text is unknown.
- Measured with at most three real devices; results beyond that are projections.
- A worker executes code sent by the coordinator for anyone who knows the shared secret. Private networks only.

## Licence and credits

Created by **ManuXD32**. You are free to use this, build on it and research with it:

- **Code** (`src/`, `scripts/`): Apache License 2.0 (`LICENSE`). Keep the `NOTICE` file when you redistribute.
- **Article, documents, results and the trained models**: CC BY 4.0 (`LICENSE-DOCS-AND-MODELS.md`). Give credit and
  link back here.

If this helps your work, please cite it: GitHub shows "Cite this repository" from `CITATION.cff`, or use

> ManuXD32. Branchy: parallel learned branches for language-model inference across devices on a wide-area network. 2026.

The models were trained on TinyStories (Eldan and Li, 2023). LFM3 and Qwen weights are not included and stay under
their publishers' licences. The layer split is our own implementation of the layout used by llama.cpp's RPC backend,
not its code. Code and the article were produced with thee help of Claude under my direction and supervision while I orchestrated the experiments