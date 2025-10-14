## Map the Flow: Revealing Hidden Pathways of Information in VideoLLMs

> [**Map the Flow: Revealing Hidden Pathways of Information in VideoLLMs**]() \
> [Minji Kim*](https://byminji.github.io), [Taekyung Kim*](https://scholar.google.co.kr/citations?user=u-9bdkwAAAAJ&hl=en), [Bohyung Han](https://cv.snu.ac.kr/index.php/~bhhan/) <br>
<sub> (* Equal Contribution) <br>

[![website](https://img.shields.io/badge/Project-Page-yellow.svg)](https://map-the-flow.github.io/)
[![paper](https://img.shields.io/badge/arXiv-Paper-red.svg)]()


Official PyTorch implementation of the paper "**Map the Flow: Revealing Hidden Pathways of Information in VideoLLMs**"


### Abstract
> Video Large Language Models (VideoLLMs) extend the capabilities of vision-language models to spatiotemporal inputs, enabling tasks such as video question answering (VideoQA). Despite recent advances in VideoLLMs, their internal mechanisms on where and how they extract and propagate video and textual information remain less explored. In this study, we investigate the internal information flow of VideoLLMs using mechanistic interpretability techniques. Our analysis reveals consistent patterns across diverse VideoQA tasks: (1) temporal reasoning in VideoLLMs initiates with active cross-frame interactions in early-to-middle layers, (2) followed by progressive video-language integration in middle layers. This is facilitated by alignment between video representations and linguistic embeddings containing temporal concepts. (3) Upon completion of this integration, the model is ready to generate correct answers in middle-to-late layers. (4) Based on our analysis, we show that VideoLLMs can retain their VideoQA performance by selecting these effective information pathways while suppressing a substantial amount of attention edges, e.g., 58% in LLaVA-NeXT-7B-Video-FT. These findings provide a blueprint on how VideoLLMs perform temporal reasoning and offer practical insights for improving model interpretability and downstream generalization.


### Updates
Code will be released soon.
