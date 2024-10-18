Minimal HTTP inference server in OpenAI API[^1].

> _When you don't want to install countless frameworks, generators, etc. When all you need is small Docker file and single main for http server._

> [!WARNING]  
> Limited OpenAI API compatibility.

- 100 lines of code
- CUDA
- Pytorch
- HuggingFace models (e.g. Llama 3.2 11B Vision)
- OpenTelemetry
- JSON schema output
- 150 token/s: NVIDIA L4 (GCP `g2-standard-8`) Llama 3.2 11B Vision Instruct

[^1]: https://github.com/openai/openai-openapi
