How to:

```bash
docker build -t my-gpu-env . -f .devcontainer/Dockerfile
docker run --gpus all --rm -v $(pwd):/app -it my-gpu-env bash
```

For dockercontainer need old vscode version 1.108.1