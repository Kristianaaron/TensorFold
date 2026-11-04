# GLM-5.3 WP1 fixtures

`config.json` is a copy of the public GLM-5.3 tokenizer `config.json` (no weights).
The `census/` directory is a tiny synthetic index and safetensors header used by `test-glm53-cuda-host`.

After the assembled build is available, build and run the census with:

```sh
zig build tf-glm53-census
zig-out/bin/tf-glm53-census /path/to/assembled/GLM-5.3
```
