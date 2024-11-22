# Setup

- [ ] look into uv's env setup and an entrypoint for the CLI again

# Cache viewer tui
- [ ] minimal cache viewing utility to manage data/checkpoint/run caches
  - 2 pane (list w/ select, config view pane) (textual library)
- [ ] allow for selective deletion/exports

# Sweep (implement a wandb sweep search)
- [ ] new entrypoint for hparam search (modify config to have ranges)
- [ ] include bash script to parallelize

# Profiling and performance

- [ ] look into torch compile.config with composer to speedup
- [ ] see if speedup with FlexAttention vs SDPA attention
- [ ] also see if flashattention is enabled?

```
https://docs.mosaicml.com/projects/composer/en/latest/_modules/composer/trainer/trainer.html#:~:text=..%20seealso%3A%3A%20The%20%3Amod,None%60%60.%20(default%3A%20%60%60None%60%60)
```
