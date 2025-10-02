# FDiT

FDiT (Flow Diffusion Transfomers)

## Dependencies settings

```bash
    pip install -r requirements.txt
```

## 1. Datasets processing
Download datasets UTD-MHAD from [link](https://personal.utdallas.edu/~kehtar/UTD-MHAD.html)
- Cut frames in video datasets, runing:
```bash
    python -m preprocessing.preprocess_MHAD
```

## 2. Training Latent Flow Auto-Encoder (LFAE)
- LFAE used determine positions on frames
```bash
    python -m LFAE.run_mhad
```
- Continue training LFAE from checkpoint
```bash
    python -m LFAE.run_mhad --checpoints /path/checkpoint --set-start True
```

## 3. Training Diffusion Transformers model
- After training and save to checkpoint LFAE, continue training transformer diffusion model to synthesis video from single image input
```bash
    python -m DM.train
```
- Continue training diffusion transformer model from checkpoint
```bash
    python -m DM.train --checpoints /path/checkpoint --set-start True
```

## 4. Synthesis video from single images
```bash
    python -m demo.demo_mhad
```

- Evaluation FID and IS metrices
```bash
    python -m demo.eval
```