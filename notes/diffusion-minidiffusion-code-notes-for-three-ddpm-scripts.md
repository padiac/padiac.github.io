# MiniDiffusion - Code Notes

This document is a compact record mini diffuison. It focuses on how the code is structured and how it runs, not on theory derivations or timing benchmarks.

We cover:

- The shared DDPM training and sampling logic
- The simple MNIST script with a dummy CNN
- The CelebA / CIFAR-10 scripts with a UNet
- The `NaiveUnet` architecture (including its time embedding)

The aim is: if you forget the details later, you can read this once and immediately remember how everything fits together.

---

## 1. Shared DDPM Structure

All three training scripts (MNIST / CelebA / CIFAR-10) use the same core pattern:

1. **DDPM schedules and buffers**
   - Precompute `beta_t`, `alpha_t = 1 - beta_t`
   - Compute `alphabar_t = product(alpha_t)`, and from that:
     - `sqrtab = sqrt(alphabar_t)`
     - `sqrtmab = sqrt(1 - alphabar_t)`
     - `sqrt_beta_t`, `oneover_sqrta`, etc.
   - Stored with `register_buffer`, so they:
     - move with the model across devices
     - are not trainable parameters

2. **Forward pass (training)**  
   Given a clean image `x`:
   - Sample a timestep `t` uniformly
   - Sample noise `eps ~ N(0, I)`
   - Synthesize a noisy image:
     ```python
     x_t = sqrtab[t] * x + sqrtmab[t] * eps
     ```
   - Use a model `eps_model(x_t, t)` to predict the noise
   - Loss is MSE between true and predicted noise:
     ```python
     loss = criterion(eps, eps_model(x_t, t))
     ```

   In the code, `DDPM.forward(x)` directly returns this loss, so the training loop just calls `loss = ddpm(x)`.

3. **Sampling (reverse diffusion)**  
   To generate images:
   - Start from `x_T ~ N(0, I)`
   - For `i = T, T-1, ..., 1`:
     - Predict noise `eps = eps_model(x_i, i)`
     - Compute the mean of `p(x_{i-1} | x_i)` using the stored schedules
     - Add stochastic noise term when `i > 1`
   - The implementation looks like:
     ```python
     for i in range(self.n_T, 0, -1):
         z = torch.randn(...) if i > 1 else 0
         eps = self.eps_model(x_i, i / self.n_T)
         x_i = self.oneover_sqrta[i] * (x_i - eps * self.mab_over_sqrtmab[i])                + self.sqrt_beta_t[i] * z
     ```
   - Sampling is done inside `ddpm.sample(...)` with `model.eval()` and `torch.no_grad()`.

4. **Common training-loop structure**  
   Every script uses the same skeleton:
   ```python
   ddpm.train()
   for x, _ in dataloader:
       optim.zero_grad()
       x = x.to(device)
       loss = ddpm(x)          # DDPM.forward -> noise-prediction MSE
       loss.backward()
       optim.step()
   ddpm.eval()
   with torch.no_grad():
       xh = ddpm.sample(...)
       # save grid image
   ```

The main differences between scripts are: dataset, input shape, model used for `eps_model`, and normalization.

---

## 2. MNIST Script - Tiny Dummy Model

### 2.1 Dataset and preprocessing

- Dataset: `torchvision.datasets.MNIST`
- Images: single-channel, `1 x 28 x 28`
- Transform:
  ```python
  transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize((0.5,), (1.0,))
  ])
  ```

This shifts data from `[0,1]` roughly to `[-0.5, 0.5]`. It is a mild zero-centering, not full scaling to `[-1,1]`.

### 2.2 Dummy epsilon model (`DummyEpsModel`)

The model is a pure CNN with BatchNorm and LeakyReLU:

```python
blk = lambda ic, oc: nn.Sequential(
    nn.Conv2d(ic, oc, 7, padding=3),
    nn.BatchNorm2d(oc),
    nn.LeakyReLU(),
)
```

Stacked as:

```python
self.conv = nn.Sequential(
    blk(n_channel, 64),
    blk(64, 128),
    blk(128, 256),
    blk(256, 512),
    blk(512, 256),
    blk(256, 128),
    blk(128, 64),
    nn.Conv2d(64, n_channel, 3, padding=1),
)
```

Key points:

- There is no explicit UNet structure: no downsampling, no upsampling, no skip connections.
- The forward signature is `forward(self, x, t)` but `t` is ignored in the implementation.
- The author calls it "dummy" because it is a generic CNN, just good enough for MNIST.

### 2.3 Training loop highlights

- `ddpm.train()` enables BatchNorm training behavior.
- `loss_ema` is an exponential moving average used only to make tqdm's `loss:` display smoother:
  ```python
  if loss_ema is None:
      loss_ema = loss.item()
  else:
      loss_ema = 0.9 * loss_ema + 0.1 * loss.item()
  ```
- `optim.step()` is the only place parameters actually change.
- After each epoch, the script:
  - switches to `ddpm.eval()`
  - samples 16 images via `ddpm.sample(16, (1, 28, 28), device)`
  - uses `make_grid` to tile them into a `4 x 4` grid
  - saves the PNG and the model weights

MNIST sample grid produced by this script:

![MNIST DDPM samples after training](notes-assets/ddpm_sample_99.png)

This script is intentionally minimal and is a good baseline to understand the mechanics of DDPM.

---

## 3. CelebA and CIFAR-10 Scripts - Real UNet, Real Images

CelebA and CIFAR-10 share the same `DDPM` class and the same overall training logic as MNIST. They differ in:

- Input resolution
- Dataset
- Model: they use `NaiveUnet` instead of the dummy CNN
- Normalization configuration
- Batch sizes and DataLoader settings

### 3.1 CelebA version

Key settings:

- Model:
  ```python
  ddpm = DDPM(eps_model=NaiveUnet(3, 3, n_feat=128),
              betas=(1e-4, 0.02),
              n_T=1000)
  ```
- Transform:
  ```python
  transforms.Compose([
      transforms.Resize((128, 128)),
      transforms.ToTensor(),
      transforms.Normalize((0.5, 0.5, 0.5),
                           (0.5, 0.5, 0.5)),
  ])
  ```
  This maps RGB values from `[0,1]` to `[-1,1]`.
- Dataset: `ImageFolder(root=CELEBA_PATH, transform=tf)`
- DataLoader: `batch_size=32`, `shuffle=True`.

Sampling after each epoch:

- Generate 8 fake faces: `xh = ddpm.sample(8, (3, 128, 128), device)`
- Concatenate them with 8 real images from the last batch:
  ```python
  xset = torch.cat([xh, x[:8]], dim=0)
  ```
- `make_grid(..., normalize=True, value_range=(-1,1))` rescales from `[-1,1]` back to `[0,1]` when writing PNG.
- Save both the sample grid and the model checkpoint.

### 3.2 CIFAR-10 version

Key settings:

- Model is the same `NaiveUnet(3, 3, 128)`.
- Dataset: `torchvision.datasets.CIFAR10` with:
  ```python
  transforms.ToTensor(),
  transforms.Normalize((0.5, 0.5, 0.5),
                       (0.5, 0.5, 0.5))
  ```
- Image size: `3 x 32 x 32` (no explicit resize, CIFAR-10 is already 32x32).
- DataLoader: `batch_size=512`, `shuffle=True`.
- Training loop is identical in structure to the CelebA one.
- Sampling step:
  ```python
  xh = ddpm.sample(8, (3, 32, 32), device)
  xset = torch.cat([xh, x[:8]], dim=0)
  grid = make_grid(xset, normalize=True, value_range=(-1, 1), nrow=4)
  save_image(grid, f"./contents/ddpm_sample_cifar{i}.png")
  ```

CelebA and CIFAR-10 thus demonstrate the same DDPM logic applied to different resolutions and the same UNet backbone.

---

## 4. NaiveUnet Architecture (with Time Embedding)

This is the more realistic UNet used by the CelebA and CIFAR-10 scripts. It has:

- Residual conv blocks (`Conv3`)
- Downsampling paths (`UnetDown`)
- Upsampling paths (`UnetUp`)
- A simple time embedding (`TimeSiren`)
- Skip connections between encoder and decoder
- A final conv to match output channels

### 4.1 Conv3 block

```python
class Conv3(nn.Module):
    def __init__(self, in_channels, out_channels, is_res=False):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(),
        )
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(),
        )
        self.is_res = is_res

    def forward(self, x):
        x = self.main(x)
        if self.is_res:
            x = x + self.conv(x)
            return x / 1.414
        else:
            return self.conv(x)
```

Notes:

- `main` does the initial projection to `out_channels`.
- `conv` applies two extra conv + GN + ReLU layers.
- If `is_res=True`, the block adds the residual connection and scales by `1/sqrt(2)` to keep variance roughly stable.

### 4.2 Downsampling (`UnetDown`)

```python
class UnetDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.model = nn.Sequential(
            Conv3(in_channels, out_channels),
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        return self.model(x)
```

- `Conv3` processes features locally.
- `MaxPool2d(2)` halves spatial resolution (downsampling).

### 4.3 Upsampling (`UnetUp`)

```python
class UnetUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 2, 2),
            Conv3(out_channels, out_channels),
            Conv3(out_channels, out_channels),
        )

    def forward(self, x, skip):
        x = torch.cat((x, skip), 1)
        x = self.model(x)
        return x
```

Important detail:

- The forward first concatenates `x` and the `skip` tensor before applying `self.model`.
- Therefore, when constructing the UNet, `in_channels` for `UnetUp` must be set to `(channels of x + channels of skip)`.

### 4.4 Time embedding (`TimeSiren`)

```python
class TimeSiren(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.lin1 = nn.Linear(1, emb_dim, bias=False)
        self.lin2 = nn.Linear(emb_dim, emb_dim)

    def forward(self, x):
        x = x.view(-1, 1)
        x = torch.sin(self.lin1(x))
        x = self.lin2(x)
        return x
```

This is a very simple time-embedding module:

- Input `t` is a scalar per sample (normalized timestep) -> reshaped to `(B, 1)`.
- First linear layer without bias, then `sin` activation (SIREN-like flavor).
- Second linear layer to produce a vector of size `emb_dim`.

The time embedding is later reshaped to `(B, 2 * n_feat, 1, 1)` so it can be added to feature maps.

### 4.5 Full `NaiveUnet` forward

```python
class NaiveUnet(nn.Module):
    def __init__(self, in_channels, out_channels, n_feat=256):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_feat = n_feat

        self.init_conv = Conv3(in_channels, n_feat, is_res=True)

        self.down1 = UnetDown(n_feat, n_feat)
        self.down2 = UnetDown(n_feat, 2 * n_feat)
        self.down3 = UnetDown(2 * n_feat, 2 * n_feat)

        self.to_vec = nn.Sequential(nn.AvgPool2d(4), nn.ReLU())

        self.timeembed = TimeSiren(2 * n_feat)

        self.up0 = nn.Sequential(
            nn.ConvTranspose2d(2 * n_feat, 2 * n_feat, 4, 4),
            nn.GroupNorm(8, 2 * n_feat),
            nn.ReLU(),
        )

        self.up1 = UnetUp(4 * n_feat, 2 * n_feat)
        self.up2 = UnetUp(4 * n_feat, n_feat)
        self.up3 = UnetUp(2 * n_feat, n_feat)
        self.out = nn.Conv2d(2 * n_feat, self.out_channels, 3, 1, 1)

    def forward(self, x, t):
        x = self.init_conv(x)

        down1 = self.down1(x)
        down2 = self.down2(down1)
        down3 = self.down3(down2)

        thro = self.to_vec(down3)
        temb = self.timeembed(t).view(-1, self.n_feat * 2, 1, 1)

        thro = self.up0(thro + temb)

        up1 = self.up1(thro, down3) + temb
        up2 = self.up2(up1, down2)
        up3 = self.up3(up2, down1)

        out = self.out(torch.cat((up3, x), 1))
        return out
```

Reading this in stages:

1. **Initial conv**  
   `x` is projected to `n_feat` channels with a residual `Conv3` block.

2. **Encoder (down path)**  
   - `down1` keeps `n_feat` channels, halves spatial size.  
   - `down2` increases to `2 * n_feat`, halves again.  
   - `down3` keeps `2 * n_feat`, halves again.

3. **Bottleneck and time embedding**  
   - `to_vec` does `AvgPool2d(4)` + ReLU, compressing spatial size further.  
   - `timeembed(t)` produces a `(B, 2 * n_feat)` vector.  
   - `temb` is reshaped to `(B, 2 * n_feat, 1, 1)` so it can broadcast over spatial dimensions.
   - `thro = up0(thro + temb)` lifts the bottleneck back to a spatial map.

4. **Decoder (up path and skips)**  
   - `up1(thro, down3)`: concatenates `thro` and `down3` then upsamples/conv; then adds `temb` again to inject timestep information at this stage.
   - `up2(up1, down2)`: another upsampling stage with skip from `down2`.
   - `up3(up2, down1)`: final upsampling stage with skip from `down1`.

5. **Output layer**  
   - Concatenate the last upsampled feature `up3` with the original `x` from `init_conv`:
     ```python
     out = self.out(torch.cat((up3, x), 1))
     ```
   - Final `Conv2d` maps from `2 * n_feat` back to `out_channels` (e.g. 3 for RGB, 1 for grayscale).  
   - This tensor is the predicted noise `eps_hat(x_t, t)`.

This `NaiveUnet` is more realistic than the dummy CNN and is suitable for real diffusion experiments on CIFAR-10 and CelebA.

---

## 5. Big Picture

All three scripts are variations on the same theme:

- **DDPM** provides the schedules and loss/sampling logic.  
- **`eps_model`** is either a simple CNN (MNIST) or a UNet with time embedding (CIFAR-10 / CelebA).  
- **The training loop is always the same**:
  - load batch  
  - send to device  
  - compute noise-prediction loss via `ddpm(x)`  
  - backprop and `optim.step()`  
  - occasionally sample and save grids

If you keep this structure in mind, you can swap datasets, change architectures, and still know exactly how the training and sampling pipelines are wired together.
