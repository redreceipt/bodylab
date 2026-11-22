# Roadmap

The goal: an avatar that looks almost exactly like you, from scanning you barely notice, that predicts how *you* will change under a plan, based on how your body has actually responded.

## Where it is now (runs on any Apple Silicon Mac)

- **Scanning:** continuous. Every useful camera frame from any angle is fused into one body per day, with width and depth from ellipse fits across yaw. It only measures you (repeated face matches, one person in view) and only bare skin. Each useful frame's outline and 3D landmarks are kept 45 days for refitting.
- **Body:** Anny (MakeHuman's mesh) fitted to every outline of the day, about a minute on the M4's GPU, shown in a relaxed stance (Anny's skeleton, skinned in the browser); without the optional install, MakeHuman's CC0 mesh stretched slice by slice to your measurements. Censored mode dresses it in underwear cut from its own surface. It has your photo's face, skin tone, hair and eyes; breasts sized from bust and underbust; genitals sized from tape measurements; draped athletic shorts and a sports bra, or nude.
- **Inputs:** all from the iPhone app: Apple Health sync (profile, about 20 metrics, workouts) and logging (food, weight, workouts, tape measurements that auto-calibrate the camera), plus a daily completeness checklist.
- **Predictions:** a daily energy-balance simulation (Hall energy densities, Forbes partitioning, adaptive maintenance, training and protein effects). Maintenance is learned from your food and weight once logged, and per-part fat patterns from your history.

## Next on an M4 Mac (no NVIDIA needed)

1. **Mirror how you're standing, live.** The browser already skins the body with Anny's bones, and the fit already solves a pose for every frame. Next: a fast solver (closed-form joint angles from MediaPipe's 3D landmarks, refined by a few steps of the same optimiser) streaming bone rotations at camera rate, plus MediaPipe Hands for fingers. Nothing new to install; if MediaPipe's landmarks prove too jittery, a dedicated pose model (e.g. a 3D human-mesh-recovery network) is the upgrade.
2. **Sharpen the body-shape fit (started).** Anny is fitted to all outlines now (split-half repeatability ±1–1.5 cm). Next: correct MediaPipe's mask spilling past the body (the fit's volume runs ~10 % high without a weigh-in), tape-measure limbs on the mesh, fit across several days. Fingers curl for a relaxed hand. The same fit on close nude frames is the only realistic route to camera-measured genital size.
3. **Photoreal skin and hair.** One 60-second iPhone walk-around video, then Gaussian splats trained locally on Apple Silicon (several Metal trainers exist: splatx-metal, mpsplat, MLX3D), then baked onto your fitted mesh as a full-body texture. Hair could render as splats (three.js has a `GaussianSplat` renderer; [Spark](https://sparkjs.dev) is the full-featured option).
4. **Your face shape.** The iPhone's TrueDepth camera (ARKit face mesh) gives true face geometry; the head morphs to match. The 478 3D landmarks from your photo are the fallback.
5. **Anatomy.** MakeHuman has no female genital geometry, and its male helper is low-poly (smoothed now). Both need sculpted, permissively licensed assets, for example sculpted in Blender with MPFB2 and released CC0, then fitted like other MakeHuman proxies.
6. **Predictions that learn.** Compare every past prediction with what happened, then correct the model's maintenance, lean-gain rate and fat-distribution terms per person. Show the error bars.
7. **Scanning you don't have to watch.** Spoken or haptic cues (Mac `say`, a speaker on the Pi, or iPhone haptics) when a new angle is captured, so the screen is never needed.

## With an NVIDIA GPU later

- Per-frame human mesh recovery models (e.g. Meta's [SAM 3D Body](https://github.com/facebookresearch/sam-3d-body) with MHR) as a strong starting point for shape and pose. Check the SAM license first.
- Animatable photoreal avatars (e.g. [LHM](https://github.com/aigc3d/LHM)-style single-image models, Gaussian avatars) that move like you.
- Clothed-body reconstruction, so loose clothing no longer has to be excluded.

## Ground rules

- Everything committed must be redistributable (CC0, MIT, Apache-2.0). Non-commercial models such as SMPL-X can only be optional downloads the user fetches themselves.
- Video and photos stay on the user's machine; only measurements, the face photo and its landmarks are stored.
