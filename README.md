# BodyLab

Track body composition by walking past a camera. BodyLab watches a network camera, measures your silhouette, estimates circumferences and body fat, and fits a realistic 3D body to you (your face, skin, eyes and hair from one photo). Apple Health supplies the inputs (food, activity, heart, sleep, workouts); **Progress** shows how they move your body, a **Today** checklist shows what's missing, and **What if…** shows you after a plan, predicted from your own data. See [ROADMAP.md](ROADMAP.md) for where it's going.

Everything runs on your machine; other devices reach it only over your Tailscale tailnet, and only when signed in as you. Camera video stays in memory; only measurements, a scale-free outline per useful frame (kept 45 days, at most 400 a day), one fitted body mesh per day (115 KB) and your face photo are stored; with Developer → reference images on, also up to 80 cropped frames of you.

## Quick start

Requires [uv](https://docs.astral.sh/uv/) and a camera that serves MJPEG or RTSP.

```sh
uv run app.py              # http://localhost:8765 (downloads MediaPipe models on first use)
uv sync --group fit        # optional: fit your real body shape (PyTorch + Anny, ~2.5 GB with Anny's cache)
tools/autostart.sh         # optional: start at login and keep running (macOS)
uv run python test_bodylab.py && node test_body.mjs
```

1. **Pair your iPhone** (Settings → scan the QR code). Apple Health and the phone supply your height, sex, age and units, and your history flows in.
2. Set the **camera URL** in Settings.
3. Just live your life in front of it. BodyLab measures every frame it can use — any angle, any distance, partly in view — and fuses them into one body per day. Views from different angles pin down depth, not just width.

Env: `PORT` (8765), `BODYLAB_DB` (default `data/bodylab.db`).

## iPhone: onboarding + Apple Health

`ios/` is the **BodyLab** app, the product: your 3D body, Today, Progress and What if… (the same web UI, inside a native shell), plus Apple Health sync and logging food, weight and workouts. Scan the pairing QR, allow Health, scan your face with the Face ID camera (Me → Scan your face: it shapes your 3D face and takes its photo), done. Everything else comes from the cameras, and you can tap any part to enter your own tape measurement: it's the truth for that day, and for a month the body fit must match it. It sends 90 days of data on open and in the background whenever Health changes; re-syncs update records rather than duplicating them. This Mac's web page (`/`) is the developer console: cameras, the body fit and skin bake, reference images, and reference measurements with where to take each (compared only with BodyLab's estimate from the same day, never used for your numbers). The app is `/?app=1`. Icon and launch mark: `uv run python tools/brand.py`.

| | Apple Health data |
|---|---|
| Profile | height, sex, date of birth |
| Body | weight, smart-scale body fat & lean mass, waist circumference |
| Fuel | calories, protein, carbs, fat, fiber, sugar, water, alcoholic drinks |
| Activity | active & resting energy, steps, exercise minutes, workouts (type, duration, heart rate, effort, calories) |
| Heart & recovery | resting & walking heart rate, HRV, VO₂ max, respiratory rate |
| Sleep | total, deep, REM |

Calories in minus active and resting energy gives your **energy balance**, so Insights compares predicted weight change (~7700 kcal per kg) with what the scale and camera measured.

```sh
tailscale serve --bg --https=8443 http://127.0.0.1:8765   # tailnet-only HTTPS: the iPhone syncs here, and you can open
                                                          # https://<this-mac>.<tailnet>.ts.net:8443 from your other devices
cd ios && BODYLAB_TEAM=<your team id> xcodegen && open BodyLabSync.xcodeproj   # then ⌘R on your iPhone
```

## Raspberry Pi camera

`pi/camstream.py` serves `http://<tailscale-ip>:8000/stream.mjpg`, bound to the Tailscale address only: a Pi camera via picamera2, or any Linux webcam via ffmpeg (`DEVICE`, `SIZE`, `FPS`). Up to three cameras at different angles (console → Cameras) see width and depth at once, so you needn't turn; each camera checks your face itself. Any MJPEG or RTSP stream works, including a Wi-Fi IP camera or a spare phone running an IP-camera app (no computer needed); keep them off the internet.

```sh
PI=you@raspberrypi
tailscale ssh $PI 'mkdir -p ~/bodylab-cam ~/.config/systemd/user'
tailscale ssh $PI 'cat > ~/bodylab-cam/camstream.py' < pi/camstream.py
tailscale ssh $PI 'cat > ~/.config/systemd/user/bodylab-cam.service' < pi/bodylab-cam.service
tailscale ssh $PI 'loginctl enable-linger && systemctl --user daemon-reload && systemctl --user enable --now bodylab-cam'
```

Mounted sideways? Set **Camera rotation** in Settings.

## How it works

- **Measuring:** [MediaPipe Pose](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker) finds 33 landmarks (2D and 3D) and a person mask per frame. Each frame yields whatever it can: widths across the mask at anatomical levels, scaled by your height (or, when your feet are out of frame, by your learned torso length), plus how far you're turned from the camera. Slices an arm or clothing would distort are skipped individually.
- **Fusing:** a torso slice is roughly an ellipse, whose silhouette width at yaw θ is 2√(a²cos²θ + b²sin²θ). Fitting that (robustly) to all of a day's widths gives both width and depth; limbs use medians. Circumferences come from the ellipses (scaled for torsos being boxier than ellipses; the neck from front views, since a turned neck's outline includes the jaw; men's waist at the navel); body fat from the US Navy formula.
- **Body shape:** with the `fit` group installed, [Anny](https://github.com/naver/anny) (Apache-2.0, MakeHuman's mesh) is fitted to all of a day's outlines at once (`fit.py`, about a minute on an M4's GPU): each frame gets its own pose and turn from MediaPipe's landmarks, one shape is shared, and parts cut off by the image's edge are ignored. It refits as frames arrive. With reference captures on, the same fit bakes your skin's fine detail (marks, hair, texture) from them onto the body's texture (`bake.py`), once they cover half the body. Circumferences are then tape-measured on the fitted mesh and replace the ellipse estimates; the 3D body is that mesh, shown in your own relaxed stance (from how you hold your arms on camera), and What if… reshapes it. Outlines leave overall volume loose (about ±10 %), so a weigh-in from the last few days pins it.
- **3D body:** [MakeHuman](http://www.makehumancommunity.org)'s CC0 base mesh, fitted in the browser (`static/body.js`): torso width/depth per level, limbs radially around their bones, heights to your landmarks, blended by the rig's skin weights. Hair, brows, lashes and eyes are MakeHuman proxies anchored to the mesh, so they follow the fit.
- **Your face:** the template head is rendered once and run through MediaPipe's face landmarker, giving each of 478 landmarks a spot on the skin texture. Your photo's landmarks then warp it, triangle by triangle, onto that texture (`static/look.js`), keeping the photo's fine detail but taking broad colour and shading from the skin, so studio lighting doesn't make it look pasted on. Skin tone and eye colour are read from the same photo; your hair is segmented to pick its colour and the closest haircut. An underwear toggle shows the bare body.
- **Only you, only real scans:** a frame counts only when one person is in view and your face has matched your photo twice ([OpenCV](https://github.com/opencv/opencv_zoo) YuNet + SFace), within a minute of the frame. Your face is re-checked every 2 s; a face that clearly isn't yours, a second person, or a room too dark to tell discards everything not yet confirmed. Frames count only standing (not sitting or crouching), and by default with a bare torso: MediaPipe's multiclass segmenter must see mostly skin, not clothes, from shoulders to hips (underwear is fine; covered legs drop out of the shape fit). Saved frames must also have human proportions and bare skin on each measured slice (checked against your face's skin tone in the same frame), since clothes inflate every measurement. The live preview is a silhouette unless you switch it to the camera image.
- **Vision check (optional, `uv sync --group vision`):** Cloudflare's [Clef-flash](https://huggingface.co/Cloudflare/clef-flash) decision model (Apache-2.0, on MLX, 10.7 GB, downloaded on first use) looks at an accepted frame every few seconds per camera and rules out what the rules above miss: not standing (sitting or lying in bed, on a table), under a blanket, clothes on, someone else in view, a child (`judge.py`). It gives a probability per answer, so close calls lean the right way. A ruled-out frame takes its camera's frames and reference captures within 2.5 s with it. The image stays on this computer and isn't kept. `uv run python judge.py` checks existing reference captures the same way.
- **Incomplete data:** today's running totals and partly logged food days are left out of averages; energy balance needs 14+ fully logged days out of 28; weight and body trends use a Theil–Sen line so one noisy reading can't swing them; Insights lists exactly which inputs are missing.
- **What if…:** a daily energy-balance simulation (Hall's energy densities, Forbes's fat/lean partitioning, maintenance that falls as you lose) with training and protein effects. Maintenance is learned from your logged food and weight trend; each body part changes the way yours has, once there's history. It refuses to go below essential body fat.
- **Anatomy:** breasts are shaped from bust and underbust (MakeHuman cup and firmness morphs); male genitals at an average size (MakeHuman's genital helper, smoothed) until the camera can measure them. The body shows nude; **censored** puts it in underwear cut from its own surface (trunks; briefs and a sports bra), so it fits any body exactly.
- `uv run python tools/build_assets.py` rebuilds `static/assets/` from the original sources.

Expect roughly ±3–4 body-fat points without calibration. Clothing, posture and lighting matter more than the math.

## License

MIT. Bundles [three.js](https://threejs.org) (MIT), [uqr](https://github.com/unjs/uqr) (MIT), assets derived from [MakeHuman](http://www.makehumancommunity.org) (CC0: mesh, skins, hair, eyes) and MediaPipe's canonical face-mesh triangulation (Apache-2.0); optionally installs Anny (Apache-2.0, CC0 data) and PyTorch (BSD); downloads MediaPipe pose, face and hair models (Apache-2.0) and OpenCV Zoo face detection/recognition models (MIT, Apache-2.0).
