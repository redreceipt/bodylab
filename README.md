# BodyLab

Track body composition by walking past a camera. BodyLab watches a network camera, measures your silhouette, estimates circumferences and body fat, and fits a realistic 3D body to you (your face, skin, eyes and hair from one photo). Click any part for its history and goals. Apple Health supplies the inputs (food, activity, heart, sleep, workouts), and an **inputs → outputs** view shows how they move your body.

Everything runs on your machine; other devices reach it only over your Tailscale tailnet, and only when signed in as you. Camera video stays in memory; only measurements (and your face photo, if you add one) are stored.

## Quick start

Requires [uv](https://docs.astral.sh/uv/) and a camera that serves MJPEG or RTSP.

```sh
uv run app.py              # http://localhost:8765 (downloads MediaPipe models on first use)
uv run python test_bodylab.py && node test_body.mjs
```

1. **Pair your iPhone** (Profile → scan the QR code). Apple Health fills in height, sex and age, and your history flows in. Or type your height.
2. Set the **camera URL** in Profile.
3. Just live your life in front of it. BodyLab measures every frame it can use — any angle, any distance, partly in view — and fuses them into one body per day. Views from different angles pin down depth, not just width.

Env: `PORT` (8765), `BODYLAB_DB` (default `data/bodylab.db`).

## iPhone: onboarding + Apple Health

`ios/` is a small SwiftUI app. Scan the pairing QR, allow Health, take a face photo, done. It sends 90 days of data on open and in the background whenever Health changes; re-syncs update records rather than duplicating them.

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

`pi/camstream.py` serves `http://<pi-tailscale-ip>:8000/stream.mjpg` using picamera2, bound to the Pi's Tailscale address only.

```sh
PI=you@raspberrypi
tailscale ssh $PI 'mkdir -p ~/bodylab-cam ~/.config/systemd/user'
tailscale ssh $PI 'cat > ~/bodylab-cam/camstream.py' < pi/camstream.py
tailscale ssh $PI 'cat > ~/.config/systemd/user/bodylab-cam.service' < pi/bodylab-cam.service
tailscale ssh $PI 'loginctl enable-linger && systemctl --user daemon-reload && systemctl --user enable --now bodylab-cam'
```

Mounted sideways? Set **Camera rotation** in Profile.

## How it works

- **Measuring:** [MediaPipe Pose](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker) finds 33 landmarks (2D and 3D) and a person mask per frame. Each frame yields whatever it can: widths across the mask at anatomical levels, scaled by your height (or, when your feet are out of frame, by your learned torso length), plus how far you're turned from the camera. Slices an arm or clothing would distort are skipped individually.
- **Fusing:** a torso slice is roughly an ellipse, whose silhouette width at yaw θ is 2√(a²cos²θ + b²sin²θ). Fitting that (robustly) to all of a day's widths gives both width and depth; limbs use medians. Circumferences come from the ellipses; body fat from the US Navy formula. Tape-measure any part once to calibrate it (past days are corrected too).
- **3D body:** [MakeHuman](http://www.makehumancommunity.org)'s CC0 base mesh, fitted in the browser (`static/body.js`): torso width/depth per level, limbs radially around their bones, heights to your landmarks, blended by the rig's skin weights. Hair, brows, lashes and eyes are MakeHuman proxies anchored to the mesh, so they follow the fit.
- **Your face:** the template head is rendered once and run through MediaPipe's face landmarker, giving each of 478 landmarks a spot on the skin texture. Your photo's landmarks then warp it, triangle by triangle, onto that texture (`static/look.js`), keeping the photo's fine detail but taking broad colour and shading from the skin, so studio lighting doesn't make it look pasted on. Skin tone and eye colour are read from the same photo; your hair is segmented to pick its colour and the closest haircut. An underwear toggle shows the bare body.
- **Only you, only real scans:** a scan is saved only when one person is in frame, your face matches your photo ([OpenCV](https://github.com/opencv/opencv_zoo) YuNet + SFace), the proportions are human, and — by default — your torso shows skin (checked against your face's skin tone in the same frame), since clothes inflate every measurement. The live preview is a silhouette unless you switch it to the camera image.
- **Incomplete data:** today's running totals and partly logged food days are left out of averages; energy balance needs 14+ fully logged days out of 28; weight and body trends use a Theil–Sen line so one noisy reading can't swing them; Insights lists exactly which inputs are missing.
- `uv run python tools/build_assets.py` rebuilds `static/assets/` from the original sources.

Expect roughly ±3–4 body-fat points without calibration. Clothing, posture and lighting matter more than the math.

## License

MIT. Bundles [three.js](https://threejs.org) (MIT), [uqr](https://github.com/unjs/uqr) (MIT), assets derived from [MakeHuman](http://www.makehumancommunity.org) (CC0: mesh, skins, hair, eyes) and MediaPipe's canonical face-mesh triangulation (Apache-2.0); downloads MediaPipe pose, face and hair models (Apache-2.0) and OpenCV Zoo face detection/recognition models (MIT, Apache-2.0).
